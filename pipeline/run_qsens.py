"""Deterministic per-channel quantization sensitivity: ground truth and cheap estimators (no random noise).

Ground truth (--mode gt). For output channel c of a q/k/v/o/gate/up/down projection, add only that row's real
INT4 weight error e_c = W_eff[c] - W[c] to the clean fp32 model and measure dloss (mean NLL per token):
  int4: W_eff = Q(W)                              (RTN, symmetric, one scale per --group input columns)
  rot : W_eff = R_out Q(R_out^T W R_in) R_in^T    (block Hadamard of size --group, seeds as src/wrapper.py, seed 0)
The channel's output moves by delta_t = x_t . e_c. Each module is also tested whole (channel = -1, every row's error).
Text: A = validation window 0, B = window 1 (two independent ground truths), C = windows 2.. (calibration).
Runs start at the module's decoder layer from cached clean inputs, so earlier layers aren't recomputed.

Estimators (--mode est), every channel, from one forward + 1 + K backward passes per window, divided by tokens:
  g1  : sum_t g_t delta_t, g = dloss/d(output) with the true next tokens (first-order Taylor term)
  qf  : mean_k (sum_t gs_t delta_t)^2, gs with labels sampled from the model: unbiased delta^T GGN delta
  fd  : mean_k sum_t gs_t^2 delta_t^2   (token-diagonal Fisher)
  fde : sum_t g_t^2 delta_t^2           (token-diagonal empirical Fisher)
  wf  : mean_k sum_j (sum_t gs_t x_tj)^2 e_cj^2   (diagonal weight Fisher, SqueezeLLM-style)
  fish: mean_k sum_t gs_t^2            (error-free channel Fisher)
  en  : sum_t delta_t^2                (output error energy, what GPTQ minimises)
  enw : ||e_c||^2                      (weight error energy)
Second-order Taylor: dloss ~ g1 + qf / 2. Module files add g1, qf with all cross-channel terms, and qf_diag = sum_c qf_c.

Usage (on the VM):
  python run_qsens.py --model HuggingFaceTB/SmolLM2-135M --group 64 --mode est --out-dir /workspace/results_qsens/smollm2
  python run_qsens.py --model HuggingFaceTB/SmolLM2-135M --group 64 --mode gt --out-dir /workspace/results_qsens/smollm2
"""
import argparse
import csv
import json
import os
import time

import torch
import torch.nn.functional as F
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import sdpa_attention

from src.data import load_wikitext2
from src.hadamard import BlockHadamardRotation
from src.quant import quantize_weight
from src.wrapper import PROJECTIONS
from run_sensitivity import find_targets

EST_FIELDS = ("g1", "qf", "fd", "fde", "wf", "fish", "en")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["est", "gt"], required=True)
    ap.add_argument("--group", type=int, default=128, help="weight-quant group size and Hadamard block size")
    ap.add_argument("--kinds", default="int4,rot", help="error kinds tested whole-module (gt) and estimated (est)")
    ap.add_argument("--chan-kinds", default="int4,rot", help="gt: error kinds tested per channel")
    ap.add_argument("--no-sym", action="store_true", help="gt: don't add the mirrored kind (int4 -> int4n) of each kind")
    ap.add_argument("--gt-file", default="gt.csv")
    ap.add_argument("--per-module", type=int, default=10, help="gt: random channels per module")
    ap.add_argument("--batch", type=int, default=8, help="gt: tests per forward")
    ap.add_argument("--calib", type=int, default=8, help="calibration windows (set C)")
    ap.add_argument("--fisher-samples", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-minutes", type=float, default=1e9, help="gt: stop after this many minutes (resumable)")
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def weight_error(mod, layer: int, kind: str, group: int) -> torch.Tensor:
    """E = W_eff - W in the original basis, fp32 [out, in]."""
    if kind.endswith("n"):  # mirrored error -e: averaging +e and -e cancels the first-order term exactly
        return -weight_error(mod, layer, kind[:-1], group)
    if kind.endswith("h"):  # half-size error: a purely quadratic effect falls to exactly 1/4
        return 0.5 * weight_error(mod, layer, kind[:-1], group)
    W = mod.weight.float()
    if kind == "int4":
        return quantize_weight(W, 4, group) - W
    seed = 2 * layer  # src/wrapper.py seeds with cfg.seed = 0: R_in uses 2N, R_out uses 2N + 1
    r_in = BlockHadamardRotation(W.shape[1], group, seed, W.device)
    r_out = BlockHadamardRotation(W.shape[0], group, seed + 1, W.device)
    Wp = r_out.apply(r_in.apply(W).T).T  # R_out^T W R_in
    unrot = lambda A: r_out.apply_T(r_in.apply_T(A).T).T  # R_out A R_in^T
    assert torch.allclose(unrot(Wp), W, atol=1e-4), "rotation round trip is not exact"
    return unrot(quantize_weight(Wp, 4, group)) - W


def nll_rows(head, h: torch.Tensor, ids: torch.LongTensor) -> torch.Tensor:
    """Summed next-token NLL per row in float64, logits made one row at a time."""
    # shortcut: plain lm_head on the decoder output; logit-softcapping models (Gemma 2) need their own head.
    return torch.stack([F.cross_entropy(head(h[i, :-1]).float(), ids[i, 1:], reduction="none").double().sum()
                        for i in range(h.shape[0])])


def write_csv(path: str, rows: list, append: bool = False) -> None:
    new = not (append and os.path.exists(path))
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def run_est(args, ids, model, targets) -> None:
    kinds, K = args.kinds.split(","), args.fisher_samples
    layer_of = {n: l for l, n, _ in targets}
    sets = {"A": ids[0:1], "B": ids[1:2], "C": ids[2:]}
    st = {"set": None, "pass": None}
    z = lambda n: torch.zeros(n, dtype=torch.float64, device="cuda")
    acc = {(s, k, n): {**{f: z(m.out_features) for f in EST_FIELDS}, "mod_g1": z(1), "mod_qf": z(1)}
           for s in sets for k in kinds for _, n, m in targets}

    def accumulate(name, mod, x, g):
        g = g.float()
        for k in kinds:
            E = weight_error(mod, layer_of[name], k, args.group)
            d = (x @ E.T).double()
            gd = g.double() * d
            a = acc[st["set"], k, name]
            if st["pass"] == "real":
                a["g1"] += gd.sum(0)
                a["fde"] += gd.pow(2).sum(0)
                a["en"] += d.pow(2).sum(0)
                a["mod_g1"] += gd.sum()
            else:
                s = gd.sum(0)
                a["qf"] += s.pow(2) / K
                a["fd"] += gd.pow(2).sum(0) / K
                a["mod_qf"] += s.sum().pow(2) / K
                a["fish"] += g.double().pow(2).sum(0) / K
                a["wf"] += ((g.T @ x).double().pow(2) * E.double().pow(2)).sum(1) / K

    def fwd_hook(name):
        def h(mod, inp, out):
            if out.requires_grad:
                x = inp[0].detach()[0].float()
                out.register_hook(lambda g, x=x, mod=mod: accumulate(name, mod, x, g[0]))
        return h

    hs = [m.register_forward_hook(fwd_hook(n)) for _, n, m in targets]
    gen = torch.Generator(device="cuda").manual_seed(0)
    n_tok, clean = {}, {}
    t0 = time.time()
    for s, windows in sets.items():
        st["set"], n_tok[s], clean[s] = s, windows.shape[0] * (windows.shape[1] - 1), 0.0
        for w in range(windows.shape[0]):
            win = windows[w:w + 1]
            emb = model.get_input_embeddings()(win).detach().requires_grad_()
            logits = model(inputs_embeds=emb, use_cache=False).logits[0, :-1].float()
            st["pass"] = "real"
            L = F.cross_entropy(logits, win[0, 1:], reduction="sum")
            clean[s] += L.item()
            torch.autograd.grad(L, emb, retain_graph=True)
            probs = torch.softmax(logits.detach(), dim=-1)
            for k in range(K):
                st["pass"] = k
                y = torch.multinomial(probs, 1, generator=gen).squeeze(1)
                torch.autograd.grad(F.cross_entropy(logits, y, reduction="sum"), emb, retain_graph=k < K - 1)
            del logits, probs, emb, L
            print(f"set {s} window {w + 1}/{windows.shape[0]} done, {time.time() - t0:.0f} s", flush=True)
    for h in hs:
        h.remove()

    for s in sets:
        for k in kinds:
            rows, mrows = [], []
            for layer, n, m in targets:
                a = {f: (v / n_tok[s]).tolist() for f, v in acc[s, k, n].items()}
                enw = weight_error(m, layer, k, args.group).double().pow(2).sum(1).tolist()
                short = n.split(".")[-1]
                rows += [{"layer": layer, "module": short, "channel": c, **{f: a[f][c] for f in EST_FIELDS}, "enw": enw[c]}
                         for c in range(m.out_features)]
                mrows.append({"layer": layer, "module": short, "g1": a["mod_g1"][0], "qf": a["mod_qf"][0],
                              "qf_diag": sum(a["qf"]), "en": sum(a["en"])})
            assert all(torch.isfinite(torch.tensor([r[f] for r in rows])).all() for f in EST_FIELDS), "non-finite estimator"
            write_csv(os.path.join(args.out_dir, f"est_{s}_{k}.csv"), rows)
            write_csv(os.path.join(args.out_dir, f"estmod_{s}_{k}.csv"), mrows)
    meta = {"args": vars(args), "clean_loss": {s: clean[s] / n_tok[s] for s in sets}, "n_tok": n_tok,
            "wall_s": time.time() - t0, "torch": torch.__version__, "transformers": transformers.__version__}
    json.dump(meta, open(os.path.join(args.out_dir, "est_meta.json"), "w"), indent=1)
    print(f"est done: clean loss {meta['clean_loss']} | {meta['wall_s']:.0f} s", flush=True)


def make_hook(chunk, W, E):
    def hook(mod, inp, out):
        x = inp[0]
        for j, (k, c) in enumerate(chunk):
            if k == "noop":
                continue
            r = slice(j * W, (j + 1) * W)
            if c < 0:
                out[r] += x[r] @ E[k].T
            else:
                out[r, :, c] += x[r] @ E[k][c]
        return out
    return hook


@torch.no_grad()
def run_gt(args, ids, model, targets) -> None:
    kinds, chan_kinds = args.kinds.split(","), args.chan_kinds.split(",")
    if not args.no_sym:
        kinds, chan_kinds = [x for k in kinds for x in (k, k + "n")], [x for k in chan_kinds for x in (k, k + "n")]
    ids = ids[:2]
    W, T, Bt = ids.shape[0], ids.shape[1], args.batch
    dec, head = model.get_decoder(), model.get_output_embeddings()
    all_layers = dec.layers

    H = {}  # clean input to each decoder layer
    def pre(i):
        def h(mod, a, kw):
            H[i] = (a[0] if a else kw["hidden_states"]).detach().clone()
        return h
    hs = [lay.register_forward_pre_hook(pre(i), with_kwargs=True) for i, lay in enumerate(all_layers)]
    clean = nll_rows(head, dec(input_ids=ids, use_cache=False).last_hidden_state, ids)
    for h in hs:
        h.remove()

    def from_layer(l, h):
        dec.layers = all_layers[l:]
        try:
            return dec(inputs_embeds=h, use_cache=False).last_hidden_state
        finally:
            dec.layers = all_layers

    rep = ids.repeat(Bt, 1)
    out_path = os.path.join(args.out_dir, args.gt_file)
    done = set()
    if os.path.exists(out_path):
        done = {(int(r["layer"]), r["module"]) for r in csv.DictReader(open(out_path))}
    meta_path = os.path.join(args.out_dir, args.gt_file.replace(".csv", "_meta.json"))
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"noop_max_abs": 0.0, "start_check": {}}
    meta.update({"args": vars(args), "clean_loss_rows": (clean / (T - 1)).tolist(),
                 "torch": torch.__version__, "transformers": transformers.__version__})
    t0 = time.time()
    layers = sorted({layer for layer, _, _ in targets})
    # Shuffled layer order, so a run cut short by --max-minutes still covers the whole depth.
    for l in [layers[i] for i in torch.randperm(len(layers), generator=torch.Generator().manual_seed(0)).tolist()]:
        mods = [(mi, n, m) for mi, (layer, n, m) in enumerate(targets) if layer == l and (l, n.split(".")[-1]) not in done]
        if not mods:
            continue
        if (time.time() - t0) / 60 > args.max_minutes:
            print(f"time limit reached before layer {l}", flush=True)
            break
        hrep = H[l].repeat(Bt, 1, 1)
        base = nll_rows(head, from_layer(l, hrep), rep)
        # Starting at layer l from cached inputs must reproduce the full clean loss.
        gap = (base.view(Bt, W)[0] - clean).abs().max().item() / (T - 1)
        meta["start_check"][str(l)] = gap
        assert gap < 1e-5, f"layer {l}: partial forward differs from full by {gap}"
        for mi, n, m in mods:
            t1 = time.time()
            E = {k: weight_error(m, l, k, args.group) for k in set(kinds) | set(chan_kinds)}
            chans = sorted(torch.randperm(m.out_features, generator=torch.Generator().manual_seed(mi))[:args.per_module].tolist())
            tests = [(k, -1) for k in kinds] + [(k, c) for k in chan_kinds for c in chans]
            res = {}
            for b in range(0, len(tests), Bt):
                chunk = tests[b:b + Bt]
                chunk = chunk + [("noop", 0)] * (Bt - len(chunk))
                hd = m.register_forward_hook(make_hook(chunk, W, E))
                try:
                    d = ((nll_rows(head, from_layer(l, hrep), rep) - base) / (T - 1)).view(Bt, W).tolist()
                finally:
                    hd.remove()
                for t, dv in zip(chunk, d):
                    if t[0] == "noop":
                        meta["noop_max_abs"] = max(meta["noop_max_abs"], *map(abs, dv))
                    else:
                        res[t] = dv
            short = n.split(".")[-1]
            write_csv(out_path, [{"layer": l, "module": short, "kind": k, "channel": c, "d_A": res[k, c][0], "d_B": res[k, c][1]}
                                 for k, c in tests], append=True)
            json.dump(meta, open(meta_path, "w"), indent=1)
            print(f"{time.strftime('%H:%M:%S')} L{l} {short}: {len(tests)} tests, {time.time() - t1:.1f} s | "
                  f"noop max {meta['noop_max_abs']:.1e} | start gap {gap:.1e}", flush=True)
    print("GT DONE", flush=True)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = load_wikitext2(tok, split="validation", seq_len=args.seq_len, max_windows=2 + args.calib).cuda()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    targets = find_targets(model, PROJECTIONS)
    print(f"{args.model}: {len(targets)} modules, {sum(m.out_features for _, _, m in targets)} channels, "
          f"{ids.shape[0]} windows x {ids.shape[1]}", flush=True)
    # fp32 + GQA (enable_gqa) sends SDPA to the math kernel, which builds the full T x T attention matrix
    # (5 GB per call). Repeating K/V instead and forcing the memory-efficient kernel keeps fp32 exact and small.
    sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        (run_est if args.mode == "est" else run_gt)(args, ids, model, targets)


if __name__ == "__main__":
    main()
