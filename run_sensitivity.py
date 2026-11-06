"""Per-channel ground truth on an unquantized model.

For every output channel of every q/k/v/o/gate/up/down projection, run these tests on a fixed token slice:
  zero  : set the channel's output to 0                    -> importance
  mean  : set it to its mean over the slice                -> importance without the offset
  noise : add +z and -z, Gaussian of std sigma, and average the pair (--draws pairs) -> sensitivity,
          score = mean dloss / sigma^2. The +/- pair cancels the random first-order term g.z exactly.
sigma = --alpha x the module's output RMS, the same for every channel of that module.
Each result is dloss = (loss with the change) - (clean loss), paired row by row in identical batch shapes.

Usage (on the VM):
  python run_sensitivity.py --model HuggingFaceTB/SmolLM2-135M --out-dir /workspace/results_sens/smollm2_bf16
  python run_sensitivity.py ... --sample-per-module 2 --dtype fp32   # pilot on a few channels
"""
import argparse
import csv
import json
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_wikitext2
from src.wrapper import PROJECTIONS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--split", default="validation", help="not test: groupings chosen from these scores are judged on test")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16, help="tests per forward pass")
    ap.add_argument("--alpha", type=float, default=0.1, help="noise std as a fraction of the module's output RMS")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--sample-per-module", type=int, default=None, help="pilot: this many random channels per module")
    ap.add_argument("--layers", default=None, help="comma list of layer indices (default: all)")
    ap.add_argument("--projections", default=",".join(PROJECTIONS))
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def find_targets(model: nn.Module, projections):
    """[(layer, name, module)] for the decoder-layer projections, in model order (same rule as wrap_model)."""
    out = []
    for name, mod in model.named_modules():
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if m and isinstance(mod, nn.Linear) and name.split(".")[-1] in projections:
            out.append((int(m.group(1)), name, mod))
    return out


@torch.no_grad()
def row_nll(model, ids: torch.LongTensor) -> torch.Tensor:
    """Summed next-token NLL per row, with the logits made one row at a time (a batch of full logits would not fit)."""
    # shortcut: plain lm_head on the decoder output; models with logit softcapping (Gemma 2) need their own head.
    h = model.get_decoder()(input_ids=ids, use_cache=False).last_hidden_state
    head = model.get_output_embeddings()
    return torch.stack([F.cross_entropy(head(h[i, :-1]).float(), ids[i, 1:], reduction="sum") for i in range(h.shape[0])])


def make_hook(tests, W, mean, sigma, noise):
    def hook(mod, inp, out):
        for j, (kind, c, k) in enumerate(tests):
            r = slice(j * W, (j + 1) * W)
            if kind == "zero":
                out[r, :, c] = 0
            elif kind == "mean":
                out[r, :, c] = mean[c].to(out.dtype)
            elif kind in ("noise+", "noise-"):
                out[r, :, c] += ((sigma if kind == "noise+" else -sigma) * noise[j]).to(out.dtype)
        return out
    return hook


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = load_wikitext2(tok, split=args.split, seq_len=args.seq_len, max_windows=args.windows).cuda()
    W, T, B = ids.shape[0], ids.shape[1], args.batch
    n_tok = W * (T - 1)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda").eval()
    targets = find_targets(model, args.projections.split(","))
    print(f"{len(targets)} modules, {sum(m.out_features for _, _, m in targets)} output channels, "
          f"{W} windows x {T} tokens, {args.dtype}", flush=True)

    # Clean pass: per-channel mean and per-module output RMS on the slice.
    sums, sqs = {}, {}
    def stat_hook(name):
        def h(mod, inp, out):
            o = out.double().reshape(-1, out.shape[-1])
            sums[name], sqs[name] = o.sum(0), o.pow(2).sum(0)
        return h
    hs = [m.register_forward_hook(stat_hook(n)) for _, n, m in targets]
    clean = row_nll(model, ids).sum().item() / n_tok
    for h in hs:
        h.remove()
    n_rows = W * T
    mean = {n: (sums[n] / n_rows).float() for _, n, _ in targets}
    ch_rms = {n: (sqs[n] / n_rows).sqrt().float() for _, n, _ in targets}
    sigma = {n: args.alpha * (sqs[n].sum() / (n_rows * sqs[n].numel())).sqrt().item() for _, n, _ in targets}

    # Baseline in the exact batch shape every test uses, so each test is paired with its own row position.
    rep = ids.repeat(B, 1)
    base = row_nll(model, rep).view(B, W).sum(1)
    meta = {"args": vars(args), "clean_loss": clean, "baseline_rows": (base / n_tok).tolist(), "n_tok": n_tok,
            "sigma": sigma, "torch": torch.__version__, "transformers": transformers.__version__, "noop_max_abs": {}}
    meta_path = os.path.join(args.out_dir, "meta.json")
    json.dump(meta, open(meta_path, "w"))
    print(f"clean loss {clean:.5f} | batched baseline rows {base.min().item() / n_tok:.6f}..{base.max().item() / n_tok:.6f}", flush=True)

    t_start = time.time()
    layers = sorted({l for l, _, _ in targets})
    if args.layers:
        layers = [l for l in layers if l in {int(i) for i in args.layers.split(",")}]
    for layer in layers:
        path = os.path.join(args.out_dir, f"layer{layer:02d}.csv")
        if os.path.exists(path):
            print(f"layer {layer}: done already, skipping", flush=True)
            continue
        rows, noop_max = [], 0.0
        for mi, (l, name, mod) in enumerate(targets):
            if l != layer:
                continue
            out_f = mod.out_features
            # Sample seeded by module index, so a resumed or partial run picks the same channels.
            chans = (torch.randperm(out_f, generator=torch.Generator().manual_seed(mi))[: args.sample_per_module]
                     .sort().values.tolist() if args.sample_per_module else range(out_f))
            tests = [(kind, c, k) for c in chans
                     for kind, k in [("zero", 0), ("mean", 0)]
                     + [(s, k) for k in range(args.draws) for s in ("noise+", "noise-")]]
            res = {}
            t0 = time.time()
            for b in range(0, len(tests), B):
                chunk = tests[b:b + B]
                chunk = chunk + [("noop", -1, 0)] * (B - len(chunk))
                # Noise fixed by (module index, channel, draw), so any rerun repeats the same draw.
                noise = [torch.randn(W, T, generator=torch.Generator().manual_seed(mi * 10_000_000 + c * 100 + k)).cuda()
                         if kind.startswith("noise") else None for kind, c, k in chunk]
                handle = mod.register_forward_hook(make_hook(chunk, W, mean[name], sigma[name], noise))
                try:
                    d = ((row_nll(model, rep).view(B, W).sum(1) - base) / n_tok).tolist()
                finally:
                    handle.remove()
                for (kind, c, k), dv in zip(chunk, d):
                    if kind == "noop":
                        noop_max = max(noop_max, abs(dv))
                    else:
                        res[(kind, c, k)] = dv
            for c in chans:
                rows.append({"layer": layer, "module": name.split(".")[-1], "channel": c, "sigma": sigma[name],
                             "ch_mean": mean[name][c].item(), "ch_rms": ch_rms[name][c].item(),
                             "d_zero": res[("zero", c, 0)], "d_mean": res[("mean", c, 0)],
                             **{f"d_noise{k}": (res[("noise+", c, k)] + res[("noise-", c, k)]) / 2
                                for k in range(args.draws)},
                             **{f"d_noise{k}{s[-1]}": res[(s, c, k)] for k in range(args.draws) for s in ("noise+", "noise-")}})
            print(f"  {name}: {len(chans)} channels, {time.time() - t0:.1f} s", flush=True)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
        meta["noop_max_abs"][str(layer)] = noop_max
        json.dump(meta, open(meta_path, "w"))
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} layer {layer} done | noop max |dloss| {noop_max:.2e} | "
              f"elapsed {time.time() - t_start:.0f} s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
