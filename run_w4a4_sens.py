"""Channel sensitivity in the CAT paper's W4A4KV4 setting (Llama-3.2-1B-Instruct), built on run_cat_repro.py.

1. Sensitivity-weighted CAT (SW-CAT). If the loss change is ~ 1/2 sum_o s_o E[dy_o^2] (s_o = output-channel Fisher), both
   noise terms of CAT's SQNR model become tr(M Sx M) tr(M^-1 W^T S W M^-1), so the optimal M is Eq. 7 with Sw = W^T S W.
   --sw fisher uses s_o = E[g_o^2] (sampled labels); --sw shuffled permutes s within each module (control: same spread of
   weights, no information); --sw none is plain CAT.
2. Mackin's exemption in W4A4: on top of CAT, the top-k output channels per module are recomputed in fp (fp input, fp row),
   chosen by a cheap estimator of each channel's real W4A4 error (g1 + qf/2, as in run_mackin.py) or by baselines / random.
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

import run_cat_repro as R

SEL = {"pred": lambda e: e["g1"] + e["qf"] / 2, "curv": lambda e: e["qf"] / 2, "fisher": lambda e: e["fish"],
       "err": lambda e: e["en"], "energy": lambda e: e["ey"]}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--sw", default="none,fisher,shuffled", help="comma list of none, fisher, shuffled")
    ap.add_argument("--cat-k", type=int, default=128)
    ap.add_argument("--seeds", default="0", help="comma list; calibration sample seeds")
    ap.add_argument("--fisher-windows", type=int, default=32, help="calibration windows used for Fisher / estimator passes")
    ap.add_argument("--fisher-samples", type=int, default=4)
    ap.add_argument("--est-len", type=int, default=1024, help="tokens per window for the channel estimator (2048 ran out of GPU memory on the L4)")
    ap.add_argument("--exempt-k", default="", help="comma list of channels per module to exempt (empty = skip part 2)")
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--eval-windows", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # quantization set-up fixed to the paper's (see run_cat_repro.py)
    a.w_bits, a.a_bits, a.kv_bits, a.w_range, a.calib, a.n_calib, a.seq_len, a.k1_formula = 4, 4, 4, "l2.4", "dclm", 128, 2048, "eq8"
    return a


def output_fisher(model, targets, ids, K: int) -> dict:
    """s_o = mean over tokens of E_y~p[g_o^2], g = dloss/d(module output), fp model, K sampled labels per window."""
    acc, n = {}, 0
    def fh(name):
        def h(mod, inp, out):
            if out.requires_grad:
                out.register_hook(lambda g: acc.__setitem__(name, acc.get(name, 0) + g[0].float().pow(2).sum(0).double() / K))
        return h
    hs = [m.register_forward_hook(fh(name)) for _, name, m in targets]
    gen = torch.Generator(device="cuda").manual_seed(0)
    for i in range(ids.shape[0]):
        win = ids[i:i + 1].cuda()
        emb = model.get_input_embeddings()(win).detach().requires_grad_()
        logits = model(inputs_embeds=emb, use_cache=False).logits[0, :-1].float()
        probs = torch.softmax(logits.detach(), -1)
        for k in range(K):
            y = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            torch.autograd.grad(F.cross_entropy(logits, y, reduction="sum"), emb, retain_graph=k < K - 1)
        n += win.shape[1] - 1
        del logits, probs, emb
    for h in hs:
        h.remove()
    return {k: v / n for k, v in acc.items()}


def channel_estimates(model, targets, qmods, ids, K: int) -> dict:
    """Per output channel, for the real W4A4 error d = Qlinear(x) - x W^T of each module (fp model, fp input):
    g1 = sum g d (true labels), qf = mean_k (sum g_k d)^2 (sampled labels), fish = mean_k sum g_k^2, en = sum d^2, ey = sum y^2."""
    st = {"pass": None}
    acc = {name: {f: 0 for f in ("g1", "qf", "fish", "en", "ey")} for _, name, _ in targets}
    def fh(name):
        def h(mod, inp, out):
            if not out.requires_grad:
                return
            with torch.no_grad():  # both paths in fp32: a difference of two bf16 outputs would lose a few % of d
                qm, x = qmods[name], inp[0].detach()[0].float()
                h = x @ qm.A if qm.A is not None else x
                b = None if mod.bias is None else mod.bias.float()
                d = F.linear(R.quant_a(h, qm.a_bits), qm.Wq, qm.bias) - F.linear(x, mod.weight.float(), b)  # fp32, [T, out]
                a = acc[name]
                a["en"] = a["en"] + d.double().pow(2).sum(0)
                a["ey"] = a["ey"] + out.detach()[0].double().pow(2).sum(0)
            def gh(g, d=d, a=acc[name]):
                gd = (g[0].float() * d).double()
                if st["pass"] == "real":
                    a["g1"] = a["g1"] + gd.sum(0)
                else:
                    a["qf"] = a["qf"] + gd.sum(0).pow(2) / K
                    a["fish"] = a["fish"] + g[0].double().pow(2).sum(0) / K
            out.register_hook(gh)
        return h
    hs = [m.register_forward_hook(fh(name)) for _, name, m in targets]
    gen = torch.Generator(device="cuda").manual_seed(1)
    n = 0
    for i in range(ids.shape[0]):
        win = ids[i:i + 1].cuda()
        emb = model.get_input_embeddings()(win).detach().requires_grad_()
        logits = model(inputs_embeds=emb, use_cache=False).logits[0, :-1].float()
        st["pass"] = "real"
        torch.autograd.grad(F.cross_entropy(logits, win[0, 1:], reduction="sum"), emb, retain_graph=True)
        probs = torch.softmax(logits.detach(), -1)
        for k in range(K):
            st["pass"] = k
            y = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            torch.autograd.grad(F.cross_entropy(logits, y, reduction="sum"), emb, retain_graph=k < K - 1)
        n += win.shape[1] - 1
        del logits, probs, emb
    for h in hs:
        h.remove()
    return {name: {f: (v / n).cpu() for f, v in a.items()} for name, a in acc.items()}


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = R.AutoTokenizer.from_pretrained(args.model)
    model = R.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                   attn_implementation="kvq_sdpa").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    targets = R.find_targets(model)
    test = R.load_wikitext2(tok, split="test", seq_len=args.seq_len, max_windows=args.eval_windows)
    K = args.cat_k

    def record(**row):
        row.update(test_windows=test.shape[0], cat_k=K, model=args.model)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    for seed in [int(s) for s in args.seeds.split(",")]:
        calib = R.calib_windows(tok, args, seed)
        Sx = R.input_covariances(model, targets, calib, K)
        need_s = any(v != "none" for v in args.sw.split(","))
        s_out = output_fisher(model, targets, calib[: args.fisher_windows], args.fisher_samples) if need_s else None
        for sw in args.sw.split(","):
            Sw = {}
            for li, name, m in targets:
                key = (li, R.SHARE.get(name.split(".")[-1]))
                if key[1] is None:
                    continue
                W = m.weight.data.double()
                if sw != "none":
                    s = s_out[name]
                    if sw == "shuffled":
                        s = s[torch.randperm(s.numel(), generator=torch.Generator().manual_seed(seed * 7919 + li)).to(s.device)]
                    W = W * s.sqrt()[:, None]  # W^T S W = (S^1/2 W)^T (S^1/2 W)
                W = W.view(m.out_features, -1, K)
                Sw[key] = Sw.get(key, 0) + torch.einsum("obi,obj->bij", W, W)
            Mx = {key: R.cat_matrix(Sx[key], Sw[key], K, "eq8") for key in Sx}
            t0 = time.time()
            orig = R.build(model, targets, "cat", args, Mx)
            record(part="swcat", sw=sw, seed=seed, ppl=R.ppl(model, test, args.batch), seconds=round(time.time() - t0, 1))

            if sw == "none" and args.exempt_k and seed == int(args.seeds.split(",")[0]):
                qmods = {name: model.get_submodule(name) for _, name, _ in targets}
                R.restore(model, orig)  # estimator runs on the fp model with each module's W4A4 error injected locally
                del Mx, Sw
                Mx = None
                torch.cuda.empty_cache()
                est = channel_estimates(model, targets, qmods, calib[: args.fisher_windows, : args.est_len], args.fisher_samples)
                torch.save(est, os.path.splitext(args.out)[0] + f"_est_seed{seed}.pt")
                for _, name, _ in targets:  # swap the same QLinears back in
                    parent, _, child = name.rpartition(".")
                    setattr(model.get_submodule(parent), child, qmods[name])
                R.KV["bits"], R.KV["H"] = args.kv_bits, R.sylvester_hadamard(model.config.head_dim).cuda().float()
                fpw = {name: orig[name].weight.data for name in orig}
                ex = {}
                def hook(name):
                    def h(mod, inp, out):
                        if name in ex:
                            idx = ex[name]
                            b = None if orig[name].bias is None else orig[name].bias[idx]
                            out[..., idx] = F.linear(inp[0], fpw[name][idx], b).to(out.dtype)
                        return out
                    return h
                hs = [qmods[name].register_forward_hook(hook(name)) for _, name, _ in targets]
                for k in [int(x) for x in args.exempt_k.split(",")]:
                    sels = {s: {name: f(est[name]).topk(k).indices.cuda() for _, name, _ in targets} for s, f in SEL.items()}
                    for r in range(args.random_seeds):
                        sels[f"random{r}"] = {name: torch.randperm(m.out_features, generator=torch.Generator().manual_seed(r * 100_003 + i))[:k].cuda()
                                              for i, (_, name, m) in enumerate(targets)}
                    for s, sel in sels.items():
                        ex.clear()
                        ex.update(sel)
                        t0 = time.time()
                        record(part="exempt", sw=sw, seed=seed, k=k, selector=s, ppl=R.ppl(model, test, args.batch),
                               seconds=round(time.time() - t0, 1))
                ex.clear()
                for h in hs:
                    h.remove()
                R.restore(model, orig)
                del qmods, est, fpw, sels  # the QLinears held here kept ~10 GB alive and the next build ran out of GPU memory
                torch.cuda.empty_cache()
            R.restore(model, orig)
            del Mx
    print("W4A4 SENS DONE", flush=True)


if __name__ == "__main__":
    main()
