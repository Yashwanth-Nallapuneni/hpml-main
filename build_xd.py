"""Multi-domain sensitivity estimator: combine per-domain est/grad files into one eval directory.

score s_rob = beta[module] * mean_d g1_d + qf_mix / 2
  g1_d   first-order term from --mode grad on calibration domain d (many windows)
  qf_mix curvature term (sampled-label GGN quadratic) averaged over the --qf domains (--mode est)
  beta   leave-one-domain-out slope: how well the mean g1 of the other domains predicts a held-out domain's g1,
         per (layer, module), clipped to [0, 1]. Near 0 when g1 is domain-specific, so the score falls back to curvature.
Also writes s_c4 (single-domain baseline: C4 g1 + C4 qf/2, the old recipe) and g1 = plain domain mean (pred_rot_big = naive mix).
usage: build_xd.py MODEL_DIR OUT_DIR --grad c4,pilecc,... --qf c4,github,...
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

K = ["layer", "module", "channel"]
EST = ["g1", "qf", "fish", "en", "ey"]

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("out_dir")
ap.add_argument("--grad", required=True)
ap.add_argument("--qf", required=True)
a = ap.parse_args()
G, Q = a.grad.split(","), a.qf.split(",")
assert len(G) >= 2 and "c4" in G and "c4" in Q
os.makedirs(a.out_dir, exist_ok=True)

for kind in ("pre4", "rot4", "pre8"):  # run_eval reads pre4 and rot4; pre8 kept for completeness
    fs = [pd.read_csv(f"{a.model_dir}/est_{d}/est_{kind}.csv") for d in Q]
    for f in fs[1:]:
        assert f[K].equals(fs[0][K])
    mix = fs[0][K].copy()
    for c in EST:
        mix[c] = np.mean([f[c].values for f in fs], axis=0)
    mix.to_csv(f"{a.out_dir}/est_{kind}.csv", index=False)
    if kind == "rot4":
        qmix, qc4 = mix.qf.values / 2, fs[Q.index("c4")].qf.values / 2

gs = [pd.read_csv(f"{a.model_dir}/grad_{d}/est_grad_rot4.csv") for d in G]
for g in gs[1:]:
    assert g[K].equals(gs[0][K])
assert gs[0][K].equals(mix[K])
g = np.stack([x.g1.values for x in gs])  # [domains, channels]
out = gs[0][K].copy()
out["g1"] = g.mean(0)
grp = out.groupby(["layer", "module"]).indices
beta = np.zeros(len(out))
betas = {}
for key, idx in grp.items():
    sl = []
    for j in range(len(G)):
        x, y = np.delete(g[:, idx], j, 0).mean(0), g[j, idx]
        assert x.var(ddof=1) > 0, f"{key}: constant g1"
        sl.append(np.cov(x, y)[0, 1] / x.var(ddof=1))
    b = float(np.clip(np.mean(sl), 0, 1))
    beta[idx], betas[f"{key[0]}.{key[1]}"] = b, b
out["s_rob"] = beta * out.g1.values + qmix
out["s_c4"] = g[G.index("c4")] + qc4
assert np.isfinite(out[["g1", "s_rob", "s_c4"]].values).all()
out.to_csv(f"{a.out_dir}/est_grad_rot4.csv", index=False)

bv = np.array(list(betas.values()))
json.dump({"grad_domains": G, "qf_domains": Q, "beta": betas, "beta_mean": bv.mean(), "beta_median": float(np.median(bv)),
           "g1_domain_corr": np.corrcoef(g).round(4).tolist()}, open(f"{a.out_dir}/xd_meta.json", "w"), indent=1)
print(f"built {a.out_dir}: beta mean {bv.mean():.3f} median {np.median(bv):.3f}")
print("g1 cross-domain corr:\n", np.corrcoef(g).round(3))
