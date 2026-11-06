"""Offline tests of the Hadamard mixing predictions on est CSVs (per output channel: g1, qf, fish, en, ey).
P1: under rot4 the output-error energy en is flat within each 128-row output block; under pre4 it is not.
P2: under rot4, qf (curvature) ~ fish_c * en_block, so ranking is loss-side; under pre4 the error side matters.
P3: g1 is a random-sign mix: within a block, g1 sign is ~ independent of everything weight-side."""
import sys
import numpy as np
import pandas as pd

B = 128
rk = lambda s: s.rank()
sp = lambda a, b: rk(a).corr(rk(b))
for path in sys.argv[1:]:
    print(f"\n### {path}")
    for kind in ("pre4", "rot4"):
        d = pd.read_csv(f"{path}/est_{kind}.csv")
        d["blk"] = d.channel // B
        key = ["layer", "module", "blk"]
        grp = d.groupby(key)
        cv = (grp.en.std() / grp.en.mean())  # coefficient of variation of error energy within an output block
        enb = grp.en.transform("mean")
        d["pred_q"] = d.fish * enb
        d["pred_q_own"] = d.fish * d.en
        # within-block rank correlations, averaged over blocks
        wb = lambda a, b: grp.apply(lambda x: sp(x[a], x[b])).mean()
        print(f"{kind}: en CV within block median {cv.median():.3f} (p90 {cv.quantile(.9):.3f}) | "
              f"spearman(qf, fish*en_block) {sp(d.qf, d.pred_q):.3f}, (qf, fish*en_own) {sp(d.qf, d.pred_q_own):.3f}, (qf, fish) {sp(d.qf, d.fish):.3f}, (qf, en) {sp(d.qf, d.en):.3f} | "
              f"within-block: (qf,fish) {wb('qf','fish'):.3f} (qf,en) {wb('qf','en'):.3f} | frac g1>0 {np.mean(d.g1 > 0):.3f}")
