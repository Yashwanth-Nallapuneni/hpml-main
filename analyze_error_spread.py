"""Does rotation spread per-channel error evenly? Reads results_snapshot_v2 JSONs only (no GPU, no numpy).
Run from team20-hadamard/: python3 analyze_error_spread.py"""
import glob
import json
import statistics as st

ROOT = "results_snapshot_v2"


def load(p):
    return json.load(open(p))["layers"]


def ranks(x):
    r = [0] * len(x)
    for i, j in enumerate(sorted(range(len(x)), key=x.__getitem__)):
        r[j] = i
    return r


def spearman(x, y):
    rx, ry, mid = ranks(x), ranks(y), (len(x) - 1) / 2
    return sum((a - mid) * (b - mid) for a, b in zip(rx, ry)) / sum((a - mid) ** 2 for a in rx)


for model in ["Llama-3.2-1B", "gemma-3-1b-pt"]:
    a = load(glob.glob(f"{ROOT}/{model}/int4_seed0_*.json")[0])
    b = load(glob.glob(f"{ROOT}/{model}/int4_rot_seed1_*.json")[0])
    cv_a, cv_b, rho_a, rho_b, small, big = [], [], [], [], [], []
    for n in a:
        ea, eb, r = a[n]["ch_rms_err"], b[n]["ch_rms_err"], a[n]["ch_rms_ref"]
        cv_a.append(st.pstdev(ea) / st.mean(ea))
        cv_b.append(st.pstdev(eb) / st.mean(eb))
        rho_a.append(spearman(ea, r))
        rho_b.append(spearman(eb, r))
        s = sorted(r)
        q1, q3 = s[len(s) // 4], s[3 * len(s) // 4]
        worse = [y > x for x, y in zip(ea, eb)]
        small.append(st.mean([w for w, ri in zip(worse, r) if ri <= q1]))
        big.append(st.mean([w for w, ri in zip(worse, r) if ri >= q3]))
    m = st.median
    print(f"{model} W4A16 ({len(a)} modules; int4 seed0 vs int4_rot seed1), medians over modules:")
    print(f"  spread of per-channel error (std/mean): int4 {m(cv_a):.3f}  rot {m(cv_b):.3f}")
    print(f"  Spearman(channel error, channel size):  int4 {m(rho_a):.3f}  rot {m(rho_b):.3f}")
    print(f"  share of channels worse with rotation: smallest quarter {m(small):.3f}  largest quarter {m(big):.3f}")
