#!/usr/bin/env python3
"""Deep analysis of per-channel sensitivity CSVs (stdlib only). Run from team20-hadamard/."""
import csv, glob, math, os, statistics as st
from collections import defaultdict

D = "results_sens_snapshot/smollm2_fp32_s20"
TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BANDS = [(0, 4), (5, 14), (15, 24), (25, 29)]

# ---------- helpers ----------
def ranks(x):  # average ranks for ties
    idx = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and x[idx[j + 1]] == x[idx[i]]: j += 1
        for k in range(i, j + 1): r[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return float("nan") if sa == 0 or sb == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)
def spear(a, b): return pearson(ranks(a), ranks(b))
def q(x, p):  # linear-interp quantile
    s = sorted(x); k = (len(s) - 1) * p; f = int(math.floor(k)); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)
med = lambda x: q(x, 0.5)
def gini(x):
    s = sorted(x); n = len(s); t = sum(s)
    return float("nan") if t == 0 else (2 * sum((i + 1) * v for i, v in enumerate(s)) / (n * t)) - (n + 1) / n
def sb(r, k=3): return k * r / (1 + (k - 1) * r)  # Spearman-Brown
def H(t): print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)
f = lambda v, n=4: "nan" if v != v else f"{v:.{n}f}"
e = lambda v: "nan" if v != v else f"{v:.3e}"

# ---------- load ----------
R = []
for fn in sorted(glob.glob(f"{D}/layer*.csv")):
    for r in csv.DictReader(open(fn)):
        o = {k: (r[k] if k == "module" else float(r[k])) for k in r}
        o["layer"] = int(o["layer"]); o["channel"] = int(o["channel"])
        o["dn"] = [o[f"d_noise{k}"] for k in range(3)]
        o["s"] = sum(o["dn"]) / 3 / o["sigma"] ** 2
        o["fo"] = [abs(o[f"d_noise{k}+"] - o[f"d_noise{k}-"]) / 2 for k in range(3)]  # first-order
        o["so"] = [abs(x) for x in o["dn"]]  # second-order (pair mean)
        o["abs_mean"] = abs(o["ch_mean"])
        R.append(o)
N = len(R); mods = defaultdict(list)
for o in R: mods[(o["layer"], o["module"])].append(o)
bytype = {t: [o for o in R if o["module"] == t] for t in TYPES}
band_of = lambda l: next(i for i, (a, b) in enumerate(BANDS) if a <= l <= b)
tag = lambda o: f"L{o['layer']:02d}.{o['module']}[{o['channel']}]"
print(f"Rows {N}, modules {len(mods)}, layers {len(set(o['layer'] for o in R))}")
chk = max(abs((o[f'd_noise{k}+'] + o[f'd_noise{k}-']) / 2 - o[f'd_noise{k}']) for o in R for k in range(3))
print(f"sanity: max |(d+ + d-)/2 - d_noise| = {chk:.2e}")

# ---------- 1. reliability ----------
H("1. RELIABILITY OF NOISE DRAWS")
def rel(rows):
    rs = [spear([o["dn"][i] for o in rows], [o["dn"][j] for o in rows]) for i, j in ((0, 1), (0, 2), (1, 2))]
    return rs
print("Spearman between draws (pooled channels). r01 r02 r12 mean | SpearmanBrown 3-draw (ESTIMATE, assumes parallel draws)")
print(f"{'group':14s}{'n':>6s}  r01     r02     r12    mean   |  rel3")
def row(name, rows):
    rs = rel(rows); m = sum(rs) / 3
    print(f"{name:14s}{len(rows):6d} " + " ".join(f"{f(x,3):>7s}" for x in rs) + f" {f(m,3):>7s} | {f(sb(m),3):>6s}")
row("ALL", R)
print("-- by module type"); [row(t, bytype[t]) for t in TYPES]
print("-- by layer band"); [row(f"L{a}-{b}", [o for o in R if a <= o["layer"] <= b]) for a, b in BANDS]
print("-- type x band (mean pairwise r / rel3 estimate)")
print(f"{'':10s}" + "".join(f"{'L%d-%d' % b:>16s}" for b in BANDS))
for t in TYPES:
    cells = []
    for a, b in BANDS:
        rows = [o for o in bytype[t] if a <= o["layer"] <= b]; m = sum(rel(rows)) / 3
        cells.append(f"{f(m,2)}/{f(sb(m),2)}".rjust(16))
    print(f"{t:10s}" + "".join(cells))
print("\nFirst-order |d+ - d-|/2 vs second-order |pair mean| (per draw, pooled draws): median fo, median so, median of per-channel ratio fo/so, share with fo>so")
print(f"{'type':10s}{'med_fo':>12s}{'med_so':>12s}{'med_ratio':>11s}{'frac fo>so':>11s}")
for t in TYPES + ["ALL"]:
    rows = R if t == "ALL" else bytype[t]
    fo = [x for o in rows for x in o["fo"]]; so = [x for o in rows for x in o["so"]]
    ra = [a / b for a, b in zip(fo, so) if b > 0]
    print(f"{t:10s}{e(med(fo)):>12s}{e(med(so)):>12s}{f(med(ra),2):>11s}{f(sum(a > b for a, b in zip(fo, so)) / len(fo),3):>11s}")

# ---------- 2. importance vs sensitivity ----------
H("2. IMPORTANCE (d_zero) vs SENSITIVITY (s)")
print(f"Spearman(d_zero, s) overall = {f(spear([o['d_zero'] for o in R],[o['s'] for o in R]),3)}")
print(f"Spearman(d_mean, s) overall = {f(spear([o['d_mean'] for o in R],[o['s'] for o in R]),3)}")
print("within type:")
for t in TYPES:
    rr = bytype[t]; print(f"  {t:10s} d_zero~s {f(spear([o['d_zero'] for o in rr],[o['s'] for o in rr]),3):>7s}   d_mean~s {f(spear([o['d_mean'] for o in rr],[o['s'] for o in rr]),3):>7s}")
tz, ts = q([o["d_zero"] for o in R], 0.8), q([o["s"] for o in R], 0.8)
print(f"top-20% thresholds (global): d_zero >= {e(tz)}, s >= {e(ts)}")
def quad(rows):
    c = defaultdict(int)
    for o in rows: c[(o["d_zero"] >= tz, o["s"] >= ts)] += 1
    return c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]
print(f"{'group':10s}{'n':>6s}{'crit+fragile':>14s}{'crit+robust':>13s}{'minor+fragile':>15s}{'minor+robust':>14s}")
for name, rows in [("ALL", R)] + [(t, bytype[t]) for t in TYPES]:
    a, b, c, d = quad(rows); print(f"{name:10s}{len(rows):6d}{a:14d}{b:13d}{c:15d}{d:14d}")
print("(expected counts if independent, ALL: crit+fragile = 0.04*N = %.0f)" % (0.04 * N))

# ---------- 3. offset-like ----------
H("3. OFFSET-LIKE CHANNELS")
off = [o for o in R if o["d_zero"] > 0.005 and o["d_mean"] < 0.3 * o["d_zero"]]
print(f"d_zero>0.005 and d_mean<0.3*d_zero : {len(off)} channels (of {sum(o['d_zero']>0.005 for o in R)} with d_zero>0.005)")
for o in sorted(off, key=lambda o: -o["d_zero"]):
    print(f"  {tag(o):22s} d_zero {f(o['d_zero'],5)} d_mean {f(o['d_mean'],5)} s {e(o['s'])} ch_mean {f(o['ch_mean'],3)} ch_rms {f(o['ch_rms'],3)}")
rat = [(o["abs_mean"] / o["ch_rms"], o) for o in R if o["ch_rms"] > 0]
print(f"\n|ch_mean|/ch_rms: median {f(med([r for r,_ in rat]),3)}, p90 {f(q([r for r,_ in rat],.9),3)}, p99 {f(q([r for r,_ in rat],.99),3)}")
big = [(r, o) for r, o in rat if r > 0.8]
print(f"channels with |ch_mean|/ch_rms > 0.8: {len(big)}; by type: " + ", ".join(f"{t}={sum(o['module']==t for _,o in big)}" for t in TYPES))
print("top 15 by ratio:")
for r, o in sorted(big, key=lambda x: -x[0])[:15]:
    print(f"  {tag(o):22s} ratio {f(r,3)} ch_mean {f(o['ch_mean'],3)} ch_rms {f(o['ch_rms'],3)} d_zero {e(o['d_zero'])} d_mean {e(o['d_mean'])}")
print(f"Spearman(|ch_mean|/ch_rms, d_zero - d_mean) = {f(spear([r for r,_ in rat],[o['d_zero']-o['d_mean'] for _,o in rat]),3)}")

# ---------- 4. massive channels ----------
H("4. MASSIVE CHANNELS (ch_rms > 10x module median ch_rms)")
mass = []
for k, rows in mods.items():
    m = med([o["ch_rms"] for o in rows]); mass += [(o["ch_rms"] / m, o) for o in rows if o["ch_rms"] > 10 * m]
print(f"count {len(mass)}")
for r, o in sorted(mass, key=lambda x: -x[0]):
    print(f"  {tag(o):22s} rms/med {f(r,1):>7s} ch_rms {f(o['ch_rms'],3)} ch_mean {f(o['ch_mean'],3)} d_zero {e(o['d_zero'])} d_mean {e(o['d_mean'])} s {e(o['s'])}")
if mass:
    ms = set(id(o) for _, o in mass); non = [o for o in R if id(o) not in ms]
    print(f"massive: median d_zero {e(med([o['d_zero'] for _,o in mass]))} s {e(med([o['s'] for _,o in mass]))} | others: d_zero {e(med([o['d_zero'] for o in non]))} s {e(med([o['s'] for o in non]))}")
    print(f"massive channels in global top-5% s: {sum(o['s']>=q([x['s'] for x in R],.95) for _,o in mass)}/{len(mass)}, top-5% d_zero: {sum(o['d_zero']>=q([x['d_zero'] for x in R],.95) for _,o in mass)}/{len(mass)}")

# ---------- 5. depth ----------
H("5. DEPTH PROFILE")
print(f"{'L':>3s} | {'med d_zero':>11s}{'p90':>11s} | {'med d_mean':>11s}{'p90':>11s} | {'med s':>10s}{'p90':>10s}")
ls = sorted(set(o["layer"] for o in R)); mls = []
for l in ls:
    rr = [o for o in R if o["layer"] == l]; z = [o["d_zero"] for o in rr]; m = [o["d_mean"] for o in rr]; s = [o["s"] for o in rr]
    mls.append(med(s))
    print(f"{l:3d} | {e(med(z)):>11s}{e(q(z,.9)):>11s} | {e(med(m)):>11s}{e(q(m,.9)):>11s} | {e(med(s)):>10s}{e(q(s,.9)):>10s}")
print(f"Spearman(layer, median s) = {f(spear(ls, mls),3)}")
print(f"Spearman(layer, median d_zero) = {f(spear(ls,[med([o['d_zero'] for o in R if o['layer']==l]) for l in ls]),3)}")
print("band medians of s / d_zero / sum d_zero share:")
tot = sum(o["d_zero"] for o in R)
for a, b in BANDS:
    rr = [o for o in R if a <= o["layer"] <= b]
    print(f"  L{a}-{b}: med s {e(med([o['s'] for o in rr]))} med d_zero {e(med([o['d_zero'] for o in rr]))} share of sum d_zero {f(sum(o['d_zero'] for o in rr)/tot,3)}")

# ---------- 6. module-type ----------
H("6. MODULE-TYPE PROFILE")
s95, z95 = q([o["s"] for o in R], .95), q([o["d_zero"] for o in R], .95)
print(f"{'type':10s}{'med d_zero':>12s}{'med d_mean':>12s}{'med s':>10s}{'p90 s':>10s}{'#top5% s':>10s}{'#top5% dz':>11s}")
for t in TYPES:
    rr = bytype[t]
    print(f"{t:10s}{e(med([o['d_zero'] for o in rr])):>12s}{e(med([o['d_mean'] for o in rr])):>12s}{e(med([o['s'] for o in rr])):>10s}{e(q([o['s'] for o in rr],.9)):>10s}{sum(o['s']>=s95 for o in rr):>10d}{sum(o['d_zero']>=z95 for o in rr):>11d}")
print(f"(top-5% thresholds: s >= {e(s95)}, d_zero >= {e(z95)}; 5% of N = {0.05*N:.0f}; per type 600 rows -> expected 30 each)")

# ---------- 7. concentration ----------
H("7. CONCENTRATION (negatives clipped to 0 for shares and Gini)")
for name, key in (("s", "s"), ("d_zero", "d_zero")):
    pos = sorted([max(o[key], 0) for o in R], reverse=True); T = sum(pos)
    print(f"{name}: sum of positive values {e(T)}; " + "; ".join(f"top {p}% -> {f(sum(pos[:max(1,round(N*p/100))])/T,3)}" for p in (1, 5, 10)) + f"; Gini {f(gini(pos),3)}")
    for t in TYPES:
        pt = sorted([max(o[key], 0) for o in bytype[t]], reverse=True)
        print(f"   {t:10s} share of total {f(sum(pt)/T,3)}  within-type top 5% share {f(sum(pt[:30])/sum(pt),3)} Gini {f(gini(pt),3)}")

# ---------- 8. size relationships ----------
H("8. SIZE RELATIONSHIPS (Spearman within each of 210 modules, 20 channels; median [p10,p90] over modules)")
def per_mod(fa, fb, rows_by_mod):
    v = [spear([fa(o) for o in rr], [fb(o) for o in rr]) for rr in rows_by_mod]; v = [x for x in v if x == x]
    return med(v), q(v, .1), q(v, .9), len(v)
pairs = [("s ~ ch_rms", lambda o: o["s"], lambda o: o["ch_rms"]), ("d_zero ~ ch_rms", lambda o: o["d_zero"], lambda o: o["ch_rms"]),
         ("d_zero ~ |ch_mean|", lambda o: o["d_zero"], lambda o: o["abs_mean"]), ("s ~ |ch_mean|", lambda o: o["s"], lambda o: o["abs_mean"])]
print(f"{'pair':20s}" + "".join(f"{t:>20s}" for t in ["ALL"] + TYPES))
for nm, fa, fb in pairs:
    cells = []
    for t in ["ALL"] + TYPES:
        rbm = [rr for (l, m), rr in mods.items() if t == "ALL" or m == t]
        a, b, c, n = per_mod(fa, fb, rbm); cells.append(f"{f(a,2)}[{f(b,2)},{f(c,2)}]".rjust(20))
    print(f"{nm:20s}" + "".join(cells))
print("pooled (all 4200) Spearman: " + "; ".join(f"{nm} {f(spear([fa(o) for o in R],[fb(o) for o in R]),3)}" for nm, fa, fb in pairs))
print("(ESTIMATE of null: Spearman on n=20 has SD ~ 1/sqrt(19) = 0.229; |median| below ~0.1 is not distinguishable from 0 by eye)")

# ---------- 9. negatives ----------
H("9. NEGATIVE VALUES")
nz = [o for o in R if o["d_zero"] < 0]; ns = [o for o in R if o["s"] < 0]; nm = [o for o in R if o["d_mean"] < 0]
print(f"d_zero<0: {len(nz)} ({f(100*len(nz)/N,2)}%), d_mean<0: {len(nm)}, s<0: {len(ns)} ({f(100*len(ns)/N,2)}%)")
mfo = {k: med([x for o in rr for x in o["fo"]]) for k, rr in mods.items()}  # per-module typical first-order term
w1 = sum(abs(o["d_zero"]) < mfo[(o["layer"], o["module"])] for o in nz)
w2 = sum(abs(sum(o["dn"]) / 3) < 2 * st.stdev(o["dn"]) / math.sqrt(3) for o in ns)
print(f"negative d_zero with |d_zero| < module median first-order term: {w1}/{len(nz)}; negative s with |mean d_noise| < 2*SE of the 3 draws (SE=sd/sqrt3): {w2}/{len(ns)}  (heuristic yardstick)")
print(f"most negative d_zero: {e(min(o['d_zero'] for o in R))}; most negative s: {e(min(o['s'] for o in R))}; most negative d_zero relative to module median fo: {f(min(o['d_zero']/mfo[(o['layer'],o['module'])] for o in R),2)}")
print("by type: " + ", ".join(f"{t} dz<0={sum(o['d_zero']<0 for o in bytype[t])} s<0={sum(o['s']<0 for o in bytype[t])}" for t in TYPES))
print("5 most negative d_zero:"); [print(f"  {tag(o):22s} d_zero {e(o['d_zero'])} d_mean {e(o['d_mean'])} s {e(o['s'])}") for o in sorted(R, key=lambda o: o['d_zero'])[:5]]
print("5 most negative s:"); [print(f"  {tag(o):22s} d_zero {e(o['d_zero'])} s {e(o['s'])} fo(med) {e(med(o['fo']))}") for o in sorted(R, key=lambda o: o['s'])[:5]]

# ---------- 10. other ----------
H("10. OTHER OBSERVATIONS")
print("-- top 15 channels by s")
for o in sorted(R, key=lambda o: -o["s"])[:15]: print(f"  {tag(o):22s} s {e(o['s']):>10s} d_zero {e(o['d_zero'])} ch_rms {f(o['ch_rms'],3)} sigma {f(o['sigma'],3)}")
print("-- top 15 channels by d_zero")
for o in sorted(R, key=lambda o: -o["d_zero"])[:15]: print(f"  {tag(o):22s} d_zero {e(o['d_zero'])} s {e(o['s']):>10s} ch_rms {f(o['ch_rms'],3)} ch_mean {f(o['ch_mean'],3)}")
print("-- module-level: sum of positive d_zero (sampled 20 ch) and share of total; top 12 modules")
msum = sorted(((sum(max(o["d_zero"], 0) for o in rr), k) for k, rr in mods.items()), reverse=True); T = sum(x for x, _ in msum)
for x, (l, m) in msum[:12]: print(f"  L{l:02d}.{m:10s} sum d_zero {e(x)} share {f(x/T,3)}")
print("-- GQA: k/v (192 outputs) vs q (576 outputs), same sampled-20 design")
for t in ("q_proj", "k_proj", "v_proj", "o_proj"):
    rr = bytype[t]; print(f"  {t}: med s {e(med([o['s'] for o in rr]))}, p90 s {f(q([o['s'] for o in rr],.9),3)}, med d_zero {e(med([o['d_zero'] for o in rr]))}, p90 d_zero {e(q([o['d_zero'] for o in rr],.9))}, share of positive s {f(sum(max(o['s'],0) for o in rr)/sum(max(o['s'],0) for o in R),3)}")
print("  distinct channel indices sampled in k_proj with same head-group (channel//64) -> per-kv-head sample counts:")
cnt = defaultdict(int)
for o in bytype["k_proj"]: cnt[o["channel"] // 64] += 1
print("   k_proj kv-head counts " + str(dict(sorted(cnt.items()))) + "; v_proj " + str(dict(sorted((lambda c: [c.__setitem__(o['channel']//64, c.get(o['channel']//64,0)+1) for o in bytype['v_proj']] and c)({}).items()))))
print("  k_proj rope-half effect: median s by channel position within head (d<32 'first half' vs >=32 'second half'):")
for t in ("q_proj", "k_proj"):
    a = [o["s"] for o in bytype[t] if o["channel"] % 64 < 32]; b = [o["s"] for o in bytype[t] if o["channel"] % 64 >= 32]
    print(f"   {t}: first half n={len(a)} med s {e(med(a))} | second half n={len(b)} med s {e(med(b))}")
print("-- k_proj s by band"); [print(f"   L{a}-{b}: med {e(med([o['s'] for o in bytype['k_proj'] if a<=o['layer']<=b]))} p90 {e(q([o['s'] for o in bytype['k_proj'] if a<=o['layer']<=b],.9))}") for a, b in BANDS]
print("-- layer 0 vs rest, by type (median s, median d_zero, median ch_rms)")
for t in TYPES:
    a = [o for o in bytype[t] if o["layer"] == 0]; b = [o for o in bytype[t] if o["layer"] > 0]
    print(f"   {t:10s} L0: s {e(med([o['s'] for o in a])):>10s} dz {e(med([o['d_zero'] for o in a]))} rms {f(med([o['ch_rms'] for o in a]),3)} | rest: s {e(med([o['s'] for o in b])):>10s} dz {e(med([o['d_zero'] for o in b]))} rms {f(med([o['ch_rms'] for o in b]),3)}")
print("-- layer 0 o_proj channels (sorted by d_zero)")
for o in sorted([o for o in R if o["layer"] == 0 and o["module"] == "o_proj"], key=lambda o: -o["d_zero"])[:6]:
    print(f"   {tag(o):22s} d_zero {e(o['d_zero'])} d_mean {e(o['d_mean'])} s {e(o['s'])} ch_mean {f(o['ch_mean'],3)} ch_rms {f(o['ch_rms'],4)} sigma {f(o['sigma'],4)}")
print("-- sigma vs s: Spearman(module sigma, module median s) across 210 modules = " + f(spear([mods[k][0]['sigma'] for k in mods],[med([o['s'] for o in mods[k]]) for k in mods]),3))
print("-- fraction of channels with s>0 by type: " + ", ".join(f"{t} {f(sum(o['s']>0 for o in bytype[t])/600,3)}" for t in TYPES))
print("-- d_zero == d_mean exactly (ch_mean ~ 0 or zero-effect) count: " + str(sum(o['d_zero'] == o['d_mean'] for o in R)))
print("-- total d_zero distribution: p50 %s p90 %s p99 %s max %s" % tuple(e(x) for x in (q([o['d_zero'] for o in R],.5), q([o['d_zero'] for o in R],.9), q([o['d_zero'] for o in R],.99), max(o['d_zero'] for o in R))))
