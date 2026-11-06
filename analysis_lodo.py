import numpy as np, pandas as pd, os, sys
R="/workspace/results_xd"; G=["c4","pilecc","github","stackex","pmc","freelaw"]; Q=["c4","github","pmc"]; K=["layer","module","channel"]
os.makedirs(R+"/offline",exist_ok=True)
rows=[]
for m in ["qwen3","llama"]:
    gs=[pd.read_csv(f"{R}/{m}/grad_{d}/est_grad_rot4.csv") for d in G]
    es={d:pd.read_csv(f"{R}/{m}/est_{d}/est_rot4.csv") for d in Q}
    base=gs[0][K]
    for x in gs[1:]+list(es.values()): assert x[K].equals(base)
    g=np.stack([x.g1.values for x in gs]); qf={d:es[d].qf.values/2 for d in Q}
    fish=np.mean([es[d].fish.values for d in Q],0); en=np.mean([es[d].en.values for d in Q],0)
    grp=base.groupby(["layer","module"]).indices
    for hi,h in enumerate(G):
        O=[i for i,d in enumerate(G) if d!=h]; QO=[d for d in Q if d!=h]
        qm=np.mean([qf[d] for d in QO],0)
        truth=g[hi]+(qf[h] if h in qf else qm)
        sc={n:np.zeros(len(base)) for n in ["curv","naive","s_rob","lcb0.5","lcb1","lcb2","minimax","signagree","cheap","cheap+bg","cheap_sc+bg"]}
        for key,idx in grp.items():
            gg=g[np.ix_(O,idx)]; mu=gg.mean(0); q=qm[idx]
            sl=[]
            for j in range(5):
                x,y=np.delete(gg,j,0).mean(0),gg[j]; sl.append(np.cov(x,y)[0,1]/x.var(ddof=1))
            b=float(np.clip(np.mean(sl),0,1))
            sd=gg.std(0,ddof=1)
            mm=np.min(np.stack([gg[k]+(qf[G[O[k]]][idx] if G[O[k]] in QO else q) for k in range(5)]),0)
            sa=(np.sign(gg)==np.sign(mu)).mean(0)
            n=len(idx); blk=np.arange(n)//128
            bm=(np.bincount(blk,en[idx])/np.bincount(blk))[blk]
            ch=fish[idx]*bm
            scl=q.sum()/ch.sum() if ch.sum()!=0 else 1
            s=sc; s["curv"][idx]=q; s["naive"][idx]=mu+q; s["s_rob"][idx]=b*mu+q
            for l in (0.5,1,2): s[f"lcb{l:g}"][idx]=mu-l*sd/np.sqrt(5)+q
            s["minimax"][idx]=mm; s["signagree"][idx]=mu*sa+q
            s["cheap"][idx]=ch; s["cheap+bg"][idx]=ch+b*mu; s["cheap_sc+bg"][idx]=scl*ch+b*mu
        for nme,sv in sc.items():
            r=[]
            for key,idx in grp.items():
                k=max(1,len(idx)//32); t=truth[idx]
                r.append(t[np.argsort(-sv[idx])[:k]].sum()/np.sort(t)[::-1][:k].sum())
            rows.append((m,h,nme,np.mean(r)))
d=pd.DataFrame(rows,columns=["model","heldout","score","share"]); d.to_csv(R+"/offline/lodo_scores.csv",index=False)
p=d.pivot_table(index=["model","score"],columns="heldout",values="share")
p["mean"]=p[G].mean(1); p["worst"]=p[G].min(1); p["worst_dom"]=p[G].idxmin(1)
print(p.round(3).to_string())
