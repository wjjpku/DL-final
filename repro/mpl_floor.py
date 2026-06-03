#!/usr/bin/env python3
"""MPL + a universal current-LR 'floor' term D*[1-(eta/eta_peak)^c], motivated by
the universal collapse of MPL's decay residual onto f(log(eta_peak/eta)).
Leave-one-out test: does it remove the held-out decay residual? (non-overfit check)"""
import sys; sys.path.insert(0,'repro')
import numpy as np
from scipy.optimize import minimize
from reproduce_cosine_to_wsd import load_curve, huber_log_residual, metrics, MPL_PRECOMPUTED_INIT
from validate_theory import ld_matrix, _coarse
EP=3e-4
ALL9=["cosine_24000.csv","cosine_72000.csv","constant_24000.csv","constant_72000.csv",
      "wsd_20000_24000.csv","wsdld_20000_24000.csv","wsdcon_3.csv","wsdcon_9.csv","wsdcon_18.csv"]
def pred(p,c,floor):
    L0,A,al,B,C,be,ga = p[:7]; s1=np.cumsum(c.lrs)[c.step]
    out=L0+A*np.power(s1,-al)+B*ld_matrix(c,C,be,ga,1500)
    if floor:
        D,cc=p[7],p[8]; eta=c.lrs[c.step]
        out=out+D*(1-np.power(eta/EP, cc))
    return out
B9=[(0,10),(1e-8,100),(0.05,1.5),(1e-8,1e6),(0.05,20),(0.05,1.5),(0.1,1.5),(-1,1),(0.05,4)]
def fit(curves,floor,init):
    fc=[_coarse(c,120) for c in curves]; free=list(range(9 if floor else 7))
    bnd=[B9[i] for i in free]
    def obj(pf):
        p=np.array(init,float); p[free]=pf; pr=[];ys=[]
        for c in fc:
            v=pred(p,c,floor)
            if not np.all(np.isfinite(v)) or np.any(v<=0): return 1e18
            pr.append(v);ys.append(c.loss)
        return huber_log_residual(np.concatenate(ys),np.concatenate(pr))
    rng=np.random.RandomState(0); b0=np.array(init,float)[free]; best,bf=None,np.inf
    for x0 in [b0]+[b0*np.exp(rng.uniform(-0.3,0.3,len(b0))) for _ in range(6)]:
        x0=np.clip(x0,[b[0] for b in bnd],[b[1] for b in bnd]); 
        r=minimize(obj,x0,method="L-BFGS-B",bounds=bnd,options={"maxiter":500})
        if r.fun<bf: bf,best=r.fun,r.x
    p=np.array(init,float); p[free]=best; return p
def evals(p,c,floor):
    e=np.abs(c.loss-pred(p,c,floor)); m=c.step>=20000
    return float(e.mean()), (float(e[m].mean()) if m.any() else np.nan)
print("留一: 训练其余8条, 预测留出1条 | MPL vs MPL+floor")
print(f"{'scale':>5} {'held':>8} | {'MPL test':>8} {'MPL dec':>8} | {'+floor test':>11} {'+floor dec':>10} | D, c")
for s in ["100","400"]:
    for held in ["wsd_20000_24000.csv","wsdld_20000_24000.csv"]:
        tr=[load_curve(s,n) for n in ALL9 if n!=held]
        pm=fit(tr,False,list(MPL_PRECOMPUTED_INIT[s])+[0,1])
        pf=fit(tr,True, list(MPL_PRECOMPUTED_INIT[s])+[0.01,1.0])
        c=load_curve(s,held); mt,md=evals(pm,c,False); ft,fd=evals(pf,c,True)
        print(f"{s+'M':>5} {held[:5]:>8} | {mt:8.5f} {md:8.5f} | {ft:11.5f} {fd:10.5f} | D={pf[7]:.4f} c={pf[8]:.3f}")
