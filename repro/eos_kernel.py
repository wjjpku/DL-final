#!/usr/bin/env python3
"""EoS edge-suppressed kernel: the annealing progress is Phi = S - kappa*Q
(Q=sum eta^2), from the exact one-step relaxation rate eta*lam*(2-eta*lam)
=> phase 2 lam dS - lam^2 dQ. Edge (eta*lam~2) suppresses relaxation. Q encodes
the eta-history, so it distinguishes sharp (wsd) from gradual (cosine) decays.
kappa replaces MPL's empirical gamma (same param count). LOO validation."""
import sys; sys.path.insert(0,'repro')
import numpy as np
from scipy.optimize import minimize
from reproduce_cosine_to_wsd import load_curve, huber_log_residual, metrics, MPL_PRECOMPUTED_INIT
from validate_theory import _coarse, ld_matrix
ALL9=["cosine_24000.csv","cosine_72000.csv","constant_24000.csv","constant_72000.csv",
      "wsd_20000_24000.csv","wsdld_20000_24000.csv","wsdcon_3.csv","wsdcon_9.csv","wsdcon_18.csv"]
def ld_phi(c,C,beta,kappa,nk=800):
    lrs=c.lrs.astype(float); T=len(lrs)
    S=np.cumsum(lrs); Q=np.cumsum(lrs*lrs); Phi=S-kappa*Q
    kj=np.unique(np.linspace(1,T-1,min(nk,T-1)).astype(int)); prev=np.concatenate([[0],kj[:-1]])
    a=lrs[prev]-lrs[kj]; ref=Phi[prev]
    gap=Phi[c.step][:,None]-ref[None,:]
    mask=(kj[None,:]<=c.step[:,None])&(gap>0)
    return np.sum(np.where(mask,-a[None,:]*(1-np.power(1+C*np.maximum(gap,0),-beta)),0.0),axis=1)
def pred(p,c,kind):
    L0,A,al,B,C,be,x=p; s1=np.cumsum(c.lrs)[c.step]
    if kind=="mpl": ld=ld_matrix(c,C,be,x,1500)          # x=gamma
    else:           ld=ld_phi(c,C,be,x)                  # x=kappa (EoS)
    return L0+A*np.power(s1,-al)+B*ld
BND={"mpl":[(0,10),(1e-8,100),(0.05,1.5),(1e-8,1e6),(0.05,20),(0.05,1.5),(0.1,1.5)],
     "eos":[(0,10),(1e-8,100),(0.05,1.5),(1e-8,1e6),(0.05,20),(0.05,1.5),(0.0,2500)]}
def fit(curves,kind,init):
    fc=[_coarse(c,120) for c in curves]; bnd=BND[kind]
    def obj(p):
        pr=[];ys=[]
        for c in fc:
            v=pred(p,c,kind)
            if not np.all(np.isfinite(v)) or np.any(v<=0): return 1e18
            pr.append(v);ys.append(c.loss)
        return huber_log_residual(np.concatenate(ys),np.concatenate(pr))
    rng=np.random.RandomState(0); init=np.array(init,float); best,bf=None,np.inf
    seeds=[init]+[init*np.append(np.exp(rng.uniform(-.3,.3,6)),rng.uniform(.3,1.8)) for _ in range(6)]
    for x0 in seeds:
        x0=np.clip(x0,[b[0] for b in bnd],[b[1] for b in bnd])
        r=minimize(obj,x0,method="L-BFGS-B",bounds=bnd,options={"maxiter":500})
        if r.fun<bf: bf,best=r.fun,r.x
    return best
def ev(p,c,kind):
    e=np.abs(c.loss-pred(p,c,kind)); m=c.step>=20000
    return float(e.mean()), float(e[m].mean())
print("留一(训练8条含连续急降) | MPL(gamma) vs EoS-kernel(kappa)")
print(f"{'scale':>5} {'held':>6} | {'MPL test':>8} {'MPL dec':>8} | {'EoS test':>8} {'EoS dec':>8} | kappa*")
agg={}
for s in ["100","400"]:
    for held in ["wsd_20000_24000.csv","wsdld_20000_24000.csv"]:
        tr=[load_curve(s,n) for n in ALL9 if n!=held]
        pm=fit(tr,"mpl",list(MPL_PRECOMPUTED_INIT[s]))
        pe=fit(tr,"eos",list(MPL_PRECOMPUTED_INIT[s])[:6]+[1500.0])
        c=load_curve(s,held); mt,md=ev(pm,c,"mpl"); et,ed=ev(pe,c,"eos")
        print(f"{s+'M':>5} {held[:5]:>6} | {mt:8.5f} {md:8.5f} | {et:8.5f} {ed:8.5f} | {pe[6]:.1f}")
