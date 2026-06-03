#!/usr/bin/env python3
"""Fully EoS-consistent law: replace cumulative LR S=sum(eta) by a generalized
progress Psi_nu=sum(eta^nu) in BOTH the backbone and the annealing kernel
(nu=1-s, EoS curvature lambda~eta^{-s}). Compare to MPL on cosine->WSD."""
import sys; sys.path.insert(0,'repro')
import numpy as np
from scipy.optimize import minimize
from reproduce_cosine_to_wsd import load_curve, huber_log_residual, metrics, MPL_PRECOMPUTED_INIT
from validate_theory import _coarse
SCALES=["25","100","400"]; COS=["cosine_24000.csv","cosine_72000.csv"]
WSD=["wsd_20000_24000.csv","wsdld_20000_24000.csv","wsdcon_3.csv","wsdcon_9.csv","wsdcon_18.csv"]
def grids(lrs,nk=600):
    T=len(lrs); kj=np.unique(np.linspace(1,T-1,min(nk,T-1)).astype(int)); prev=np.concatenate([[0],kj[:-1]])
    return kj,prev,lrs[prev]-lrs[kj]
def ld_from(P,steps,kj,prev,a,C,be):
    gap=P[steps][:,None]-P[prev][None,:]; mask=(kj[None,:]<=steps[:,None])&(gap>0)
    return np.sum(np.where(mask,-a[None,:]*(1-np.power(1+C*np.maximum(gap,0),-be)),0.0),axis=1)
def pred(p,lrs,steps,kind):
    kj,prev,a=grids(lrs); S1=np.cumsum(lrs)
    if kind=="mpl":
        L0,A,al,B,C,be,ga=p
        gap=S1[steps][:,None]-S1[prev][None,:]; base=C*np.power(lrs[kj],-ga)
        mask=(kj[None,:]<=steps[:,None])&(gap>0)
        ld=np.sum(np.where(mask,-a[None,:]*(1-np.power(1+base[None,:]*np.maximum(gap,0),-be)),0.0),axis=1)
        return L0+A*np.power(S1[steps],-al)+B*ld
    if kind=="full":          # shared nu in backbone + annealing (7p)
        L0,A,al,B,C,be,nu=p; P=np.cumsum(np.power(lrs,nu))
        return L0+A*np.power(P[steps],-al)+B*ld_from(P,steps,kj,prev,a,C,be)
    if kind=="2nu":           # separate backbone nu_b and annealing nu_a (8p)
        L0,A,al,B,C,be,nua,nub=p; Pa=np.cumsum(np.power(lrs,nua)); Pb=np.cumsum(np.power(lrs,nub))
        return L0+A*np.power(Pb[steps],-al)+B*ld_from(Pa,steps,kj,prev,a,C,be)
B={"mpl":[(0,10),(1e-8,100),(0.05,1.5),(1e-8,1e6),(0.05,20),(0.05,1.5),(0.01,1.5)],
   "full":[(0,10),(1e-8,100),(0.05,1.5),(1e-8,1e6),(0.05,1e3),(0.05,1.5),(0.0,1.5)],
   "2nu":[(0,10),(1e-8,100),(0.05,1.5),(1e-8,1e6),(0.05,1e3),(0.05,1.5),(0.0,1.5),(0.0,1.5)]}
def fit(curves,kind,init):
    fc=[_coarse(c,110) for c in curves]; pre=[(c.lrs,c.step,c.loss) for c in fc]
    def obj(p):
        pr=[];ys=[]
        for lrs,st,loss in pre:
            v=pred(p,lrs,st,kind)
            if not np.all(np.isfinite(v)) or np.any(v<=0): return 1e18
            pr.append(v);ys.append(loss)
        return huber_log_residual(np.concatenate(ys),np.concatenate(pr))
    best,bf=None,np.inf
    for f in (0.6,0.8,1.0,1.2,1.5):
        x0=np.array(init,float); x0[6:]=x0[6:]*f
        r=minimize(obj,x0,method="L-BFGS-B",bounds=B[kind],options={"maxiter":500})
        if r.fun<bf: bf,best=r.fun,r.x
    return best
def te(p,s,kind): return float(np.mean([metrics(load_curve(s,n).loss,pred(p,load_curve(s,n).lrs,load_curve(s,n).step,kind))["mae"] for n in WSD]))
print(f"{'scale':>6} | {'MPL':>9} | {'EoS-full(7p)':>12} | {'EoS-2nu(8p)':>12}")
tot={'mpl':0,'full':0,'2nu':0}
for s in SCALES:
    cur=[load_curve(s,n) for n in COS]; I=list(MPL_PRECOMPUTED_INIT[s])
    pm=fit(cur,"mpl",I); pf=fit(cur,"full",I[:6]+[0.3]); p2=fit(cur,"2nu",I[:6]+[0.3,1.0])
    m,f2,t2=te(pm,s,"mpl"),te(pf,s,"full"),te(p2,s,"2nu")
    tot['mpl']+=m;tot['full']+=f2;tot['2nu']+=t2
    print(f"{s+'M':>6} | {m:>9.5f} | {f2:>12.5f} | {t2:>12.5f}   (nu_full={pf[6]:.3f})")
print(f"\n  平均: MPL={tot['mpl']/3:.5f}  EoS-full={tot['full']/3:.5f}  EoS-2nu={tot['2nu']/3:.5f}")
