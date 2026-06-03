#!/usr/bin/env python3
"""EoS-MPL: replace MPL's empirical eta_k^{-gamma}(S(t)-S(k)) by a DERIVED
generalized-progress kernel Psi_nu(t)-Psi_nu(k), Psi_nu=sum eta^nu, from the
edge-of-stability assumption lambda_eff ~ eta^{-s} (nu=1-s). Same #params.
Compare MPL vs EoS-MPL on the same pipeline: fit cosine -> test WSD."""
import sys; sys.path.insert(0,'repro')
import numpy as np
from scipy.optimize import minimize
from reproduce_cosine_to_wsd import load_curve, huber_log_residual, metrics, MPL_PRECOMPUTED_INIT
from validate_theory import _coarse

SCALES=["25","100","400"]
COS=["cosine_24000.csv","cosine_72000.csv"]
WSD=["wsd_20000_24000.csv","wsdld_20000_24000.csv","wsdcon_3.csv","wsdcon_9.csv","wsdcon_18.csv"]

def grids(lrs,nk=600):
    T=len(lrs); kj=np.unique(np.linspace(1,T-1,min(nk,T-1)).astype(int)); prev=np.concatenate([[0],kj[:-1]])
    a=lrs[prev]-lrs[kj]   # decrement over cell
    return kj,prev,a

def pred(p,lrs,steps,kind):
    L0,A,al,B,C,be,x=p
    S1=np.cumsum(lrs)
    kj,prev,a=grids(lrs)
    if kind=="mpl":   # eta_k^{-gamma} * (S1 gap)
        base=C*np.power(lrs[kj],-x); refS=S1[prev]
        gap=(S1[steps][:,None]-refS[None,:]); arg=base[None,:]*np.maximum(gap,0)
    else:             # EoS: C * (Psi_nu gap), Psi=cumsum(eta^nu)
        Psi=np.cumsum(np.power(lrs,x)); refP=Psi[prev]
        gap=(Psi[steps][:,None]-refP[None,:]); arg=C*np.maximum(gap,0)
    mask=(kj[None,:]<=steps[:,None])&(gap>0)
    ld=np.sum(np.where(mask,-a[None,:]*(1-np.power(1+arg,-be)),0.0),axis=1)
    return L0+A*np.power(S1[steps],-al)+B*ld

B7=[(0,10),(1e-8,100),(0.05,1.5),(1e-8,1e6),(0.05,20),(0.05,1.5),(0.01,2.0)]
def fit(curves,kind,init):
    fc=[_coarse(c,110) for c in curves]; pre=[(c.lrs,c.step[::1],c.loss) for c in fc]
    def obj(p):
        pr=[];ys=[]
        for lrs,st,loss in pre:
            v=pred(p,lrs,st,kind)
            if not np.all(np.isfinite(v)) or np.any(v<=0): return 1e18
            pr.append(v);ys.append(loss)
        return huber_log_residual(np.concatenate(ys),np.concatenate(pr))
    best,bf=None,np.inf
    for f in (1,0.8,1.2,1.5,0.6):
        r=minimize(obj,np.array(init)*([1]*6+[f]),method="L-BFGS-B",bounds=B7,options={"maxiter":400})
        if r.fun<bf: bf,best=r.fun,r.x
    return best
def te(p,s,kind): return float(np.mean([metrics(load_curve(s,n).loss,pred(p,load_curve(s,n).lrs,load_curve(s,n).step,kind))["mae"] for n in WSD]))

print(f"{'scale':>6} | {'MPL(gamma)':>11} | {'EoS-MPL(nu)':>12} | {'nu*':>6} (=1-s) | winner")
tot=[0,0]
for s in SCALES:
    cur=[load_curve(s,n) for n in COS]
    init=list(MPL_PRECOMPUTED_INIT[s])  # L0,A,al,B,C,be,gamma
    mpl=fit(cur,"mpl",init)
    eos=fit(cur,"eos",init[:6]+[0.4])   # nu init 0.4
    m,e=te(mpl,s,"mpl"),te(eos,s,"eos")
    tot[0]+=m;tot[1]+=e
    print(f"{s+'M':>6} | {m:>11.5f} | {e:>12.5f} | {eos[6]:>6.3f} | {'EoS-MPL' if e<m else 'MPL'}")
print(f"\n  平均 test MAE:  MPL={tot[0]/3:.5f}   EoS-MPL={tot[1]/3:.5f}   "
      f"{'EoS-MPL 更优' if tot[1]<tot[0] else 'MPL 更优'} ({100*(tot[0]-tot[1])/tot[0]:+.1f}%)")
