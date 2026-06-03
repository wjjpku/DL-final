#!/usr/bin/env python3
"""lag-MPL: the realized annealing benefit LD_r lags the instantaneous LD with a
relaxation timescale tau in cumulative-LR progress S. Sharp decays (small dS) ->
LD_r can't keep up -> loss stays higher, matching the diagnosed decay-phase
residual. Fit on cosine, evaluate on WSD; also a fixed-tau sweep (cosine may not
constrain tau)."""
import sys; sys.path.insert(0,'repro')
import numpy as np
from scipy.optimize import minimize
from reproduce_cosine_to_wsd import load_curve, huber_log_residual, metrics, MPL_PRECOMPUTED_INIT
from validate_theory import ld_matrix, _coarse
SCALES=["25","100","400"]; COS=["cosine_24000.csv","cosine_72000.csv"]
WSD=["wsd_20000_24000.csv","wsdld_20000_24000.csv","wsdcon_3.csv","wsdcon_9.csv","wsdcon_18.csv"]

def lag(ld, S, tau):
    if tau<=0: return ld
    out=np.empty_like(ld); out[0]=ld[0]
    for i in range(1,len(ld)):
        a=np.exp(-(S[i]-S[i-1])/tau); out[i]=a*out[i-1]+(1-a)*ld[i]
    return out
def pred(p,c,tau=None):
    L0,A,al,B,C,be,ga = p[:7]; tau = p[7] if tau is None else tau
    S=np.cumsum(c.lrs); s1=S[c.step]
    ld=ld_matrix(c,C,be,ga,1500)
    ldr=lag(ld, s1, tau)
    return L0+A*np.power(s1,-al)+B*ldr
B8=[(0,10),(1e-8,100),(0.05,1.5),(1e-8,1e6),(0.05,20),(0.05,1.5),(0.1,1.5),(0.0,5.0)]
def fit(curves,init,fixtau=None):
    fc=[_coarse(c,120) for c in curves]
    free=list(range(7)) if fixtau is not None else list(range(8))
    bnd=[B8[i] for i in free]
    def asm(pf):
        f=np.array(init,float); f[free]=pf; return f
    def obj(pf):
        p=asm(pf); pr=[];ys=[]
        for c in fc:
            v=pred(p,c,fixtau)
            if not np.all(np.isfinite(v)) or np.any(v<=0): return 1e18
            pr.append(v);ys.append(c.loss)
        return huber_log_residual(np.concatenate(ys),np.concatenate(pr))
    rng=np.random.RandomState(0); b0=np.array(init,float)[free]; best,bf=None,np.inf
    for x0 in [b0]+[b0*np.exp(rng.uniform(-0.3,0.3,len(b0))) for _ in range(6)]:
        x0=np.clip(x0,[b[0] for b in bnd],[b[1] for b in bnd])
        r=minimize(obj,x0,method="L-BFGS-B",bounds=bnd,options={"maxiter":400})
        if r.fun<bf: bf,best=r.fun,r.x
    return asm(best)
def teval(p,s,fixtau=None):
    errs=[]; dec=[]
    for n in WSD:
        c=load_curve(s,n); e=np.abs(c.loss-pred(p,c,fixtau)); errs.append(e.mean())
        if "wsd_2" in n or "wsdld" in n:
            m=c.step>=20000; dec.append(e[m].mean())
    return float(np.mean(errs)), float(np.mean(dec))

print("=== lag-MPL: 在 cosine 拟合(tau自由),WSD 评估 ===")
print(f"{'scale':>5} | {'MPL test':>8} {'MPL decay':>9} | {'lag test':>8} {'lag decay':>9} | tau*")
for s in SCALES:
    cur=[load_curve(s,n) for n in COS]; I=list(MPL_PRECOMPUTED_INIT[s])+[0.0]
    pm=fit(cur,I,fixtau=0.0); pl=fit(cur,I)
    mt,md=teval(pm,s,0.0); lt,ld_=teval(pl,s)
    print(f"{s+'M':>5} | {mt:8.5f} {md:9.5f} | {lt:8.5f} {ld_:9.5f} | {pl[7]:.4f}")
print("\n=== 固定 tau 扫描(看滞后机理能否压 decay 残差),以 400M 为例 ===")
cur=[load_curve("400",n) for n in COS]; I=list(MPL_PRECOMPUTED_INIT["400"])+[0.0]
for tau in [0.0,0.05,0.1,0.2,0.5,1.0]:
    p=fit(cur,I,fixtau=tau); t,d=teval(p,"400",tau)
    print(f"  tau={tau:4.2f} | 400M test={t:.5f}  decay残差={d:.5f}")
