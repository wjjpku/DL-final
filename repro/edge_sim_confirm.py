#!/usr/bin/env python3
"""Confirmatory: does the SGD dynamics WITH near-edge modes (eta*lambda ~ 2)
spontaneously produce the sharp-decay residual that MPL's closed form misses?
Simulate cosine & wsd, fit MPL to cosine, predict wsd, inspect the residual."""
import sys; sys.path.insert(0,'repro')
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import minimize
from reproduce_cosine_to_wsd import build_lrs, huber_log_residual, Curve
from validate_theory import ld_matrix, _coarse
WARMUP=2160; EP=3e-4
# spectrum: backbone (active band) + NOISE modes extending to the EDGE (eta*lam up to ~1.9 at peak)
lam_b=np.geomspace(0.02,5.0,300); g=lam_b**(-0.5); g*=2.6/g.sum()
lam_n=np.geomspace(0.05,6.3e3,700)              # lam up to ~6300 -> eta_peak*lam ~ 1.9 (near edge!)
hw=lam_n**(-0.5); hw*= 0.5/np.sum(hw*(EP/(2*lam_n)))
def simulate(lrs):
    bf=np.ones_like(lam_b); var=np.zeros_like(lam_n); loss=np.empty(len(lrs))
    for t in range(len(lrs)):
        eta=lrs[t]
        bf=(1-eta*lam_b)**2*bf
        d2=np.minimum((1-eta*lam_n)**2,1.0)          # exact rate; clamp past edge (nonlinear stabilize)
        var=np.minimum(d2*var+eta*eta,1e8)
        loss[t]=2.5+np.dot(g,bf)+np.dot(hw,var)
    return loss
def mkcurve(fn):
    lrs=build_lrs(fn); loss=simulate(lrs)
    idx=np.unique(np.linspace(WARMUP+200,len(lrs)-1,300).astype(int))
    return Curve(fn,"sim",idx,loss[idx],lrs)
cos=mkcurve("cosine_24000.csv"); wsd=mkcurve("wsd_20000_24000.csv")
print(f"模拟 loss: cosine {cos.loss[0]:.3f}->{cos.loss[-1]:.3f}, wsd ->{wsd.loss[-1]:.3f}")
# fit MPL (7p, gamma) to the SIMULATED cosine
def pred(p,c): return p[0]+p[1]*np.power(np.cumsum(c.lrs)[c.step],-p[2])+p[3]*ld_matrix(c,p[4],p[5],p[6],1500)
BND=[(0,10),(1e-8,100),(0.05,1.5),(1e-8,1e6),(0.05,20),(0.05,1.5),(0.1,1.5)]
def fit(curves,init):
    fc=[_coarse(c,120) for c in curves]
    def obj(p):
        pr=[];ys=[]
        for c in fc:
            v=pred(p,c)
            if not np.all(np.isfinite(v)) or np.any(v<=0): return 1e18
            pr.append(v);ys.append(c.loss)
        return huber_log_residual(np.concatenate(ys),np.concatenate(pr))
    rng=np.random.RandomState(0); init=np.array(init,float); best,bf=None,np.inf
    for x0 in [init]+[init*np.exp(rng.uniform(-.3,.3,7)) for _ in range(6)]:
        x0=np.clip(x0,[b[0] for b in BND],[b[1] for b in BND])
        r=minimize(obj,x0,method="L-BFGS-B",bounds=BND,options={"maxiter":500})
        if r.fun<bf: bf,best=r.fun,r.x
    return best
p=fit([cos],[2.5,0.5,0.5,300,2.0,0.5,0.5])
rcos=cos.loss-pred(p,cos); rwsd=wsd.loss-pred(p,wsd)
m=wsd.step>=20000
print(f"MPL拟合模拟cosine, 预测模拟wsd: 整体MAE={np.abs(rwsd).mean():.5f}  decay段MAE={np.abs(rwsd[m]).mean():.5f}  decay前={np.abs(rwsd[~m]).mean():.5f}")
print("模拟wsd残差 L_sim-L_MPL 随 step:")
for st in [10000,19500,20500,22000,23900]:
    j=np.argmin(np.abs(wsd.step-st)); print(f"  step={wsd.step[j]:6d} eta={wsd.lrs[wsd.step[j]]:.2e} resid={rwsd[j]:+.5f}")
fig,ax=plt.subplots(figsize=(7,4.5))
ax.plot(wsd.step,rwsd,color="#C44E52",label="sim wsd residual")
ax.axvline(20000,ls=":",c="gray"); ax.axhline(0,c="k",lw=.5)
ax.set_xlabel("step");ax.set_ylabel("L_sim - L_MPL");ax.set_title("Does edge dynamics reproduce the decay residual?");ax.legend();ax.grid(alpha=.3)
fig.savefig("results/edge_sim_residual.png",dpi=130); print("saved results/edge_sim_residual.png")
