#!/usr/bin/env python3
"""Cross-scale auxiliary prediction: 2x25M + 2x100M + 1x400M -> predict 400M WSD.
Tests the original proposal: do cheap small-model curves help when the target
(400M) has only ONE training curve?"""
import sys; sys.path.insert(0,'repro')
import numpy as np
import validate_theory as V
from reproduce_cosine_to_wsd import load_curve, MPL_PRECOMPUTED_INIT

COS = ["cosine_24000.csv", "cosine_72000.csv"]
WSD = ["wsd_20000_24000.csv","wsdld_20000_24000.csv","wsdcon_3.csv","wsdcon_9.csv","wsdcon_18.csv"]

def fit_small(scale):
    cur = [load_curve(scale, n) for n in COS]
    return V.fit_mpl(cur, np.array(MPL_PRECOMPUTED_INIT[scale], float), V.F_MPL)

print("拟合辅助小模型 (2x25M, 2x100M cosine)...")
p25, p100 = fit_small("25"), fit_small("100")
shared = np.mean([p25[[2,4,5,6]], p100[[2,4,5,6]]], axis=0)   # alpha,C,beta,gamma
print(f"  借来的共享指数: alpha={shared[0]:.3f} C={shared[1]:.3f} beta={shared[2]:.3f} gamma={shared[3]:.3f}")

t1 = [load_curve("400", "cosine_24000.csv")]                  # 仅 1 条 400M
base = p100.copy()                                            # 诚实初值:最近的小模型(100M)
A = V.fit_mpl(t1, base, V.F_MPL)                              # 只用1条,MPL 7参
baseB = p100.copy(); baseB[[2,4,5,6]] = shared
B = V.fit_mpl(t1, baseB, V.F_SC3)                             # 1条 + 小模型指数,拟3振幅
ref = V.fit_mpl([load_curve("400", n) for n in COS], base, V.F_MPL)   # 全量2条参考

def te(p): return float(np.mean(list(V.mae_on(p, "400", WSD).values())))
print("\n=== 预测 400M 的 5 条 WSD,test MAE ===")
print(f"  (A) 仅 1 条400M, MPL(7参)            : {te(A):.5f}")
print(f"  (B) 1 条400M + 2x25M+2x100M 辅助(3振幅): {te(B):.5f}   <- 用小模型")
print(f"  (C) 2 条400M全量, MPL(7参) [参考上限]  : {te(ref):.5f}")
g = 100*(te(A)-te(B))/te(A)
print(f"\n  辅助小模型相对'仅1条'的提升: {g:+.1f}%   => 小模型{'有帮助' if te(B)<te(A) else '没帮助'}")
