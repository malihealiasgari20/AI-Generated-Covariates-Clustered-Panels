"""Independent re-check of the reviewer's two diagnostics, on the sealed FINAL_V1 generator and production seeds.

A. Ratio linearization for the uniform rectifier (E-4), baseline cell, kappa = 0, gamma_W = 0.15.
B. Future-shock contamination versus a variance-matched independent shock, Experiment-3 configurations (rho_e = 0,
   mean loading 0.60), naive estimator, paired within replication.
Read-only use of the sealed code. Output: printed table.
"""
import sys, os, math
from dataclasses import replace
CODE = "/mnt/user-data/outputs/Final_Study_v1/code"
sys.path[:0] = [CODE, os.path.join(CODE, "deps")]
import numpy as np
from fv1 import registry, seeds
from stage0_dgp import generate_panel, discretize
from stage0_naive import naive_fit, _period_design
from s2_core import designs
from e5_core import solve_rect

configs, _ = registry.load()
ns = seeds.PRODUCTION
nA = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
nB = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

# ---------------- A
base = [c for c in configs if c.config_id.startswith("N=300|T=8|icc=0.3|rho=0.6|tau=0.1|pi=0.15,0.3,0.6|sde=0.55|sdW=0.62|lam=1.2,1.05,0.45|g=0.15|kappa=0")
        and c.config_id.endswith("kappa=0")][0]
th, se0, se1 = [], [], []
for r in range(nA):
    pan = generate_panel(base.p, seeds.panel_rng(ns, base.seed_cell_id, r))
    N, T = pan.Y.shape
    D = _period_design(N, T)
    d = designs(pan, D, pan.ZL, pan.ZL)
    R = pan.labeled.astype(float); G = R.sum()
    rw = np.repeat(R * N / G, T)
    t_, s_, _, A = solve_rect(d["WL"], d["WL"], d["WH"], d["y"], rw, N, T)
    k = A.shape[0]
    thv = np.linalg.solve(A, d["WL"].T @ d["y"] + (d["WH"] * rw[:, None]).T @ d["y"] - (d["WL"] * rw[:, None]).T @ d["y"])
    y, WL, WH = d["y"], d["WL"], d["WH"]
    rL, rH = y - WL @ thv, y - WH @ thv
    psiL = (WL * rL[:, None]).reshape(N, T, k).sum(1)
    c = (WH * rH[:, None] - WL * rL[:, None]).reshape(N, T, k).sum(1)        # per-cluster correction
    g = psiL + (R * N / G)[:, None] * c                                      # sealed cluster score
    mbar = (R[:, None] * c).sum(0) / G                                       # labeled-cluster mean correction
    gcor = g - ((R * N / G) - 1.0)[:, None] * mbar                           # ratio linearization
    Ai = np.linalg.inv(A)
    v0 = (Ai @ (g.T @ g) @ Ai * N / (N - 1))[0, 0]
    v1 = (Ai @ (gcor.T @ gcor) @ Ai * N / (N - 1))[0, 0]
    assert abs(thv[0] - t_) < 1e-10 and abs(math.sqrt(v0) - s_) < 1e-10
    th.append(thv[0]); se0.append(math.sqrt(v0)); se1.append(math.sqrt(v1))
th = np.array(th); z = 1.959963984540054
cov = lambda se: np.mean(np.abs(th - 0.15) <= z * np.array(se))  # theta0 = gamma_W to < 0.001 here
print(f"A. E-4 baseline, {nA} production replications")
print(f"   mean estimate {th.mean():.5f}; empirical SD {th.std(ddof=1):.5f}")
print(f"   mean SE sealed sandwich {np.mean(se0):.5f} (coverage {cov(se0):.3f}); with ratio correction {np.mean(se1):.5f} (coverage {cov(se1):.3f})")

# ---------------- B
print(f"B. naive displacement, Experiment-3 configurations (rho_e = 0, mean loading 0.60), {nB} production replications each")
for T in (4, 8, 16):
    cfg = [c for c in configs if c.config_id.startswith(f"N=300|T={T}|icc=0.3|rho=0|") and "X3" in c.modules
           and float(c.kappa) > 0][0]
    p = cfg.p
    df, di = [], []
    for r in range(nB):
        pan = generate_panel(p, seeds.panel_rng(ns, cfg.seed_cell_id, r))
        D = _period_design(*pan.Y.shape)
        Z0 = discretize(pan.Zstar - pan.C, p)
        zr = np.random.default_rng([20261113, T, r]).standard_normal(pan.eps_next.shape) * pan.eps_next.std()
        Zi = discretize(pan.Zstar - pan.C + pan.kappa_t[None, :] * zr, p)
        t_fut = naive_fit(pan, D)[0]
        t_0 = naive_fit(replace(pan, ZL=Z0), D)[0]
        t_ind = naive_fit(replace(pan, ZL=Zi), D)[0]
        df.append(t_fut - t_0); di.append(t_ind - t_0)
    df, di = np.array(df), np.array(di)
    m = lambda x: f"{x.mean():+.4f} ({x.std(ddof=1) / math.sqrt(len(x)):.4f})"
    print(f"   T={T:2d} kappa={float(cfg.kappa):.3f}: future {m(df)}  independent {m(di)}  leakage-specific {m(df - di)}")
