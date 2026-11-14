"""Mechanical tests of r1_core on throwaway R1TEST seeds (no production panels are touched)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import r1_core as R
from stage0_dgp import make_rng
from stage0_ht import ht_fit
from stage1_design_aware import crossfit_predictions
import e5_core

cfgs, th0 = R.load()
base = next(c for k, c in cfgs.items() if k.endswith("g=0.15|kappa=0") and k.startswith("N=300|T=8|icc=0.3|rho=0.6|tau=0.1|pi=0.15,0.3,0.6|sde=0.55|sdW=0.62|lam=1.2,1.05,0.45"))
fails = 0
for r in range(3):
    pan = R.panel(base, r, test=True)
    D = R._period_design(*pan.Y.shape)
    key = ("R1TEST", base.seed_cell_id, r)
    fr = lambda: R.r1_rng("folds", base.seed_cell_id, r, True)
    res = R.rect_all(pan, fr)
    # sealed comparisons
    th_ppi, _, se_ppi = R.ppi_fit(pan, D)
    th5, se5 = R.crossppi_fit(pan, fr(), D)[:2]
    th6, se6 = ht_fit(pan, D)[:2]
    y, W = R.designs(pan); N, T = pan.Y.shape
    Zf, Zo, _ = crossfit_predictions(pan, fr(), True)
    out12b = e5_core.solve_rect(W(Zf), W(Zo), W(pan.ZH), y, np.repeat(R.rect_weights(pan, "ht"), T), N, T)
    Zf, Zo, _ = crossfit_predictions(pan, fr(), False)
    out12a = e5_core.solve_rect(W(Zf), W(Zo), W(pan.ZH), y, np.repeat(R.rect_weights(pan, "ht"), T), N, T)
    checks = [("E-4", th_ppi, se_ppi), ("E-5", th5, se5), ("E-6", th6, se6), ("E-12a", out12a[0], out12a[1]), ("E-12b", out12b[0], out12b[1])]
    for n, a, b in checks:
        ok = abs(res[n]["th"] - a) < 1e-9 and abs(res[n]["se"] - b) < 1e-9
        fails += not ok
        print(f"r{r} {n}: theta {res[n]['th']:.6f} vs sealed {a:.6f}; se {res[n]['se']:.6f} vs {b:.6f} {'OK' if ok else 'MISMATCH'}")
    # numerical derivative check of the calibration term for E-5 and E-12b
    for name, fam, kind in (("E-5", False, "uniform"), ("E-12b", True, "ht")):
        Zf, Zo, coefs, folds, C = R.crossfit(pan, fr(), fam)
        rw = R.rect_weights(pan, kind)
        fit = R.rect(W(Zf), W(Zo), W(pan.ZH), y, rw, N, T)
        thv = fit["th"]
        def moment(cf):
            Zf_ = (C @ cf.mean(0)).reshape(N, T); Zo_ = np.zeros((N, T))
            for j, h in enumerate(folds):
                if h.size: Zo_[h] = (C[R.rows(h, T)] @ cf[j]).reshape(h.size, T)
            Wf_, Wo_, WH_ = W(Zf_), W(Zo_), W(pan.ZH); rwt = np.repeat(rw, T)
            return Wf_.T @ (y - Wf_ @ thv) + (WH_ * rwt[:, None]).T @ (y - WH_ @ thv) - (Wo_ * rwt[:, None]).T @ (y - Wo_ @ thv)
        # analytic: d moment / d beta_j = J_f/K - J_j ; compare by perturbing fold 0 coefficients
        h = 1e-6; dnum = []
        for q in range(C.shape[1]):
            cf = coefs.copy(); cf[0, q] += h
            dnum.append((moment(cf) - moment(coefs)) / h)
        dnum = np.array(dnum).T
        # reconstruct analytic via calib_term internals
        Ct = np.column_stack([R.within_demean(C[:, j].reshape(N, T)).reshape(N * T) for j in range(C.shape[1])])
        def J(v, Wm, rr_):
            e1 = np.zeros((Wm.shape[1], 1)); e1[0, 0] = 1
            return e1 @ ((v * rr_) @ Ct)[None, :] - thv[0] * (Wm * v[:, None]).T @ Ct
        Wf_, Wo_ = W(Zf), W(Zo)
        Jf = J(np.ones(N * T), Wf_, fit["rf"]); v = np.zeros(N * T); v[R.rows(folds[0], T)] = np.repeat(rw, T)[R.rows(folds[0], T)]
        Jj = J(v, Wo_, fit["ro"]); ana = Jf / len(folds) - Jj
        err = np.abs(ana - dnum).max() / np.abs(dnum).max()
        fails += err > 1e-4
        print(f"r{r} {name}: calibration derivative relative error {err:.2e}")
    print(f"r{r} linearized SEs:", {k: (round(v['se'], 4), round(v['se_lin'], 4)) for k, v in res.items() if v})
    print(f"r{r} PTD:", R.ptd(pan, R.r1_rng("ptd", base.seed_cell_id, r, True)))
    print(f"r{r} E-10w:", R.e10w(pan))
    for kind in ("machine", "proxy"):
        p2 = R.restratify(pan, kind, R.r1_rng("strata", base.seed_cell_id, r, True))
        print(f"r{r} {kind} strata: labeled {int(p2.labeled.sum())}, corr(new stratum, oracle) {np.corrcoef(p2.strata, pan.strata)[0,1]:.2f}")
print("FAILURES", fails)
