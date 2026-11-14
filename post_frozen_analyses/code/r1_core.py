"""R1 revision analyses (after the frozen FINAL_V1 study): estimators, variances, intervals.

Read-only use of the sealed FINAL_V1 code (generator, seeds, sealed estimator routines). Everything new is here:
  * linearized cluster scores for rectifiers whose normalization is random (uniform, Hajek, post-stratified) and
    for the fitted cross-fitted calibration (Supplement S2.4 of the revision);
  * a pairs cluster bootstrap that recomputes every sample-dependent quantity (labeled count, weights, folds,
    calibration);
  * the predict-then-debias (PTD) estimator of Kluger et al. (2025), diagonal tuning and cluster bootstrap
    (their Algorithm 5), and a design-weighted, power-tuned PPI++ comparator (E-10w);
  * small-cluster intervals: CR1 with t(G_S - 1) critical values and a wild cluster score bootstrap;
  * proxy validation strata (machine-score tertiles; tertiles of a noisy proxy of the construct).
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import replace

CODE = os.environ.get("FV1_CODE", "/mnt/user-data/outputs/Final_Study_v1/code")
for _p in (CODE, os.path.join(CODE, "deps")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402

from fv1 import registry, seeds, theta0 as T0  # noqa: E402
from stage0_dgp import generate_panel, discretize, within_demean, make_rng  # noqa: E402
from stage0_naive import naive_fit, _period_design  # noqa: E402
from stage0_labeled_only import labeled_only_fit  # noqa: E402
from stage0_rc import rc_fit  # noqa: E402
from stage0_crossppi import crossppi_fit  # noqa: E402
from stage0_ppi import ppi_fit  # noqa: E402
from stage1_design_aware import _calibration_matrix, N_FOLDS  # noqa: E402
import s2_core  # noqa: E402

Z = 1.959963984540054
STUDY = os.path.dirname(CODE)
PROD = seeds.PRODUCTION


# ----------------------------------------------------------------------------------------------- setup
def load():
    configs, _ = registry.load(STUDY)
    tm = T0.load_production_map(os.path.join(STUDY, "gates/FINAL_V1_THETA0_MAP.json"),
                                "6d18e3f01f21ed31ebe922ed1b1f4f7f1431eccb072fb889b91fb98388af00a7")
    th = T0.lookup(tm, configs, require_all=True)
    return {c.config_id: c for c in configs}, {c.config_id: th[c.index] for c in configs}


def r1_rng(tag, cell, r, test=False):
    """Seed streams of the revision analyses (namespace R1, or R1TEST for mechanical tests)."""
    ns = "R1TEST" if test else "R1"
    h = hashlib.sha256(f"{ns}|{tag}|{cell}|{r}".encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(list(np.frombuffer(h, dtype=np.uint32))))


def panel(cfg, r, test=False):
    if test:
        return generate_panel(cfg.p, r1_rng("panel", cfg.seed_cell_id, r, True))
    return generate_panel(cfg.p, seeds.panel_rng(PROD, cfg.seed_cell_id, r))


def fold_rng(cfg, r, test=False):
    if test:
        return r1_rng("folds", cfg.seed_cell_id, r, True)
    return make_rng(*seeds.fold_key(PROD, cfg.seed_cell_id, r))


UNIT_FIELDS = ("Y", "ZH", "ZL", "Zstar", "X", "eps", "eps_next", "C", "u", "strata", "labeled")


def take_units(pan, idx):
    return replace(pan, **{f: getattr(pan, f)[idx] for f in UNIT_FIELDS})


# ----------------------------------------------------------------------------------------------- designs
def designs(pan):
    N, T = pan.Y.shape
    D = _period_design(N, T)
    y = within_demean(pan.Y).reshape(N * T)
    x = within_demean(pan.X).reshape(N * T)
    W = lambda Zm: np.column_stack([within_demean(np.asarray(Zm, float)).reshape(N * T), x, D])
    return y, W


def rows(units, T):
    return np.repeat(units, T) * T + np.tile(np.arange(T), units.size)


def crossfit(pan, frng, weighted, groups=None):
    """Exactly the sealed crossfit_predictions, also returning folds and fold coefficients. With `groups`
    (bootstrap), copies of one original unit are kept in the same fold."""
    N, T = pan.Y.shape
    C = _calibration_matrix(pan)
    zh = pan.ZH.reshape(N * T)
    lab = np.flatnonzero(pan.labeled)
    pi_i = np.asarray(pan.params.pi)[pan.strata]
    if groups is None:
        folds = np.array_split(frng.permutation(lab), N_FOLDS)
    else:
        ids = np.unique(groups[lab])
        part = np.array_split(frng.permutation(ids), N_FOLDS)
        fid = np.empty(groups.max() + 1, int)
        for k, p in enumerate(part):
            fid[p] = k
        folds = [lab[fid[groups[lab]] == k] for k in range(N_FOLDS)]
    coefs = np.zeros((N_FOLDS, C.shape[1]))
    Zo = np.zeros((N, T))
    for f, held in enumerate(folds):
        train = np.setdiff1d(lab, held)
        rr = rows(train, T)
        if weighted:
            sw = np.sqrt(np.repeat(1.0 / pi_i[train], T))
            coef, *_ = np.linalg.lstsq(C[rr] * sw[:, None], zh[rr] * sw, rcond=None)
        else:
            coef, *_ = np.linalg.lstsq(C[rr], zh[rr], rcond=None)
        coefs[f] = coef
        if held.size:
            Zo[held] = (C[rows(held, T)] @ coef).reshape(held.size, T)
    Zf = (C @ coefs.mean(axis=0)).reshape(N, T)
    return Zf, Zo, coefs, folds, C


def rect_weights(pan, kind):
    N = pan.Y.shape[0]
    R = pan.labeled.astype(float)
    pi_i = np.asarray(pan.params.pi)[pan.strata]
    if kind == "uniform":
        return R * N / R.sum()
    if kind == "ht":
        return R / pi_i
    if kind == "hajek":
        ht = R / pi_i
        return ht * N / ht.sum()
    if kind == "ps":
        s = pan.strata
        nS = np.bincount(s, minlength=3)
        nL = np.bincount(s[pan.labeled], minlength=3)
        if np.any(nL == 0):
            return None
        return R * (nS / nL)[s]
    raise ValueError(kind)


def rect(Wf, Wo, WH, y, rw, N, T):
    """Rectified solve; returns theta (vector), A, cluster scores g, per-cluster corrections c, residuals."""
    k = Wf.shape[1]
    rwt = np.repeat(rw, T)
    A = Wf.T @ Wf + (WH * rwt[:, None]).T @ WH - (Wo * rwt[:, None]).T @ Wo
    b = Wf.T @ y + (WH * rwt[:, None]).T @ y - (Wo * rwt[:, None]).T @ y
    th = np.linalg.solve(A, b)
    rf, rH, ro = y - Wf @ th, y - WH @ th, y - Wo @ th
    psiF = (Wf * rf[:, None]).reshape(N, T, k).sum(1)
    c = (WH * rH[:, None] - Wo * ro[:, None]).reshape(N, T, k).sum(1)
    return dict(th=th, A=A, g=psiF + rw[:, None] * c, c=c, rf=rf, ro=ro, rw=rw)


def se_of(A, g, N):
    Ai = np.linalg.inv(A)
    return float(math.sqrt((Ai @ (g.T @ g) @ Ai)[0, 0] * N / (N - 1.0)))


def ratio_term(kind, rw, c, strata, N):
    """Linearization of the random normalization: subtract (w_i - 1) times the (stratum) mean correction."""
    if kind == "ht":
        return np.zeros_like(c)
    if kind in ("uniform", "hajek"):
        m = (rw[:, None] * c).sum(0) / N
        return -(rw - 1.0)[:, None] * m[None, :]
    if kind == "ps":
        out = np.zeros_like(c)
        for j in range(3):
            sj = strata == j
            m = (rw[sj, None] * c[sj]).sum(0) / sj.sum()
            out[sj] = -(rw[sj] - 1.0)[:, None] * m[None, :]
        return out
    raise ValueError(kind)


def calib_term(pan, fit, Wf, Wo, coefs, folds, C, weighted):
    """First-order contribution of the fitted cross-fitted calibration to each cluster score.

    The moment depends on the fold coefficients beta_(-j) through the fold-averaged prediction (all units) and the
    out-of-fold predictions (labeled units of fold j). Its derivative with respect to beta_(-j) is J_f/K - J_j,
    and unit i (labeled, fold k) moves beta_(-j), j != k, by Q_(-j)^{-1} omega_i sum_t c_it e_it.
    """
    N, T = pan.Y.shape
    th = fit["th"]
    K = len(folds)
    Ct = np.column_stack([within_demean(C[:, j].reshape(N, T)).reshape(N * T) for j in range(C.shape[1])])

    def J(v, W, r):
        e1 = np.zeros((W.shape[1], 1)); e1[0, 0] = 1.0
        return e1 @ ((v * r) @ Ct)[None, :] - th[0] * (W * v[:, None]).T @ Ct

    Jf = J(np.ones(N * T), Wf, fit["rf"])
    zh = pan.ZH.reshape(N * T)
    pi_i = np.asarray(pan.params.pi)[pan.strata]
    om = (1.0 / pi_i) if weighted else np.ones(N)
    lab = np.flatnonzero(pan.labeled)
    fold_of = np.full(N, -1)
    for k, f in enumerate(folds):
        fold_of[f] = k
    out = np.zeros((N, Wf.shape[1]))
    rwt = np.repeat(fit["rw"], T)
    for j, held in enumerate(folds):
        v = np.zeros(N * T)
        v[rows(held, T)] = rwt[rows(held, T)]
        Jj = J(v, Wo, fit["ro"])
        train = np.setdiff1d(lab, held)
        rr = rows(train, T)
        Q = (C[rr] * np.repeat(om[train], T)[:, None]).T @ C[rr]
        G = (Jf / K - Jj) @ np.linalg.inv(Q)
        e = zh - C @ coefs[j]
        u = (C * (np.repeat(om, T) * e)[:, None]).reshape(N, T, -1).sum(1)   # omega_i sum_t c_it e_it
        out[train] += u[train] @ G.T
    return out


# ----------------------------------------------------------------------------------------------- procedures
def rect_all(pan, frng_factory, groups=None, want=("E-4", "E-5", "E-6", "E-6H", "PS-I", "E-12a", "E-12b", "E-12c"),
             linearize=True):
    """Point estimates and sealed/linearized SEs. frng_factory() returns a fresh fold RNG (same stream for the
    weighted and unweighted fits, as in the sealed code)."""
    N, T = pan.Y.shape
    y, W = designs(pan)
    WL, WH = W(pan.ZL), W(pan.ZH)
    G = int(pan.labeled.sum())
    res = {}
    spec = {"E-4": ("L", "uniform"), "E-6": ("L", "ht"), "E-6H": ("L", "hajek"), "PS-I": ("L", "ps"),
            "E-5": ("u", "uniform"), "E-12a": ("u", "ht"), "E-12b": ("w", "ht"), "E-12c": ("w", "uniform")}
    cf = {}
    if G >= 10 and any(spec[n][0] in "uw" for n in want):
        for wtd, key in ((False, "u"), (True, "w")):
            Zf, Zo, coefs, folds, C = crossfit(pan, frng_factory(), wtd, groups)
            cf[key] = (W(Zf), W(Zo), coefs, folds, C, wtd)
    for name in want:
        fam, kind = spec[name]
        if fam == "L" and G < 5 or fam in "uw" and G < 10:
            res[name] = None
            continue
        rw = rect_weights(pan, kind)
        if rw is None:
            res[name] = None
            continue
        if fam == "L":
            Wf = Wo = WL
        else:
            Wf, Wo = cf[fam][0], cf[fam][1]
        try:
            fit = rect(Wf, Wo, WH, y, rw, N, T)
        except np.linalg.LinAlgError:
            res[name] = None
            continue
        out = {"th": float(fit["th"][0]), "se": se_of(fit["A"], fit["g"], N), "G": G}
        if linearize:
            g2 = fit["g"] + ratio_term(kind, rw, fit["c"], pan.strata, N)
            out["se_ratio"] = se_of(fit["A"], g2, N)
            if fam in "uw":
                _, _, coefs, folds, C, wtd = cf[fam]
                g3 = g2 + calib_term(pan, fit, Wf, Wo, coefs, folds, C, wtd)
                out["se_lin"] = se_of(fit["A"], g3, N)
            else:
                out["se_lin"] = out["se_ratio"]
        out["_A"], out["_g"] = fit["A"], fit["g"]
        res[name] = out
    return res


def e2(pan):
    out = labeled_only_fit(pan, _period_design(*pan.Y.shape))
    if out is None:
        return None
    th, se, hw, g = out
    return {"th": float(th), "se": float(se), "G": int(g)}


def e10w(pan):
    """Design-weighted, power-tuned PPI++ (identity predictor, Horvitz-Thompson weights). Two steps: E-6 fit,
    lambda minimizing the estimated variance of the target coordinate (clipped to [0, 1]), refit; CR1 with
    lambda treated as fixed."""
    N, T = pan.Y.shape
    if pan.labeled.sum() < 5:
        return None
    y, W = designs(pan)
    WL, WH = W(pan.ZL), W(pan.ZH)
    k = WL.shape[1]
    rw = rect_weights(pan, "ht")
    rwt = np.repeat(rw, T)

    def solve(lam):
        A = lam * WL.T @ WL + (WH * rwt[:, None]).T @ WH - lam * (WL * rwt[:, None]).T @ WL
        b = lam * WL.T @ y + (WH * rwt[:, None]).T @ y - lam * (WL * rwt[:, None]).T @ y
        return np.linalg.solve(A, b), A

    try:
        th1, _ = solve(1.0)
        rH, rL = y - WH @ th1, y - WL @ th1
        psiH = (WH * rH[:, None]).reshape(N, T, k).sum(1)
        psiL = (WL * rL[:, None]).reshape(N, T, k).sum(1)
        Hi = np.linalg.inv((WH * rwt[:, None]).T @ WH)
        a = (rw[:, None] * psiH) @ Hi[0]
        bb = ((1.0 - rw)[:, None] * psiL) @ Hi[0]
        lam = float(np.clip(-np.cov(a, bb)[0, 1] / np.var(bb, ddof=1), 0.0, 1.0))
        th, A = solve(lam)
        rH, rL = y - WH @ th, y - WL @ th
        g = (lam * (WL * rL[:, None]) + rwt[:, None] * (WH * rH[:, None] - lam * WL * rL[:, None])).reshape(N, T, k).sum(1)
        return {"th": float(th[0]), "se": se_of(A, g, N), "lam": lam, "G": int(pan.labeled.sum())}
    except np.linalg.LinAlgError:
        return None


def ptd(pan, rng, B=1000):
    """Predict-then-debias (Kluger et al. 2025) for the within regression: theta_dot on labeled units with Z_H and
    weights R/pi; gamma_dot on labeled units with Z_L and the same weights; gamma_circ on unlabeled units with Z_L
    and weights (1-R)/(1-pi). Diagonal tuning estimated from a cluster bootstrap (their Algorithm 5, whole units
    resampled); percentile interval from the bootstrap PTD draws with the tuning fixed."""
    N, T = pan.Y.shape
    if pan.labeled.sum() < 5:
        return None
    y, W = designs(pan)
    WL, WH = W(pan.ZL), W(pan.ZH)
    k = WL.shape[1]
    R = pan.labeled.astype(float)
    pi_i = np.asarray(pan.params.pi)[pan.strata]
    wdot, wcirc = R / pi_i, (1.0 - R) / (1.0 - pi_i)

    def cp(Wm):
        X3 = Wm.reshape(N, T, k)
        return np.einsum("ntj,ntl->njl", X3, X3).reshape(N, k * k), np.einsum("ntj,nt->nj", X3, y.reshape(N, T))

    XH, yH = cp(WH)
    XL, yL = cp(WL)

    def solve_batch(Mw, Xc, yc):
        A = (Mw @ Xc).reshape(-1, k, k)
        b = Mw @ yc
        out = np.full(A.shape[0], np.nan)
        ok = np.abs(np.linalg.det(A)) > 1e-10
        out[ok] = np.linalg.solve(A[ok], b[ok][..., None])[:, 0, 0]
        return out

    one = np.ones((1, N))
    th_dot = solve_batch(one * wdot, XH, yH)[0]
    ga_dot = solve_batch(one * wdot, XL, yL)[0]
    ga_cir = solve_batch(one * wcirc, XL, yL)[0]
    idx = rng.integers(0, N, size=(B, N))
    M = np.zeros((B, N))
    np.add.at(M, (np.repeat(np.arange(B), N), idx.ravel()), 1.0)
    tb = solve_batch(M * wdot, XH, yH)
    gb = solve_batch(M * wdot, XL, yL)
    cb = solve_batch(M * wcirc, XL, yL)
    ok = np.isfinite(tb) & np.isfinite(gb) & np.isfinite(cb)
    if ok.mean() < 0.95:
        return None
    tb, gb, cb = tb[ok], gb[ok], cb[ok]
    cov = np.cov(np.vstack([tb, gb, cb]))
    om = cov[0, 1] / (cov[1, 1] + cov[2, 2])
    est = om * ga_cir + (th_dot - om * ga_dot)
    draws = om * cb + (tb - om * gb)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"th": float(est), "lo": float(lo), "hi": float(hi), "se": float(draws.std(ddof=1)), "omega": float(om),
            "G": int(R.sum())}


def wild_interval(A, g, th, N, rng, B=399):
    """Wild cluster score bootstrap (Rademacher on cluster scores at the estimate, studentized by the fixed CR1
    SE, which Rademacher weights leave unchanged): symmetric interval from the 95% quantile of |delta*|."""
    Ai = np.linalg.inv(A)
    eps = rng.choice([-1.0, 1.0], size=(B, N))
    d = (eps @ g) @ Ai[0]
    q = np.quantile(np.abs(d), 0.95)
    return th - q, th + q


# ----------------------------------------------------------------------------------------------- proxy strata
def restratify(pan, kind, rng, rel=0.5):
    """Validation strata from information available before annotation. The outcome model (slope heterogeneity by
    the construct-level stratum) is unchanged; only the validation design and the weights use the new strata.
      machine: tertiles of the unit mean of the machine score Z_L (ties broken at random);
      proxy:   tertiles of a noisy proxy of the unit mean of Z_H with reliability `rel` (correlation sqrt(rel)).
    Units are then validated independently with probability pi of their new stratum."""
    N = pan.Y.shape[0]
    if kind == "machine":
        score = pan.ZL.mean(axis=1) + 1e-9 * rng.standard_normal(N)
    elif kind == "proxy":
        zb = pan.ZH.mean(axis=1)
        zs = (zb - zb.mean()) / zb.std()
        score = math.sqrt(rel) * zs + math.sqrt(1.0 - rel) * rng.standard_normal(N)
    else:
        raise ValueError(kind)
    ranks = np.argsort(np.argsort(score))
    s_new = (3 * ranks) // N
    lab = rng.random(N) < np.asarray(pan.params.pi)[s_new]
    return replace(pan, strata=s_new, labeled=lab)


def e2_parts(pan):
    """Labeled-only IPW (E-2) with its labeled-cluster scores, for the wild bootstrap."""
    N, T = pan.Y.shape
    lab = np.flatnonzero(pan.labeled)
    G = lab.size
    if G < 5:
        return None
    y, W = designs(pan)
    WH = W(pan.ZH)
    k = WH.shape[1]
    w = 1.0 / np.asarray(pan.params.pi)[pan.strata]
    rr = rows(lab, T)
    Xl, yl, wl = WH[rr], y[rr], np.repeat(w[lab], T)
    A = (Xl * wl[:, None]).T @ Xl
    th = np.linalg.solve(A, (Xl * wl[:, None]).T @ yl)
    g = (Xl * (wl * (yl - Xl @ th))[:, None]).reshape(G, T, k).sum(1)
    Ai = np.linalg.inv(A)
    se = math.sqrt((Ai @ (g.T @ g) @ Ai)[0, 0] * G / (G - 1.0))
    return {"th": float(th[0]), "se": se, "A": A, "g": g, "G": G}
