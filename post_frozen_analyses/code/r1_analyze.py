"""Summaries of the R1 revision analyses: CSV tables in ../analysis and a JSON of headline numbers.

Run after r1_run.py has completed every module:  python3 -I -B r1_analyze.py
Monte Carlo standard errors: binomial for rates; SD/sqrt(n) for means; nonparametric bootstrap over replications
(B = 500, fixed seed) for ratios and differences of SDs.
"""
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
AN = os.path.join(os.path.dirname(HERE), "analysis")
os.makedirs(AN, exist_ok=True)
Z = 1.959963984540054
RNG = np.random.default_rng(20261113)


def complete(module):
    p = os.path.join(OUT, module, "cells.json")
    if not os.path.exists(p):
        return False
    meta = json.load(open(p))
    missing = [k for k in meta if not os.path.exists(os.path.join(OUT, module, k + ".npz"))]
    if missing:
        print(f"{module}: {len(missing)} of {len(meta)} cells not yet computed; skipped")
    return not missing


def load(module):
    meta = json.load(open(os.path.join(OUT, module, "cells.json")))
    out = []
    for k, m in meta.items():
        d = dict(np.load(os.path.join(OUT, module, k + ".npz")))
        cell = m["cell"]
        c, s = (cell if isinstance(cell, list) else (cell, None))
        kv = dict(x.split("=", 1) for x in c.split("|"))
        out.append({"key": k, "cfg": c, "strat": s, "kv": kv, "theta0": m["theta0"], "n": m["n"], "d": d})
    return out


def gamma(cell):
    return float(cell["kv"]["g"])


def cov(th, lo, hi, t0):
    ok = np.isfinite(th) & np.isfinite(lo) & np.isfinite(hi)
    return (lo[ok] <= t0) & (t0 <= hi[ok]), ok


def rate(x):
    x = np.asarray(x, float)
    n = x.size
    p = x.mean() if n else float("nan")
    return p, math.sqrt(p * (1 - p) / n) if n else float("nan")


def mean_se(x):
    x = x[np.isfinite(x)]
    return x.mean(), x.std(ddof=1) / math.sqrt(x.size)


def boot(fn, *arrs, B=500):
    n = arrs[0].size
    vals = [fn(*[a[i] for a in arrs]) for i in (RNG.integers(0, n, n) for _ in range(B))]
    return float(np.std(vals, ddof=1))


def write(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(AN, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


HEAD = {}

# ======================================================================================== A1
if complete("A1"):
    cells = load("A1")
    rows = []
    PROCS = ["E-4", "E-5", "E-12c", "E-6H", "PS-I", "E-6", "E-12a", "E-12b", "E-2"]
    for c in cells:
        d, t0 = c["d"], c["theta0"]
        for p in PROCS:
            th = d.get(f"{p}|th")
            if th is None:
                continue
            ok = np.isfinite(th)
            row = {"cfg": c["cfg"], "gamma_W": gamma(c), "procedure": p, "n_ok": int(ok.sum()),
                   "bias": fmt(th[ok].mean() - t0, 5), "emp_sd": fmt(th[ok].std(ddof=1), 5)}
            for v in ("se", "se_ratio", "se_lin"):
                se = d.get(f"{p}|{v}")
                if se is None:
                    continue
                cv, _ = cov(th, th - Z * se, th + Z * se, t0)
                if p == "E-2":    # published E-2 interval uses t(G_S - 1)
                    from scipy.stats import t as tdist
                    q = tdist.ppf(0.975, d["G"] - 1)
                    cv, _ = cov(th, th - q * se, th + q * se, t0)
                row[f"meanSE_{v}"] = fmt(np.nanmean(se), 5)
                row[f"ratio_{v}"] = fmt(np.nanmean(se) / th[ok].std(ddof=1), 4)
                row[f"cover_{v}"] = fmt(cv.mean(), 4)
            rows.append(row)
    write("A1_variance_by_cell.csv", rows)
    # summaries: primary non-null and null cells (gamma 0.15 / 0), and Experiment 2 cells separately
    prim = {c["cfg"] for c in cells if "pi=0.15,0.3,0.6" in c["cfg"] and "tau=0.1|" in c["cfg"] and "lam=1.2,1.05,0.45" in c["cfg"]}
    summ = []
    for p in PROCS:
        for g in (0.15, 0.0):
            rr = [r for r in rows if r["procedure"] == p and r["cfg"] in prim and r["gamma_W"] == g]
            if not rr:
                continue
            s = {"procedure": p, "gamma_W": g, "cells": len(rr), "emp_sd": fmt(np.mean([r["emp_sd"] for r in rr]), 5)}
            for v in ("se", "se_ratio", "se_lin"):
                if f"meanSE_{v}" in rr[0]:
                    s[f"meanSE_{v}"] = fmt(np.mean([r[f"meanSE_{v}"] for r in rr]), 5)
                    key = "cover" if g else "type1"
                    vals = [r[f"cover_{v}"] for r in rr]
                    s[f"{key}_{v}_{'median' if g else 'mean'}"] = fmt(np.median(vals) if g else 1 - np.mean(vals), 4)
                    s[f"{key}_{v}_min"] = fmt(min(vals) if g else 1 - max(vals), 4)
                    s[f"{key}_{v}_max"] = fmt(max(vals) if g else 1 - min(vals), 4)
            summ.append(s)
    write("A1_variance_primary_summary.csv", summ)
    HEAD["A1"] = {f"{s['procedure']}|{s['gamma_W']}": s for s in summ}

# ======================================================================================== A2
if complete("A2"):
    rows = []
    for c in load("A2"):
        d = c["d"]
        for p in ["E-4", "E-5", "E-12c", "E-6H", "PS-I", "E-6", "E-12a", "E-12b"]:
            th = d[f"{p}|th"]
            ok = np.isfinite(th)
            bs = d[f"{p}|boot_sd"]
            rows.append({"cfg": c["cfg"], "procedure": p, "n": int(ok.sum()),
                         "emp_sd": fmt(th[ok].std(ddof=1), 5),
                         "emp_sd_mcse": fmt(boot(lambda a: a.std(ddof=1), th[ok]), 5),
                         "meanSE_sealed": fmt(np.nanmean(d[f"{p}|se"]), 5),
                         "meanSE_lin": fmt(np.nanmean(d[f"{p}|se_lin"]), 5),
                         "mean_boot_sd": fmt(np.nanmean(bs), 5), "boot_ok": int(np.isfinite(bs).sum())})
    write("A2_bootstrap_check.csv", rows)
    HEAD["A2"] = rows

# ======================================================================================== B
if complete("B"):
    cells = load("B")
    rows = []
    for c in cells:
        d, t0 = c["d"], c["theta0"]
        for p in ("E-1", "E-7", "E-12a", "E-12b"):
            if f"{p}|future|th" not in d:
                continue
            arm = {a: d[f"{p}|{a}|th"] for a in ("future", "none", "indep")}
            ok = np.all([np.isfinite(v) for v in arm.values()], axis=0)
            r = {"cfg": c["cfg"], "T": int(c["kv"]["T"]), "rho_e": float(c["kv"]["rho"]), "kappa": float(c["kv"]["kappa"]),
                 "gamma_W": gamma(c), "procedure": p, "n": int(ok.sum())}
            for name, (a, b) in {"future_minus_none": ("future", "none"), "indep_minus_none": ("indep", "none"),
                                 "future_minus_indep": ("future", "indep")}.items():
                m, s = mean_se(arm[a][ok] - arm[b][ok])
                r[name], r[name + "_mcse"] = fmt(m, 5), fmt(s, 5)
            for a in ("future", "none", "indep"):
                r[f"bias_{a}"] = fmt(arm[a][ok].mean() - t0, 5)
                se = d.get(f"{p}|{a}|se")
                if se is not None:
                    q = 1.959963984540054
                    if p == "E-1":
                        from scipy.stats import t as tdist
                        q = tdist.ppf(0.975, int(c["kv"]["N"]) - 1)
                    cv, _ = cov(arm[a], arm[a] - q * se, arm[a] + q * se, t0)
                    r[f"cover_{a}"] = fmt(cv.mean(), 4)
            rows.append(r)
    write("B_contamination_control.csv", rows)
    # Experiment 3: leakage-specific scaling for E-1 (contaminated non-null cells of Experiment 3)
    sys.path.insert(0, HERE)
    import r1_core
    CFG, _ = r1_core.load()
    X3CFG = {k for k, v in CFG.items() if "X3" in v.modules and float(v.kappa) > 0}
    fits = {}
    for comp in ("future_minus_none", "future_minus_indep", "indep_minus_none"):
        sel = sorted([r for r in rows if r["procedure"] == "E-1" and r["cfg"] in X3CFG and r["gamma_W"] == 0.15],
                     key=lambda r: r["T"])
        T = np.array([r["T"] for r in sel], float)
        dv = np.array([r[comp] for r in sel], float)
        se = np.array([r[comp + "_mcse"] for r in sel], float)
        y = np.log(np.abs(dv))
        w = (np.abs(dv) / se) ** 2           # delta method: var(log|d|) = (se/|d|)^2
        X = np.column_stack([np.ones_like(T), np.log(T)])
        Wm = np.diag(w)
        cov_b = np.linalg.inv(X.T @ Wm @ X)
        b = cov_b @ X.T @ Wm @ y
        fits[comp] = {"slope": float(b[1]), "ci95": [float(b[1] - Z * math.sqrt(cov_b[1, 1])),
                                                       float(b[1] + Z * math.sqrt(cov_b[1, 1]))],
                      "T": T.tolist(), "d": dv.tolist(), "d_mcse": se.tolist(), "T_abs_d": (T * np.abs(dv)).tolist(),
                      "kappa": [r["kappa"] for r in sel]}
    json.dump(fits, open(os.path.join(AN, "B_exp3_scaling.json"), "w"), indent=1)
    HEAD["B_exp3"] = fits

# ======================================================================================== C, D1
def comp_table(module, procs):
    cells = load(module)
    rows = []
    for c in cells:
        d, t0 = c["d"], c["theta0"]
        n = c["n"]
        e2 = d["E-2|th"]
        for p in procs:
            th = d.get(f"{p}|th")
            if th is None:
                continue
            ok = np.isfinite(th)
            if p == "PTD":
                lo, hi = d["PTD|lo"], d["PTD|hi"]
            elif p == "E-2":
                from scipy.stats import t as tdist
                q = tdist.ppf(0.975, d["G"] - 1)
                lo, hi = th - q * d["E-2|se"], th + q * d["E-2|se"]
            else:
                se = d.get(f"{p}|se_lin", d[f"{p}|se"]) if p in ("E-4", "E-5", "E-12c") else d[f"{p}|se"]
                lo, hi = th - Z * se, th + Z * se
            cv, okc = cov(th, lo, hi, t0)
            both = ok & np.isfinite(e2)
            rm = lambda a, b: math.sqrt(np.mean((a - t0) ** 2)) / math.sqrt(np.mean((b - t0) ** 2))
            rr = rm(th[both], e2[both]) if p != "E-2" else 1.0
            rows.append({"cfg": c["cfg"], "strat": c["strat"] or "oracle", "gamma_W": gamma(c), "procedure": p,
                         "fail_rate": fmt(1 - ok.mean(), 4), "bias": fmt(th[ok].mean() - t0, 5),
                         "bias_mcse": fmt(th[ok].std(ddof=1) / math.sqrt(ok.sum()), 5),
                         "rmse": fmt(math.sqrt(np.mean((th[ok] - t0) ** 2)), 5),
                         "R_rmse_vs_E2": fmt(rr, 4),
                         "R_rmse_mcse": fmt(boot(lambda a, b: rm(a, b), th[both], e2[both], B=300), 4) if p != "E-2" else "",
                         "coverage" if gamma(c) else "type1": fmt(cv.mean() if gamma(c) else 1 - cv.mean(), 4),
                         "median_width": fmt(np.median((hi - lo)[okc]), 4)})
    return rows


if complete("C"):
    rows = comp_table("C", ["E-2", "E-6", "E-10w", "PTD", "E-12a", "E-12b"])
    write("C_comparators.csv", rows)
    HEAD["C"] = rows
if complete("D1"):
    rows = comp_table("D1", ["E-2", "E-4", "E-5", "E-12c", "E-6", "E-10w", "PTD", "E-12a", "E-12b"])
    write("D1_proxy_strata.csv", rows)
    HEAD["D1"] = rows

# ======================================================================================== D2
if complete("D2"):
    from scipy.stats import t as tdist
    rows = []
    for c in load("D2"):
        d, t0, n = c["d"], c["theta0"], c["n"]
        G = d["G"]
        spec = {
            "E-2 (CR1, t)": ("E-2", lambda th, se: (th - tdist.ppf(0.975, G - 1) * se, th + tdist.ppf(0.975, G - 1) * se)),
            "E-2 (wild)": ("E-2", "w"),
            "E-12a (CR1, normal)": ("E-12a", lambda th, se: (th - Z * se, th + Z * se)),
            "E-12a (CR1, t)": ("E-12a", lambda th, se: (th - tdist.ppf(0.975, G - 1) * se, th + tdist.ppf(0.975, G - 1) * se)),
            "E-12a (wild)": ("E-12a", "w"),
            "E-12b (CR1, normal)": ("E-12b", lambda th, se: (th - Z * se, th + Z * se)),
            "E-12b (CR1, t)": ("E-12b", lambda th, se: (th - tdist.ppf(0.975, G - 1) * se, th + tdist.ppf(0.975, G - 1) * se)),
            "E-12b (wild)": ("E-12b", "w"),
            "PTD (percentile)": ("PTD", "p"),
        }
        for name, (p, how) in spec.items():
            th = d[f"{p}|th"]
            if how == "w":
                lo, hi = d[f"{p}|wlo"], d[f"{p}|whi"]
            elif how == "p":
                lo, hi = d["PTD|lo"], d["PTD|hi"]
            else:
                lo, hi = how(th, d[f"{p}|se"])
            made = np.isfinite(th) & np.isfinite(lo) & np.isfinite(hi)
            hit = made & (lo <= t0) & (t0 <= hi)
            pm, pm_se = rate(made)
            r = {"cfg": c["cfg"], "f": {"0.012857142857,0.025714285714,0.051428571429": 0.03,
                                        "0.021428571429,0.042857142857,0.085714285714": 0.05,
                                        "0.042857142857,0.085714285714,0.171428571429": 0.10}[c["kv"]["pi"]],
                 "sd_e": float(c["kv"]["sde"]), "gamma_W": gamma(c), "interval": name,
                 "p_interval": fmt(pm, 4), "p_interval_mcse": fmt(pm_se, 4)}
            cc, cc_se = rate(hit[made])
            if gamma(c):
                r["cond_coverage"], r["cond_coverage_mcse"] = fmt(cc, 4), fmt(cc_se, 4)
                r["p_covering_interval"] = fmt(hit.mean(), 4)
            else:
                r["cond_type1"], r["cond_type1_mcse"] = fmt(1 - cc, 4), fmt(cc_se, 4)
            r["median_width"] = fmt(np.median((hi - lo)[made]), 4) if made.any() else ""
            r["bias_cond"] = fmt(np.mean(th[made]) - t0, 5) if made.any() else ""
            rows.append(r)
    write("D2_sparse_inference.csv", rows)
    HEAD["D2"] = rows

json.dump(HEAD, open(os.path.join(AN, "R1_headline.json"), "w"), indent=1, default=str)
print("wrote", sorted(os.listdir(AN)))
