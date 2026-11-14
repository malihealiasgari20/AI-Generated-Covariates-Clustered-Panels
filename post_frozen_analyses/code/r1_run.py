"""Run the R1 revision modules (see ../R1_ANALYSIS_PLAN.md). Resumable: one .npz per module and cell.

  python3 -I -B r1_run.py <module> [--test] [--reps N] [--workers 2]

Modules: A1 variance re-analysis; A2 pairs cluster bootstrap; B contamination control; C comparators;
D1 proxy strata; D2 sparse validation. --test uses throwaway R1TEST seeds and writes to ../out_test.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import r1_core as R  # noqa: E402

CFG, TH0 = R.load()
FIX = "N=300|T=8|icc=0.3|rho=0.6|"
BASE = FIX + "tau=0.1|pi=0.15,0.3,0.6|sde=0.55|sdW=0.62|lam=1.2,1.05,0.45|"


def cid(s):
    assert s in CFG, s
    return s


def parse(c):
    return dict(kv.split("=", 1) for kv in c.split("|"))


def by_module(m):
    return [c for c, v in CFG.items() if m in v.modules]


PIF = {0.03: "0.012857142857,0.025714285714,0.051428571429", 0.05: "0.021428571429,0.042857142857,0.085714285714",
       0.10: "0.042857142857,0.085714285714,0.171428571429", 0.35: "0.15,0.3,0.6"}


def x4(f, sde, g):
    return cid(f"{FIX}tau=0.1|pi={PIF[f]}|sde={sde}|sdW=0.62|lam=1.2,1.05,0.45|g={g}|kappa=0")


A1B1 = lambda g: cid(f"{FIX}tau=0.3|pi=0.15,0.3,0.6|sde=0.55|sdW=0.62|lam=1.2,1.05,0.45|g={g}|kappa=0")
A0B1 = lambda g: cid(f"{FIX}tau=0.3|pi=0.35,0.35,0.35|sde=0.55|sdW=0.62|lam=1.2,1.05,0.45|g={g}|kappa=0")
BASEC = lambda g, k=0: cid(f"{BASE}g={g}|kappa={k}")

CELLS = {
    "A1": by_module("P") + by_module("X2"),
    "A2": [BASEC(0.15), A1B1(0.15)],
    "B": [c for c in by_module("L") + by_module("X3") if float(CFG[c].kappa) > 0],
    "C": [BASEC(0.15), BASEC(0), BASEC(0.15, 0.6), BASEC(0, 0.6), A1B1(0.15), A1B1(0), A0B1(0.15), A0B1(0),
          x4(0.10, 0.55, 0.15), x4(0.10, 0.55, 0), x4(0.35, 0.15, 0.15), x4(0.35, 0.15, 0)],
    "D1": [(c, s) for c in (BASEC(0.15), BASEC(0), A1B1(0.15), A1B1(0)) for s in ("oracle", "machine", "proxy")],
    "D2": [x4(f, sde, g) for f in (0.03, 0.05, 0.10) for sde in (0.55, 0.8) for g in (0.15, 0)],
}
REPS = {"A1": None, "A2": 400, "B": None, "C": 2000, "D1": 2000, "D2": 2000}   # None: the published count
BOOT_B = 199


def n_published(c):
    v = CFG[c]
    return max(int(v.R["FULL"]), int(v.R.get("X3", 0)))


def put(store, key, r, val, n):
    if key not in store:
        store[key] = np.full(n, np.nan)
    store[key][r] = val


def record(store, pref, out, r, n, fields=("th", "se", "se_ratio", "se_lin")):
    for f in fields:
        put(store, f"{pref}|{f}", r, np.nan if out is None or f not in out else out[f], n)


def run_cell(task):
    module, cell, r0, r1, n, test, outdir = task
    c, strat = (cell if isinstance(cell, tuple) else (cell, None))
    cfg = CFG[c]
    st = {}
    t0 = time.time()
    for r in range(r0, r1):
        pan = R.panel(cfg, r, test)
        D = R._period_design(*pan.Y.shape)
        fr = lambda: R.fold_rng(cfg, r, test)
        put(st, "G", r, pan.labeled.sum(), n)
        if module == "A1":
            res = R.rect_all(pan, fr)
            for k, v in res.items():
                record(st, k, v, r, n)
            record(st, "E-2", R.e2(pan), r, n, ("th", "se"))
        elif module == "A2":
            res = R.rect_all(pan, fr)
            for k, v in res.items():
                record(st, k, v, r, n)
            rng = R.r1_rng("boot", cfg.seed_cell_id, r, test)
            bt = {}
            for b in range(BOOT_B):
                idx = rng.integers(0, pan.Y.shape[0], pan.Y.shape[0])
                pb = R.take_units(pan, idx)
                rb = R.rect_all(pb, lambda: R.r1_rng(f"bootfolds{b}", cfg.seed_cell_id, r, test), groups=idx,
                                linearize=False)
                for k, v in rb.items():
                    bt.setdefault(k, []).append(np.nan if v is None else v["th"])
            for k, v in bt.items():
                v = np.array(v)
                put(st, f"{k}|boot_sd", r, np.nanstd(v, ddof=1) if np.isfinite(v).mean() >= 0.95 else np.nan, n)
        elif module == "B":
            p = cfg.p
            Z0 = R.discretize(pan.Zstar - pan.C, p)
            zr = R.r1_rng("indshock", cfg.seed_cell_id, r, test).standard_normal(pan.eps_next.shape) * p.sigma_eps
            Zi = R.discretize(pan.Zstar - pan.C + pan.kappa_t[None, :] * zr, p)
            arms = {"future": pan, "none": replace(pan, ZL=Z0), "indep": replace(pan, ZL=Zi)}
            for a, pa in arms.items():
                th1, se1 = R.naive_fit(pa, D)[:2]
                put(st, f"E-1|{a}|th", r, th1, n); put(st, f"E-1|{a}|se", r, se1, n)
                th7, se7, _ = R.rc_fit(pa, D)
                put(st, f"E-7|{a}|th", r, th7, n); put(st, f"E-7|{a}|se", r, se7, n)
                if "L" in cfg.modules and r < 2000:
                    res = R.rect_all(pa, fr, want=("E-12a", "E-12b"), linearize=False)
                    for k, v in res.items():
                        record(st, f"{k}|{a}", v, r, n, ("th", "se"))
        elif module in ("C", "D1"):
            if module == "D1" and strat != "oracle":
                pan = R.restratify(pan, strat, R.r1_rng(f"strata-{strat}", cfg.seed_cell_id, r, test))
                put(st, "G", r, pan.labeled.sum(), n)
            want = ("E-6", "E-12a", "E-12b") if module == "C" else ("E-4", "E-5", "E-6", "E-12a", "E-12b", "E-12c")
            res = R.rect_all(pan, fr, want=want)
            for k, v in res.items():
                record(st, k, v, r, n)
            record(st, "E-2", R.e2(pan), r, n, ("th", "se"))
            record(st, "E-10w", R.e10w(pan), r, n, ("th", "se", "lam"))
            pt = R.ptd(pan, R.r1_rng(f"ptd-{strat}", cfg.seed_cell_id, r, test))
            record(st, "PTD", pt, r, n, ("th", "se", "lo", "hi", "omega"))
        elif module == "D2":
            N = pan.Y.shape[0]
            G = int(pan.labeled.sum())
            e2p = R.e2_parts(pan)
            if e2p is not None:
                put(st, "E-2|th", r, e2p["th"], n); put(st, "E-2|se", r, e2p["se"], n)
                lo, hi = R.wild_interval(e2p["A"], e2p["g"], e2p["th"], e2p["g"].shape[0],
                                         R.r1_rng("wild-E-2", cfg.seed_cell_id, r, test))
                put(st, "E-2|wlo", r, lo, n); put(st, "E-2|whi", r, hi, n)
            res = R.rect_all(pan, fr, want=("E-12a", "E-12b"), linearize=False)
            for k, v in res.items():
                record(st, k, v, r, n, ("th", "se"))
                if v is not None:
                    lo, hi = R.wild_interval(v["_A"], v["_g"], v["th"], N, R.r1_rng(f"wild-{k}", cfg.seed_cell_id, r, test))
                    put(st, f"{k}|wlo", r, lo, n); put(st, f"{k}|whi", r, hi, n)
            pt = R.ptd(pan, R.r1_rng("ptd", cfg.seed_cell_id, r, test))
            record(st, "PTD", pt, r, n, ("th", "se", "lo", "hi"))
    return module, cell, r0, r1, st, time.time() - t0


def key_of(cell):
    s = json.dumps(cell)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module", choices=sorted(CELLS))
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=250)
    a = ap.parse_args()
    outdir = os.path.join(os.path.dirname(HERE), "out_test" if a.test else "out", a.module)
    os.makedirs(outdir, exist_ok=True)
    tasks, meta = [], {}
    for cell in CELLS[a.module]:
        c = cell[0] if isinstance(cell, tuple) else cell
        n = a.reps or REPS[a.module] or n_published(c)
        path = os.path.join(outdir, key_of(cell) + ".npz")
        meta[key_of(cell)] = {"cell": cell, "n": n, "theta0": TH0[c]}
        if os.path.exists(path):
            continue
        ch = 50 if a.module == "A2" else a.chunk
        for r0 in range(0, n, ch):
            tasks.append((a.module, cell, r0, min(n, r0 + ch), n, a.test, outdir))
    json.dump(meta, open(os.path.join(outdir, "cells.json"), "w"), indent=1)
    parts = {}
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for module, cell, r0, r1, st, dt in pool.imap_unordered(run_cell, tasks):
            k = key_of(cell)
            parts.setdefault(k, []).append(st)
            n = meta[k]["n"]
            done = sum(1 for t in tasks if key_of(t[1]) == k)
            if len(parts[k]) == done:
                merged = {}
                for s in parts[k]:
                    for name, arr in s.items():
                        merged[name] = np.where(np.isnan(merged[name]), arr, merged[name]) if name in merged else arr
                np.savez_compressed(os.path.join(outdir, k + ".npz"), **merged)
                print(f"[{time.time() - t0:7.0f}s] {a.module} {meta[k]['cell']} done", flush=True)
    print("ALL DONE", a.module, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
