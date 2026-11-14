"""Check added after the production runs: how often the floor of PS-TDS-I binds in the Module A1 cells.

Module A1 computes PS-I, the post-stratified identity-predictor rectifier without the stabilization floor. This script
compares PS-I with the published PS-TDS-I, replication by replication, on the same published panels, and counts the
replications in which the floor bound. It reads the sealed raw files of FINAL_V1 (read only) and the A1 outputs, and
writes ../analysis/A1_ps_tds_floor_check.json.

Run: python3 -I -B check_ps_floor.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.dirname(HERE)
RAW = os.environ.get("FV1_RAW", "/mnt/user-data/outputs/Final_Study_v1/run/raw")
Z = 1.959963984540054


def main():
    meta = json.load(open(os.path.join(R1, "out", "A1", "cells.json")))
    raw = {}
    for p in sorted(glob.glob(os.path.join(RAW, "C*.npz"))):
        d = np.load(p, allow_pickle=True)
        m = json.loads(str(d["meta_det"]))
        raw[m["config_id"]] = (p, m)
    groups = {"primary_grid": {}, "experiment_2_only": {}}
    for key, v in meta.items():
        cell = v["cell"][0] if isinstance(v["cell"], list) else v["cell"]
        path, m = raw[cell]
        grp = "primary_grid" if "P" in m["modules"] else "experiment_2_only"
        nonnull = dict(x.split("=", 1) for x in cell.split("|"))["g"] != "0"   # gamma_W = 0: null companion
        d = np.load(path, allow_pickle=True)
        k = list(d["est_names"]).index("PS-TDS-I")
        sel = d["rows_est"] == k
        r, est, se, bind = d["rows_r"][sel], d["rows_estimate"][sel], d["rows_se"][sel], d["rows_bind"][sel]
        a = np.load(os.path.join(R1, "out", "A1", key + ".npz"))
        th, s = a["PS-I|th"], a["PS-I|se"]
        n = len(th)
        keep = r < n
        r, est, se, bind = r[keep], est[keep], se[keep], bind[keep]
        bound = bind == 1
        free = ~bound & np.isfinite(est) & np.isfinite(th[r])
        th0 = float(v["theta0"])
        cov_sealed = np.abs(est[bound] - th0) <= Z * se[bound]
        cov_r1 = np.abs(th[r][bound] - th0) <= Z * s[r][bound]
        rej_sealed = np.abs(est[bound]) > Z * se[bound]
        rej_r1 = np.abs(th[r][bound]) > Z * s[r][bound]
        g = groups[grp].setdefault("nonnull" if nonnull else "null",
                                    {"cells": 0, "replications": 0, "floor_binds": 0, "cells_with_a_bind": 0,
                                     "max_abs_diff_estimate_unbound": 0.0, "max_abs_diff_se_unbound": 0.0,
                                     "sum_se_diff_bound": 0.0, "coverage_changed": 0, "rejection_of_zero_changed": 0})
        g["cells"] += 1
        g["replications"] += int(len(r))
        g["floor_binds"] += int(bound.sum())
        g["cells_with_a_bind"] += int(bound.any())
        if free.any():
            g["max_abs_diff_estimate_unbound"] = max(g["max_abs_diff_estimate_unbound"],
                                                     float(np.max(np.abs(est[free] - th[r][free]))))
            g["max_abs_diff_se_unbound"] = max(g["max_abs_diff_se_unbound"],
                                               float(np.max(np.abs(se[free] - s[r][free]))))
        g["sum_se_diff_bound"] += float(np.sum(se[bound] - s[r][bound]))
        g["coverage_changed"] += int(np.sum(cov_sealed != cov_r1))
        g["rejection_of_zero_changed"] += int(np.sum(rej_sealed != rej_r1))
    out = {"description": "PS-TDS-I (published, with floor) against PS-I (Module A1, without floor) on the same "
                          "published panels; a bind is a replication in which the floor changed PS-TDS-I.",
           "groups": groups}
    pg = groups["primary_grid"]
    nn = pg.get("nonnull", {})
    out["primary_grid_total"] = {
        "replications": sum(x["replications"] for x in pg.values()),
        "floor_binds": sum(x["floor_binds"] for x in pg.values()),
        # largest change in a Table S13 entry if the published PS-TDS-I values replaced PS-I in the bound replications:
        # the mean SE is averaged over 12 non-null cells of 5,000 replications; coverage and Type I error are rates
        "max_change_mean_se_table_S13": abs(nn.get("sum_se_diff_bound", 0.0)) / max(nn.get("replications", 1), 1),
        "max_change_rate_per_cell": 1.0 / 5000,
    }
    with open(os.path.join(R1, "analysis", "A1_ps_tds_floor_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
