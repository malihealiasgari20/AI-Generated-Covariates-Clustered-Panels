# R1 revision analyses: plan fixed before the production runs

Fixed on 13 November 2026, before any production replication of these analyses was computed. The SHA-256 of this file is recorded in `R1_PLAN_SEAL.txt`.

## Status relative to FINAL_V1

These analyses respond to the internal review of 10 November 2026. They were designed **after** the FINAL_V1 results were known, and after two of the reviewer's diagnostics had been reproduced on 2,000 and 1,000 published-seed replications (`verify_reviewer.py` in the review folder; the variance check on E-4 and the independent-shock check on Experiment 3). They are therefore **post-hoc analyses**, not part of the frozen study. They are reported separately and labeled as such, and they do not change any FINAL_V1 hypothesis classification.

The sealed package `Final_Study_v1/` is used read-only; `MANIFEST_SHA256_FINAL.txt` is verified after the runs.

## Code and seeds

- **Code:** `code/r1_core.py` and `code/r1_run.py`.
  - Mechanical tests (`code/test_r1_core.py`) reproduce the sealed point estimates and CR1 standard errors of E-4, E-5, E-6, E-12a and E-12b to 1e-9.
  - They check the analytic calibration derivative against finite differences (relative error below 1e-4).
  - All tests used the throwaway `R1TEST` namespace, at most 6 replications per cell.
- **Panels and folds:** where an analysis re-examines a published configuration, it uses the FINAL_V1 production panel and fold seeds, so the panels are the published ones.
- **New random draws:** these come from the `R1` namespace, seeded by SHA-256 of `R1|tag|seed_cell_id|r`. They cover independent shocks, bootstrap resamples and folds, proxy strata and their selection, PTD bootstrap draws, and wild-bootstrap signs.

## Module A: variance with random normalization (review Comments 1 and 4)

**A1, the variance re-analysis.**
- **Cells and replications:** all 24 primary-grid cells and all 10 Experiment-2 cells, with the published replication counts (5,000 per cell).
- **Procedures:** E-4, E-5, E-6, E-6H, PS-I (the post-stratified identity rectifier, i.e. PS-TDS-I with the floor removed), E-12a, E-12b, E-12c, and E-2.
- **Standard errors computed for each:**
  - *sealed* — the published CR1;
  - *ratio* — the cluster score corrected for the random normalization by ratio linearization (uniform, Hájek, post-stratified);
  - *linearized* — the ratio correction plus the first-order term of the fitted cross-fitted calibration.
- **Reported:** bias, empirical SD, mean SE for each version, coverage, Type I error, and the ratio of mean SE to empirical SD.

**A2, a bootstrap check of the linearized variances.**
- **Cells:** the baseline non-null cell and the Experiment-2 cell with unequal validation and τ = 0.30, 400 published-seed replications each.
- **Bootstrap:** a pairs cluster bootstrap, B = 199. Whole units are resampled with their labels, and the labeled count, weights, folds (copies kept in one fold) and calibration are all recomputed.
- **Reported:** for each procedure, the empirical SD of the estimates, the mean sealed SE, the mean linearized SE and the mean bootstrap SD.

## Module B: separating leakage from added measurement error (review Comment 2)

- **Cells:** every contaminated cell of the contamination ladder (κ > 0) and of Experiment 3, with the published replication counts.
- **Arms**, computed on the same panel in each replication:
  - **none:** the latent index without contamination;
  - **future:** the published contaminated measurement;
  - **indep:** the uncontaminated index plus κ_t times an independent N(0, σ²) shock, with σ² the stationary disturbance variance and the same loading schedule and fixed thresholds.
- **Procedures:** E-1 and E-7 in all replications. E-12a and E-12b in the first 2,000 replications of the ladder cells.
- **Reported:** paired displacements future − none, indep − none and future − indep (the leakage-specific part), with MCSEs, and coverage of E-7 in each arm.
- **Experiment 3:** T·|d(T)| and the inverse-variance weighted regression of log|d(T)| on log T are recomputed for the leakage-specific displacement of E-1. The proposed 1/T law is judged against the slope's 95% interval, as in FINAL_V1.

## Module C: closest competing methods (review Comment 3)

- **Cells:** 12 representative cells, 2,000 published-seed replications each:
  - baseline (κ = 0 and κ = 0.60, ρ_e = 0.60);
  - Experiment 2 with τ = 0.30, under unequal and under uniform validation;
  - Experiment 4 at f = 0.10 with σ_e = 0.55, and at f = 0.35 with σ_e = 0.15;
  - each with its γ_W = 0 companion.
- **PTD:** predict-then-debias (Kluger et al. 2025), applied to the within regression.
  - θ̂• uses the labeled units with the human construct and weights R/π.
  - γ̂• uses the labeled units with the machine score and the same weights.
  - γ̂∘ uses the unlabeled units with the machine score and weights (1 − R)/(1 − π).
  - Diagonal tuning is estimated from a cluster bootstrap with B = 1,000, resampling whole units (their Algorithm 5).
  - The interval is the percentile interval of the bootstrap PTD draws.
  - Python implementation; not validated against the authors' R package, which could not be run in this environment.
- **E-10w:** design-weighted, power-tuned PPI++.
  - Identity predictor with Horvitz–Thompson weights.
  - λ is chosen to minimize the estimated variance of the target coordinate, clipped to [0, 1].
  - CR1 standard errors, with λ treated as fixed.
  - This is a design-aware alternative, not the protocol's unweighted E-10.
- **Also computed:** E-2, E-6, E-12a and E-12b.
- **Reported:** bias, coverage, RMSE relative to E-2, median width, Type I error, and failure rate.

## Module D: implementable strata and sparse validation (review Comment 5)

**D1, strata observed before annotation.**
- **Cells:** the baseline cell (τ = 0.10) and the Experiment-2 cell with τ = 0.30, each with γ_W ∈ {0.15, 0}. 2,000 published-seed panels each.
- **Stratifiers:** three, applied to each panel:
  - **oracle:** as published;
  - **machine:** tertiles of the unit mean of Z_L;
  - **proxy:** tertiles of a noisy proxy of the unit mean of Z_H, with reliability 0.5.
- **What changes:** with the machine and proxy stratifiers, units are re-selected independently with probability π of their new stratum, and all weights use the new strata. The outcome model and the target are unchanged.
- **Procedures:** E-2, E-4, E-5, E-6, E-12a, E-12b, E-12c, PTD and E-10w. Linearized SEs are computed for the uniform-weight procedures.

**D2, small-cluster inference.**
- **Cells:** the Experiment-4 cells with f ∈ {0.03, 0.05, 0.10}, σ_e ∈ {0.55, 0.80} and γ_W ∈ {0.15, 0}, with 2,000 published-seed replications each.
- **Intervals:**
  - for E-12a and E-12b: CR1 with a normal critical value (published), CR1 with a t(G_S − 1) critical value, and a wild cluster score bootstrap (Rademacher, B = 399, symmetric interval from the 95% quantile of |δ*|);
  - for E-2: its published CR1 t(G_S − 1) interval and the wild score bootstrap;
  - PTD with its percentile interval.
- **Reported for each method:**
  - the probability that an interval is produced;
  - coverage conditional on production;
  - the unconditional probability of producing a covering interval;
  - median width;
  - Type I error.

## Reporting rules

- Every quantity carries a Monte Carlo standard error.
- The FINAL_V1 coverage bands (near-nominal 0.940–0.960, mild 0.920–0.940, material below 0.920, conservative above 0.960) and displacement bands apply.
- No result is discarded, and failures are reported with their rates.
- Any analysis added after this plan is fixed will be identified as such.
