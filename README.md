# Valid Inference with AI-Generated Covariates in Clustered Business Panels

### Informative Validation, Temporal Contamination and Stabilization

**Authors:** Yousef Nejatbakhsh, Malihe Aliasgari, Sina Shokoohyar, Reza Kia and Mahshid Fahanzadesaadi

**Corresponding author:** Malihe Aliasgari (maliasga@kean.edu)

**Target Journal:** *Applied Stochastic Models in Business and Industry (ASMBI)*

## Overview

This repository provides the reproducibility archive for our research on statistical inference with AI-generated covariates in clustered business panel data.

The study examines how informative validation sampling, measurement error, and temporal contamination affect regression inference when AI-generated measurements are corrected using human-labeled observations.

The research combines mathematical analysis with a systematic Monte Carlo simulation study.

## Research Contributions

The study investigates four principal questions:

1. **Informative Validation:** How unequal probabilities of selecting firms for human validation can introduce systematic bias into prediction-assisted inference.

2. **Design-Aware Rectification:** How inverse-probability-weighted corrections can restore population centering and improve estimation efficiency in the studied settings.

3. **Temporal Contamination:** How AI-generated measurements containing future information can compromise statistical inference, even when predictive performance improves.

4. **Numerical Stabilization:** How target-direction stabilization can reduce numerical instability in rectified estimating equations without necessarily improving interval coverage.

## Repository Contents

| Path | Contents |
|---|---|
| [`ASMBI_Public_Archive.zip`](ASMBI_Public_Archive.zip) | The FINAL_V1 reproducibility archive. It covers the frozen study: Sections 4–5 of the article and Supplementary Sections S1–S7. |
| [`post_frozen_analyses/`](post_frozen_analyses/) | The analyses run after the frozen study: Section 5.5 of the article and Supplementary Section S8, Tables S13–S19. |
| `CITATION.cff`, `.zenodo.json` | Citation metadata. |
| `LICENSE` | MIT License for the code. |

## Simulation Study

The final study, designated **FINAL_V1**, includes:

| Component | Description |
|---|---|
| Simulation configurations | 149 |
| Monte Carlo replications | 567,000 |
| Evaluated estimator implementations | 16 |
| Statistical framework | Clustered panel regression |
| Validation design | Informative cluster-level sampling |
| Main evaluation criteria | Bias, RMSE, coverage, Type I error, power, and numerical stability |

The study uses simulated data only. It does not claim validation using an actual language-model-generated dataset.

## Reproducibility Archive

**[Access the FINAL_V1 Reproducibility Archive](ASMBI_Public_Archive.zip)**

The downloadable ZIP contains:

- Frozen simulation protocol and configuration registry.
- Simulation code and random seed specifications.
- Analysis scripts and cell-level simulation results.
- Monte Carlo performance summaries.
- Numerical stability and failure diagnostics.
- Figure-generation scripts and figure data.
- SHA-256 integrity manifests.
- Reproducibility documentation.

The archive includes a detailed `README.md` with software requirements and instructions for reproducing the analyses.

**Important:** The original per-replication raw checkpoints are not included because of their size. They can be regenerated using the supplied simulation code and seed specifications.

## Analyses After the Frozen Study

The folder [`post_frozen_analyses/`](post_frozen_analyses/) holds the analyses reported in Section 5.5 of the article and in Supplementary Section S8.

**Status.** These analyses were specified after the FINAL_V1 results were known. They were fixed in a written plan before any of them was run: the plan was sealed by SHA-256 on 13 November 2026. They are reported as post-hoc analyses and change no FINAL_V1 hypothesis classification.

**What they cover:**

| Topic | Supplementary tables |
|---|---|
| Linearized and bootstrap standard errors for rectifiers with a random normalization | S13 and S14 |
| An independent-shock control that separates leakage from added measurement error | S15 and S16 |
| Predict-then-debias and design-weighted, tuned PPI++ comparators | S17 |
| Validation strata observed before annotation | S18 |
| Interval availability and small-cluster intervals under sparse validation | S19 |

**The folder contains** the plan and its seal, the code, every summary table, the run log and a SHA-256 manifest. Its `README.md` gives the commands that reproduce the results. The per-replication outputs (about 72 MB) are not included; they are regenerated exactly from the code and seeds.

## Software

The simulation environment used:

- Python 3.13.16
- NumPy 2.5.3
- SciPy 1.18.1
- Matplotlib 3.11.2 for manuscript figures

Please consult the documentation inside the archive for execution instructions and environment-specific requirements.

## Research Integrity

The FINAL_V1 experimental protocol was frozen on November 3, 2026, before the final simulation run.

The research distinguishes prospective tests from confirmatory re-estimates, reports unsuccessful hypothesis tests, and acknowledges limitations involving sparse validation and unevaluated statistical procedures. The analyses after the frozen study were fixed in a sealed plan before they were run and are reported separately from the frozen study.

## Citation

If you use this research archive, please cite:

Nejatbakhsh, Y., Aliasgari, M., Shokoohyar, S., Kia, R., & Fahanzadesaadi, M. (2026). *Valid Inference with AI-Generated Covariates in Clustered Business Panels: Informative Validation, Temporal Contamination and Stabilization*. Manuscript prepared for Applied Stochastic Models in Business and Industry.

Citation metadata is provided in `CITATION.cff`.

## License

The simulation and analysis code is distributed under the MIT License (`LICENSE`). The accompanying protocol, outputs, and documentation are released under Creative Commons Attribution 4.0 (CC BY 4.0).

## Contact

**Malihe Aliasgari** (corresponding author)  
School of Integrative Science and Technology  
Kean University  
Email: maliasga@kean.edu

**Yousef Nejatbakhsh**  
Department of Statistics  
Rutgers University–New Brunswick  
Email: yousef.nejatbakhsh@rutgers.edu
