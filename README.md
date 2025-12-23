# CNA-ResistDynamics

**CNA-ResistDynamics** is a Bayesian dynamical modeling framework for inferring
the temporal evolution of chemotherapy resistance from longitudinal
copy-number–based measurements.

The project focuses on *population-level resistance dynamics* inferred from
liquidCNA-derived resistant subclone fractions and tumor burden measurements,
using continuous-time state-space models with explicit uncertainty propagation.

This repository does **not** reprocess raw sequencing data and does **not**
attempt mechanistic drug PK/PD modeling. Its scope is evolutionary dynamics,
identifiability, and quantitative interpretation of sparse longitudinal data.

---

## Scientific motivation

Resistance to chemotherapy emerges through clonal and subclonal selection.
While copy-number alterations (CNAs) provide strong signals of resistance
evolution, clinical measurements are:

- sparse in time
- noisy
- patient-specific
- confounded by tumor purity and sampling context

Most studies analyze such data descriptively.  
This project formalizes resistance evolution as a **latent dynamical process**
and infers it using Bayesian state-space models.

The framework is designed to answer questions such as:

- How fast does resistance expand within a patient?
- Does resistance grow monotonically or episodically?
- How strongly does inferred resistance track tumor burden (CA125)?
- Are resistance dynamics consistent across treatment contexts?

---

## Data used

This repository ingests **derived, non-identifiable data only**.

### Required input
From the publicly available liquidCNA Mendeley dataset:

- `Subclonal_ratio_estimates.extended.txt`
  - resistant subclone fraction (`ratio`)
  - 95% confidence intervals (`ratio_min95`, `ratio_max95`)
  - sampling time (`Time`)
  - clinical context (`context`)
  - tumor burden proxy (`CA125`)
  - quality flag (`Accept_estimate`)

### Optional input
From patient-specific copy-number tables:

- `Copynumber_tables_<PATIENT>.combined.500.RData`
  - purity-corrected segment-level copy number values
  - ΔCN relative to baseline
  - clonal vs subclonal segment annotations

Raw sequencing reads and controlled-access datasets (e.g. EGA) are **not**
required.

---

## Modeling approach

### Latent state

For each patient \( p \), resistance is modeled as a latent fraction

$$
r_p(t) \in (0, 1)
$$

internally represented on the logit scale

$$
z_p(t) = \log\frac{r_p(t)}{1 - r_p(t)}.
$$

---

### Evolution model (continuous time)

Resistance evolves as a stochastic process with context-dependent drift:

$$
z_{p,i} = z_{p,i-1}
+ \mu_{p,c} \Delta t_i
+ \epsilon_{p,i},
\quad
\epsilon_{p,i} \sim \mathcal{N}(0, \sigma_p^2 \Delta t_i)
$$

where:

- \( \Delta t_i \) is the time between samples
- \( c \) is the treatment / clinical context
- \( \mu_{p,c} \) is a selection-like drift term
- \( \sigma_p \) captures evolutionary noise

Drift parameters are modeled hierarchically across patients.

---

### Observation models

#### Resistance fraction
Observed resistant fractions (`ratio`) are treated as noisy measurements with
uncertainty derived directly from the reported 95% confidence intervals.

Measurements are modeled on the logit scale using a Gaussian approximation.

#### Tumor burden (CA125)
CA125 is modeled as a log-normal observation coupled to the latent resistance
state:

$$
\log(\text{CA125}_{p,i})
\sim
\mathcal{N}(\alpha_p + \beta z_{p,i}, \sigma_c^2)
$$

This allows quantitative testing of whether inferred resistance dynamics
predict disease burden.

---

### Optional segment-level model

When segment-level CNA data are available, individual subclonal segments can be
modeled as reporters of the resistant fraction:

$$
\Delta \text{CN}_{s}(t) \propto r_p(t)
$$

This removes reliance on liquidCNA summary ratios and enables reconstruction of
resistance dynamics directly from CNA evidence.

---

## Inference

Models are implemented in:

- **PyMC** (recommended for exploration and diagnostics)
- **Stan / cmdstanpy** (recommended for performance and publication)

Inference is fully Bayesian and yields:

- posterior trajectories of resistance per patient
- uncertainty-aware growth / selection rates
- coupling strength between resistance and CA125
- predictive distributions for future observations

---

## Repository structure

```text
CNA-ResistDynamics/
├── data/
│   ├── Subclonal_ratio_estimates.extended.txt
│   └── Copynumber_tables_<PATIENT>.RData
├── preprocessing/
│   ├── prepare_stan_data.py
│   └── qc_filters.py
├── models/
│   ├── resistance_state_space.stan
│   └── resistance_state_space_pymc.py
├── analysis/
│   ├── posterior_checks.ipynb
│   └── trajectory_plots.ipynb
├── figures/
├── README.md
└── LICENSE
````

---

## Intended use

This repository is suitable for:

* evolutionary modeling of therapy resistance
* state-space modeling with sparse clinical data
* method development for resistance monitoring
* hypothesis testing prior to adaptive therapy trials

It is **not** intended for clinical deployment.

---

## Data availability

This project uses **derived, non-identifiable copy-number and resistance estimates** from the following public dataset:

**Mendeley Data**
Hockings, H.; Lakatos, E.; Huang, W.; et al.
*Copy number profiles and liquidCNA algorithm output for patient samples presented in “Adaptive Therapy Exploits Fitness Deficits in Chemotherapy-Resistant Ovarian Cancer to Achieve Long-Term Tumor Control”*
Version 1, 2025
DOI: [https://doi.org/10.17632/m93sk9n767.1](https://doi.org/10.17632/m93sk9n767.1)

The dataset includes:

* Longitudinal liquidCNA-derived resistant subclone fractions
* Associated uncertainty estimates
* CA125 measurements
* Purity-corrected copy-number segment tables (QDNAseq / liquidCNA)

No raw sequencing reads or controlled-access data are required.

---

## Related publication

The data originate from the following peer-reviewed study:

Hockings, H., Lakatos, E., Huang, W., Mössner, M., Khan, M. A., Bakali, N., McDermott, J., Smith, K., Baker, A.-M., Graham, T. A., & Lockley, M.
**Adaptive Therapy Exploits Fitness Deficits in Chemotherapy-Resistant Ovarian Cancer to Achieve Long-Term Tumor Control.**
*Cancer Research*, 85(18), 3503–3517 (2025).
DOI: [https://doi.org/10.1158/0008-5472.CAN-25-0351](https://doi.org/10.1158/0008-5472.CAN-25-0351)

---

## Citation guidance

If you use this repository, please cite **both**:

1. The original *Cancer Research* article (for biological context and experimental design)
2. The Mendeley Data dataset (for derived copy-number and resistance estimates)

Example BibTeX entries can be added upon request.

---

## Notes on data use

* All analyses in this repository operate on **derived outputs only**
* No patient-identifiable information is included
* This project complies with the data-use terms specified by the dataset authors

---

## License

This project is released under an open-source license.
See `LICENSE` for details.
