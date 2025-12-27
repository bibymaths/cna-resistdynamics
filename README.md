# CNA-ResistDynamics

**CNA-ResistDynamics** is a dynamical modeling framework for inferring the temporal
evolution of chemotherapy resistance from longitudinal copy-number–based
measurements.

The project models **resistant vs sensitive tumor cell populations** using:

* a **lumped ODE model** (well-mixed population dynamics)
* a **spatially resolved PDE model** (reaction–diffusion with treatment effects)

Resistance dynamics are inferred from **liquidCNA-derived resistant subclone
fractions** and **tumor burden measurements (CA125)**.

This repository focuses on **evolutionary dynamics and identifiability**, not on
raw sequencing processing or mechanistic PK/PD modeling.

---

## Scope and non-goals

**Included**

* Continuous-time ODE and PDE models of resistance dynamics
* Likelihood-based fitting to longitudinal clinical data
* Uncertainty-aware observation models
* Cohort- and patient-level analysis via a unified CLI

**Not included**

* Raw sequencing data processing
* Variant calling or CNA segmentation
* Drug PK/PD or dose–response modeling
* Clinical decision support

---

## Installation

```bash
pip install -e .
tumorfits -h
```

---

## Command-line interface

All functionality is exposed via a single CLI:

```bash
tumorfits ode ...
tumorfits pde ...
```

Run `-h` on any subcommand for full options.

---

## ODE workflow

### Single patient

```bash
tumorfits ode \
  --data Subclonal_ratio_estimates.extended.txt \
  --patient UP0018 \
  --time_unit months \
  --n_starts 8 \
  --maxiter 1200 \
  --out_points ode_gof_points_UP0018.csv \
  --diag_dir results_ode_model/per_patient_plots
```

### Cohort by `Accept_estimate` flag

```bash
tumorfits ode \
  --data Subclonal_ratio_estimates.extended.txt \
  --flag yes,maybe \
  --out_points ode_gof_points_flags_yes_maybe.csv \
  --diag_dir results_ode_model/per_patient_plots_flags_yes_maybe
```

### Cohort with QC filters and updated CA125

```bash
tumorfits ode \
  --data Subclonal_ratio_estimates.extended.txt \
  --flag yes,maybe \
  --sample_list OV_patientDNA_sampleList.txt \
  --use_ca125_updated \
  --drop_failed \
  --require_panel_sequenced \
  --require_detected_cna \
  --out_points ode_gof_points_qc.csv
```

The resulting CSV (`ode_gof_points*.csv`) is the **input to the PDE stage**.

---

## PDE workflow (uses ODE output)

### Single patient (simulation only)

```bash
tumorfits pde \
  --data Subclonal_ratio_estimates.extended.txt \
  --ode_points ode_gof_points_flags_yes_maybe.csv \
  --patient UP0018 \
  --out_dir results_pde_model_UP0018
```

### Single patient with PDE fitting

```bash
tumorfits pde \
  --data Subclonal_ratio_estimates.extended.txt \
  --ode_points ode_gof_points_flags_yes_maybe.csv \
  --patient UP0018 \
  --fit \
  --n_starts 10 \
  --maxiter 150 \
  --out_dir results_pde_model_UP0018
```

### All patients found in ODE CSV

```bash
tumorfits pde \
  --data Subclonal_ratio_estimates.extended.txt \
  --ode_points ode_gof_points_flags_yes_maybe.csv \
  --patient ALL \
  --fit \
  --out_dir results_pde_model_all
```

---

## Data used

This repository ingests **derived, non-identifiable data only**.

### Required input

From the public liquidCNA Mendeley dataset:

* `Subclonal_ratio_estimates.extended.txt`

  * resistant subclone fraction (`ratio`)
  * uncertainty estimates (95% CI)
  * sampling time
  * clinical context
  * CA125 measurements
  * quality flags

### Optional input

* patient-specific copy-number segment tables (used only in extended analyses)

No raw sequencing reads or controlled-access datasets are required.

---

## Modeling approach

### State variables

We model **sensitive** and **resistant** tumor cell populations:

$$
S(t), \quad R(t)
$$

with total tumor burden:

$$
N(t) = S(t) + R(t)
$$

and resistant fraction:

$$
r(t) = \frac{R(t)}{S(t) + R(t)}.
$$

---

## ODE model (well-mixed)

The lumped population dynamics are:

$$
\begin{aligned}
\frac{dS}{dt} &= a_S,S\left(1 - \frac{S+R}{K}\right) - u(t),d_S,S, \
\frac{dR}{dt} &= a_R,R\left(1 - \frac{S+R}{K}\right) - u(t),d_R,R,
\end{aligned}
$$

where:

* (a_S, a_R) are growth rates
* (d_S, d_R) are treatment-induced death rates
* (K) is the carrying capacity
* (u(t) \in [0,1]) is a treatment intensity determined by clinical context

---

## PDE model (reaction–diffusion)

To model spatial heterogeneity, the PDE extends the ODE with diffusion:

$$
\begin{aligned}
\frac{\partial S(x,t)}{\partial t}
&=
D_S \nabla^2 S

* a_S S\left(1 - \frac{S+R}{K}\right)

- u(t),d_S,S, [6pt]
  \frac{\partial R(x,t)}{\partial t}
  &=
  D_R \nabla^2 R

* a_R R\left(1 - \frac{S+R}{K}\right)

- u(t),d_R,R.
  \end{aligned}
  $$

where:

* (D_S, D_R) are diffusion coefficients
* spatial domain is one-dimensional: (x \in [0, L])
* zero-flux (Neumann) boundary conditions are used

The PDE is solved via **operator splitting**:

* explicit Euler for reactions
* implicit backward Euler for diffusion

---

## Observation models

### Resistant fraction

Observed resistant fractions are modeled on the **logit scale** with
Gaussian noise derived from reported confidence intervals.

### Tumor burden (CA125)

CA125 is linked to total tumor burden via:

$$
\log(\text{CA125}(t)) = \log\big(c_0 + \gamma \int_0^L (S+R),dx\big) + \epsilon,
\quad
\epsilon \sim \mathcal{N}(0, \sigma_{CA}^2).
$$

---

## Inference

* ODE and PDE parameters are fit via **multi-start likelihood optimization**
* No MCMC is required for the main pipeline
* Outputs include:

  * fitted trajectories
  * goodness-of-fit metrics
  * patient- and cohort-level summaries

---

## Repository structure

```text
CNA-ResistDynamics/
├── tumorfits/        # Python package (ODE, PDE, CLI)
├── data/
├── results_ode_model/
├── results_pde_model/
├── README.md
└── LICENSE
```

---

## Intended use

This repository is suitable for:

* evolutionary modeling of therapy resistance
* mechanistic interpretation of liquidCNA dynamics
* methodological development for resistance monitoring

It is **not intended for clinical deployment**.

---

## Data availability

Data originate from:

**Hockings et al. (2025)**
*Adaptive Therapy Exploits Fitness Deficits in Chemotherapy-Resistant Ovarian Cancer to Achieve Long-Term Tumor Control*
*Cancer Research*, 85(18), 3503–3517
DOI: [https://doi.org/10.1158/0008-5472.CAN-25-0351](https://doi.org/10.1158/0008-5472.CAN-25-0351)

Mendeley Data (derived outputs):
[https://doi.org/10.17632/m93sk9n767.1](https://doi.org/10.17632/m93sk9n767.1)

---

## License

Released under an open-source license.
See `LICENSE` for details.

---