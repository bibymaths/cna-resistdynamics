<!-- SPDX-FileCopyrightText: 2025 Abhinav Mishra -->
<!-- SPDX-License-Identifier: MIT -->

# Mathematical Model

## Biological motivation

Ovarian cancer treated with platinum-based chemotherapy typically responds
initially but relapses as drug-resistant subclones expand.  The
[liquidCNA](https://github.com/McGranahanLab/liquidCNA) algorithm estimates the
*subclonal CNA ratio* — the fraction of circulating tumour DNA carrying
subclonal copy-number alterations — from longitudinal liquid-biopsy samples.
This ratio is a proxy for the fraction of resistant cells, \( R(t) / N(t) \),
where \( N = S + R \).

The serum protein CA125 provides a complementary tumour-burden readout, related
to total cell density via a log-linear observation model.

---

## ODE model

The well-mixed model tracks sensitive (\( S \)) and resistant (\( R \))
populations subject to logistic growth and cytotoxic treatment:

\[
\frac{dS}{dt} = a_S \, S \!\left(1 - \frac{N}{K}\right) - d_S \, u(t) \, S
\]

\[
\frac{dR}{dt} = a_R \, R \!\left(1 - \frac{N}{K}\right) - d_R \, u(t) \, R
\]

where \( N = S + R \) and:

| Symbol | Meaning |
|--------|---------|
| \( a_S, a_R \) | Intrinsic growth rates of sensitive and resistant cells |
| \( d_S, d_R \) | Drug-induced death rates |
| \( K \) | Carrying capacity |
| \( u(t) \) | Treatment intensity (context-specific step function) |

### Observation model

The subclonal CNA ratio and CA125 are linked to model states via:

\[
\hat{\rho}(t) = \frac{R(t)}{S(t) + R(t)}
\]

\[
\log \text{CA125}(t) \approx \gamma \bigl(S(t) + R(t)\bigr) + c_0
\]

### Parameter space

Parameters are estimated in unconstrained space:

| Transformed | Physical | Constraint enforced |
|-------------|---------|---------------------|
| \( \log a_S \) | \( a_S > 0 \) | |
| \( \text{logit}(a_R / a_S) \) | \( 0 < a_R < a_S \) | resistant grow slower |
| \( \log d_S \) | \( d_S > 0 \) | |
| \( \text{logit}(d_R / d_S) \) | \( 0 < d_R < d_S \) | resistant die less |
| \( \log K \) | \( K > 0 \) | |

### Inference

Multi-start L-BFGS-B minimisation of the weighted negative log-likelihood:

\[
\mathcal{L} = \mathcal{L}_{\text{ratio}} + w_{\text{CA}} \cdot \mathcal{L}_{\text{CA125}}
\]

where \( \mathcal{L}_{\text{ratio}} \) is a ratio-scale NLL with
observation errors propagated to logit scale.

---

## PDE model

The 1-D reaction–diffusion model adds spatial structure along a tumour axis
\( x \in [0, L] \):

\[
\frac{\partial S}{\partial t} = D_S \frac{\partial^2 S}{\partial x^2}
  + a_S \, S \!\left(1 - \frac{N}{K}\right) - d_S \, u(t) \, S
\]

\[
\frac{\partial R}{\partial t} = D_R \frac{\partial^2 R}{\partial x^2}
  + a_R \, R \!\left(1 - \frac{N}{K}\right) - d_R \, u(t) \, R
\]

with Neumann (zero-flux) boundary conditions:

\[
\frac{\partial S}{\partial x}\bigg|_{x=0,L} = 0,
\quad
\frac{\partial R}{\partial x}\bigg|_{x=0,L} = 0
\]

### Spatial observation model

Observables are obtained by integrating across the spatial domain:

\[
\hat{\rho}(t) = \frac{\int_0^L R \, dx}{\int_0^L (S + R) \, dx}
\]

\[
\widehat{\log \text{CA125}}(t) = \gamma \int_0^L (S + R) \, dx + c_0
\]

### Numerical discretisation

The PDE is solved with operator splitting (Lie–Trotter):

1. **Reaction step** — explicit Euler update for growth and death terms
2. **Diffusion step** — implicit FEniCS/dolfinx FEM solve with PETSc backend

The linear systems for the diffusion step are cached (keyed on mesh geometry
and time-step) to avoid repeated assembly across optimiser iterations.

### Initial conditions

At \( t = 0 \) (first observation):

\[
S(x, 0) = S_0, \quad R(x, 0) = R_0
\]

where \( S_0, R_0 \) are uniform initial densities inferred from the first
observed ratio and a patient-specific total burden estimate.

---

## 2-D mesh model

The `mesh-view` pipeline runs a 2-D variant on a unit-square FEniCS mesh
for visualisation purposes.  This does not produce fitted parameters but shows
spatial resistance-zone formation and drug-efficacy gradients.

See `tumorfits.meshview` for implementation details.

---

## Identifiability and sensitivity

Sobol first-order and total-order sensitivity indices are computed for both
ODE and PDE parameter spaces using SALib.  Run via `tumorfits.identifiability`.
