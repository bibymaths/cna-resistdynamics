# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .odeio import PatientData
from .odemodel import simulate_ode
from .metrics import nll_ratio_ca


@dataclass
class ODEBayesConfig:
    """
    Bayesian config for ODE posterior.
    We use a black-box likelihood (as_op), so prefer SMC or Metropolis.
    """
    draws: int = 2000
    tune: int = 1000
    chains: int = 2
    cores: int = 1
    sampler: str = "smc"  # "smc" or "metropolis"
    w_ca: float = 0.5
    random_seed: int = 0


def _ode_theta_to_physical(theta: np.ndarray, n_ctx: int) -> Dict[str, Any]:
    """
    Helper: interpret theta into physical values (mostly for reporting).
    Theta layout must match your canonical ODE layout:
      [log_aS, logit_aR_over_aS, log_dS, logit_dR_over_dS, log_K, log_N0, logit_r0,
       log_gamma, log_ca0, log_sigma_ca, logit_u_ctx[...]]
    """
    from .utils import invlogit

    theta = np.asarray(theta, float)
    assert theta.size == 10 + n_ctx

    log_aS, logit_aR, log_dS, logit_dR, log_K, log_N0, logit_r0, log_gamma, log_ca0, log_sigma_ca = theta[:10]
    u_logits = theta[10:]

    aS = float(np.exp(log_aS))
    aR = float(aS * invlogit(np.array([logit_aR]))[0])
    dS = float(np.exp(log_dS))
    dR = float(dS * invlogit(np.array([logit_dR]))[0])
    K = float(np.exp(log_K))
    N0 = float(np.exp(log_N0))
    r0 = float(invlogit(np.array([logit_r0]))[0])
    gamma = float(np.exp(log_gamma))
    ca0 = float(np.exp(log_ca0))
    sigma_ca = float(np.exp(log_sigma_ca))
    u_ctx = invlogit(u_logits)

    return {
        "aS": aS,
        "aR": aR,
        "dS": dS,
        "dR": dR,
        "K": K,
        "N0": N0,
        "r0": r0,
        "gamma": gamma,
        "ca0": ca0,
        "sigma_ca": sigma_ca,
        "u_ctx": u_ctx,
    }


def build_ode_pymc_model(data: PatientData, cfg: ODEBayesConfig):
    """
    Build a PyMC model for the ODE, using black-box likelihood.

    Returns: pymc.Model
    """
    # Local imports so ode-only users don't need pymc installed.
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.compile.ops import as_op

    C = len(data.context_names)

    # --- black-box NLL as a PyTensor Op ---
    # NOTE: This is not differentiable => use SMC or Metropolis.
    @as_op(itypes=[pt.dvector], otypes=[pt.dscalar])
    def nll_op(theta_vec):
        theta = np.asarray(theta_vec, float)

        # Hard bounds / constraints to keep sampler sane
        if theta.size != 10 + C:
            return np.array(1e50, dtype=float)

        try:
            r_hat, logca_hat = simulate_ode(data, theta)
        except Exception:
            return np.array(1e50, dtype=float)

        sigma_ca = float(np.exp(theta[9]))  # canonical slot
        nll = nll_ratio_ca(
            ratio_obs=np.asarray(data.ratio, float),
            se_logit_ratio=np.asarray(data.se_logit_ratio, float),
            logca_obs=np.asarray(data.log_ca125, float),
            ratio_hat=np.asarray(r_hat, float),
            logca_hat=np.asarray(logca_hat, float),
            sigma_ca=sigma_ca,
            w_ca=float(cfg.w_ca),
        )
        return np.array(float(nll), dtype=float)

    with pm.Model() as model:
        # Priors on the transformed parameters (match your optimizer space).
        # Keep these moderately informative to help gradient-free samplers.

        log_aS = pm.Normal("log_aS", mu=np.log(0.5), sigma=1.0)
        logit_aR_over_aS = pm.Normal("logit_aR_over_aS", mu=0.0, sigma=2.0)

        log_dS = pm.Normal("log_dS", mu=np.log(0.8), sigma=1.0)
        logit_dR_over_dS = pm.Normal("logit_dR_over_dS", mu=-2.0, sigma=2.5)

        log_K = pm.Normal("log_K", mu=np.log(1e6), sigma=3.0)
        log_N0 = pm.Normal("log_N0", mu=np.log(1e4), sigma=3.0)

        logit_r0 = pm.Normal("logit_r0", mu=0.0, sigma=3.0)

        log_gamma = pm.Normal("log_gamma", mu=np.log(1e-3), sigma=3.0)
        log_ca0 = pm.Normal("log_ca0", mu=float(np.mean(data.log_ca125) - 1.0), sigma=2.0)

        log_sigma_ca = pm.Normal("log_sigma_ca", mu=np.log(0.5), sigma=1.0)

        # context schedule intensities
        logit_u_ctx = pm.Normal("logit_u_ctx", mu=0.0, sigma=2.5, shape=C)

        theta = pm.Deterministic(
            "theta",
            pt.concatenate(
                [
                    pt.stack(
                        [
                            log_aS,
                            logit_aR_over_aS,
                            log_dS,
                            logit_dR_over_dS,
                            log_K,
                            log_N0,
                            logit_r0,
                            log_gamma,
                            log_ca0,
                            log_sigma_ca,
                        ]
                    ),
                    logit_u_ctx,
                ]
            ),
        )

        pm.Potential("likelihood", -nll_op(theta))

    return model


def sample_ode_posterior(data: PatientData, cfg: ODEBayesConfig):
    """
    Run posterior sampling for ODE model.
    Returns: arviz.InferenceData
    """
    import pymc as pm

    model = build_ode_pymc_model(data, cfg)

    with model:
        if cfg.sampler.lower() == "smc":
            idata = pm.sample_smc(
                draws=cfg.draws,
                random_seed=cfg.random_seed,
                chains=cfg.chains,
                cores=cfg.cores,
                progressbar=True,
            )
        elif cfg.sampler.lower() == "metropolis":
            step = pm.Metropolis()
            idata = pm.sample(
                draws=cfg.draws,
                tune=cfg.tune,
                step=step,
                chains=cfg.chains,
                cores=cfg.cores,
                random_seed=cfg.random_seed,
                progressbar=True,
                discard_tuned_samples=True,
            )
        else:
            raise ValueError("cfg.sampler must be 'smc' or 'metropolis'")

    return idata


def summarize_ode_posterior(idata, data: PatientData) -> Dict[str, Any]:
    """
    Convenience summary: posterior mean in physical space for reporting.
    """
    import arviz as az

    post = idata.posterior["theta"].values  # (chain, draw, dim)
    theta_mean = np.mean(post.reshape(-1, post.shape[-1]), axis=0)
    phys = _ode_theta_to_physical(theta_mean, len(data.context_names))

    summ = az.summary(idata, var_names=["theta"], kind="stats")
    return {"theta_mean": theta_mean, "physical_mean": phys, "az_summary": summ}
