# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
# tumorfits/bayes_pde_pymc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .odeio import PatientData
from .metrics import nll_ratio_ca
from .pdemodel import PDEConfig
from .pdesolve import solve_pde


@dataclass
class PDEBayesConfig:
    draws: int = 1500
    tune: int = 1000
    chains: int = 2
    cores: int = 1
    sampler: str = "smc"  # "smc" or "metropolis"
    random_seed: int = 0

    # which PDE params to infer
    infer_diffusion: bool = False  # set True to include DS, DR in posterior


def build_pde_pymc_model(data: PatientData, cfg: PDEConfig, bcfg: PDEBayesConfig):
    """
    Build PyMC model for PDE parameters using black-box likelihood.
    Parameters inferred:
      default: [aS, aR, dS, dR, K]
      optional: + [DS, DR]
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.compile.ops import as_op

    # Select parameter dimension
    if bcfg.infer_diffusion:
        dim = 7  # aS, aR, dS, dR, K, DS, DR
    else:
        dim = 5  # aS, aR, dS, dR, K

    @as_op(itypes=[pt.dvector], otypes=[pt.dscalar])
    def nll_op(par_vec):
        p = np.asarray(par_vec, float)
        if p.size != dim:
            return np.array(1e50, dtype=float)

        # unpack and push into cfg (local copy for thread-safety)
        local_cfg = PDEConfig(**cfg.__dict__)
        if bcfg.infer_diffusion:
            aS, aR, dS, dR, K, DS, DR = map(float, p)
            local_cfg.DS = float(max(DS, 1e-12))
            local_cfg.DR = float(max(DR, 1e-12))
        else:
            aS, aR, dS, dR, K = map(float, p)

        # hard constraints
        if (aS <= 0) or (aR <= 0) or (dS < 0) or (dR < 0) or (K <= 1e-8):
            return np.array(1e50, dtype=float)

        nll, _, _, _ = solve_pde([aS, aR, dS, dR, K], local_cfg, data, comm=None, return_history=False)
        return np.array(float(nll), dtype=float)

    with pm.Model() as model:
        # weakly-informative priors in physical space (positive)
        # Tune these as needed (your ODE outputs are good prior centers).

        aS = pm.LogNormal("aS", mu=np.log(0.5), sigma=1.0)
        aR = pm.LogNormal("aR", mu=np.log(0.3), sigma=1.0)
        dS = pm.LogNormal("dS", mu=np.log(0.4), sigma=1.0)
        dR = pm.LogNormal("dR", mu=np.log(0.1), sigma=1.0)
        K  = pm.LogNormal("K",  mu=np.log(1e6), sigma=3.0)

        if bcfg.infer_diffusion:
            DS = pm.LogNormal("DS", mu=np.log(cfg.DS), sigma=1.0)
            DR = pm.LogNormal("DR", mu=np.log(cfg.DR), sigma=1.0)
            pars = pt.stack([aS, aR, dS, dR, K, DS, DR])
        else:
            pars = pt.stack([aS, aR, dS, dR, K])

        pm.Potential("likelihood", -nll_op(pars))

    return model


def sample_pde_posterior(data: PatientData, cfg: PDEConfig, bcfg: PDEBayesConfig):
    import pymc as pm

    model = build_pde_pymc_model(data, cfg, bcfg)
    with model:
        if bcfg.sampler.lower() == "smc":
            idata = pm.sample_smc(
                draws=bcfg.draws,
                random_seed=bcfg.random_seed,
                chains=bcfg.chains,
                cores=bcfg.cores,
                progressbar=True,
            )
        elif bcfg.sampler.lower() == "metropolis":
            step = pm.Metropolis()
            idata = pm.sample(
                draws=bcfg.draws,
                tune=bcfg.tune,
                step=step,
                chains=bcfg.chains,
                cores=bcfg.cores,
                random_seed=bcfg.random_seed,
                progressbar=True,
                discard_tuned_samples=True,
            )
        else:
            raise ValueError("bcfg.sampler must be 'smc' or 'metropolis'")
    return idata
