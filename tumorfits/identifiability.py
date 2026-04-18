# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import nll_ratio_ca
from .odeio import PatientData
from .odemodel import simulate_ode
from .pdemodel import PDEConfig
from .pdesolve import solve_pde


@dataclass
class SensitivityConfig:
    """
    SALib sensitivity config.
    """

    method: str = "sobol"  # currently only sobol
    n_base: int = 1024  # base sample size (Saltelli). 1024-8192 typical.
    calc_second_order: bool = False
    seed: int = 0


def _sobol_problem(names: list[str], bounds: list[tuple[float, float]]) -> dict:
    return {"num_vars": len(names), "names": names, "bounds": bounds}


def _ode_objective_from_theta(theta: np.ndarray, data: PatientData, w_ca: float = 0.5) -> float:
    """
    Objective for ODE sensitivity.
    theta must be in your canonical transformed parameterization.
    """
    try:
        r_hat, logca_hat = simulate_ode(data, theta)
    except Exception:
        return 1e50
    sigma_ca = float(np.exp(theta[9]))  # canonical slot
    return float(
        nll_ratio_ca(
            ratio_obs=data.ratio,
            se_logit_ratio=data.se_logit_ratio,
            logca_obs=data.log_ca125,
            ratio_hat=r_hat,
            logca_hat=logca_hat,
            sigma_ca=sigma_ca,
            w_ca=w_ca,
        )
    )


def _pde_objective_from_params(params: np.ndarray, cfg: PDEConfig, data: PatientData) -> float:
    """
    Objective for PDE sensitivity.
    params in physical space: [aS, aR, dS, dR, K] (and optionally DS/DR if you decide)
    """
    nll, _, _, _ = solve_pde(params, cfg, data, comm=None, return_history=False)
    return float(nll)


def run_sobol_sensitivity_ode(
    *,
    data: PatientData,
    names: list[str],
    bounds: list[tuple[float, float]],
    cfg: SensitivityConfig = SensitivityConfig(),
    w_ca: float = 0.5,
    out_prefix: str = "salib_ode",
) -> dict[str, pd.DataFrame]:
    """
    Sobol sensitivity for ODE.
    'names' and 'bounds' must correspond to the ODE theta vector you want to vary.
    """
    # Local imports so users without SALib don't break ODE/PDE usage.
    from SALib.analyze import sobol
    from SALib.sample import saltelli

    rng = np.random.default_rng(cfg.seed)
    problem = _sobol_problem(names, bounds)

    # Saltelli sample (Sobol)
    X = saltelli.sample(
        problem,
        N=cfg.n_base,
        calc_second_order=cfg.calc_second_order,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    Y = np.empty((X.shape[0],), dtype=float)
    for i in range(X.shape[0]):
        Y[i] = _ode_objective_from_theta(X[i, :], data, w_ca=w_ca)

    # Analyze
    Si = sobol.analyze(problem, Y, calc_second_order=cfg.calc_second_order, print_to_console=False)

    # Save
    df_samples = pd.DataFrame(X, columns=names)
    df_samples["objective"] = Y
    df_samples.to_csv(f"{out_prefix}_samples.csv", index=False)

    df_S1 = pd.DataFrame({"name": names, "S1": Si["S1"], "S1_conf": Si["S1_conf"]})
    df_ST = pd.DataFrame({"name": names, "ST": Si["ST"], "ST_conf": Si["ST_conf"]})
    df_S1.to_csv(f"{out_prefix}_S1.csv", index=False)
    df_ST.to_csv(f"{out_prefix}_ST.csv", index=False)

    return {"samples": df_samples, "S1": df_S1, "ST": df_ST}


def run_sobol_sensitivity_pde(
    *,
    data: PatientData,
    pde_cfg: PDEConfig,
    names: list[str],
    bounds: list[tuple[float, float]],
    cfg: SensitivityConfig = SensitivityConfig(),
    out_prefix: str = "salib_pde",
) -> dict[str, pd.DataFrame]:
    """
    Sobol sensitivity for PDE parameters in physical space.
    Typical names: ["aS","aR","dS","dR","K"].
    """
    from SALib.analyze import sobol
    from SALib.sample import saltelli

    rng = np.random.default_rng(cfg.seed)
    problem = _sobol_problem(names, bounds)

    X = saltelli.sample(
        problem,
        N=cfg.n_base,
        calc_second_order=cfg.calc_second_order,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    Y = np.empty((X.shape[0],), dtype=float)
    for i in range(X.shape[0]):
        Y[i] = _pde_objective_from_params(X[i, :], pde_cfg, data)

    Si = sobol.analyze(problem, Y, calc_second_order=cfg.calc_second_order, print_to_console=False)

    df_samples = pd.DataFrame(X, columns=names)
    df_samples["objective"] = Y
    df_samples.to_csv(f"{out_prefix}_samples.csv", index=False)

    df_S1 = pd.DataFrame({"name": names, "S1": Si["S1"], "S1_conf": Si["S1_conf"]})
    df_ST = pd.DataFrame({"name": names, "ST": Si["ST"], "ST_conf": Si["ST_conf"]})
    df_S1.to_csv(f"{out_prefix}_S1.csv", index=False)
    df_ST.to_csv(f"{out_prefix}_ST.csv", index=False)

    return {"samples": df_samples, "S1": df_S1, "ST": df_ST}
