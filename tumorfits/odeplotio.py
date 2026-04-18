# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .odeio import PatientData
from .odemodel import simulate_states
from .utils import ensure_dir


def pretty_print(title: str, context_names: list[str], theta: np.ndarray, metrics: dict):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for k, v in metrics.items():
        print(f"{k:>14s}: {v:.6g}")
    print("-" * 80)
    print(f"n_params: {theta.size}")
    print("contexts:", context_names)


def plot_gof_scatter_all(df_points: pd.DataFrame, out_prefix: str = "gof"):
    for model in sorted(df_points["model"].unique()):
        for var in ["ratio", "logCA125"]:
            sub = df_points[(df_points["model"] == model) & (df_points["var"] == var)].copy()
            if sub.empty:
                continue

            x = sub["obs"].to_numpy()
            y = sub["pred"].to_numpy()

            plt.figure(figsize=(7, 7))
            plt.scatter(x, y, alpha=0.35, s=18)

            if var == "ratio":
                lo, hi = 0.0, 1.0
            else:
                lo = float(np.nanmin(np.r_[x, y]))
                hi = float(np.nanmax(np.r_[x, y]))

            plt.plot([lo, hi], [lo, hi])
            plt.xlabel("Observed")
            plt.ylabel("Predicted")
            plt.title(f"{model} GOF scatter: {var}")

            out = sub.loc[sub["flag_out95"]].copy()
            if not out.empty:
                out["dev"] = np.abs(out["obs"] - out["pred"])
                out_best = (
                    out.sort_values("dev", ascending=False)
                    .groupby("patient", as_index=False)
                    .head(1)
                )
                for _, r in out_best.iterrows():
                    plt.text(r["obs"], r["pred"], str(r["patient"]), fontsize=8)

            plt.tight_layout()
            plt.savefig(f"{out_prefix}_{model}_{var}.png", dpi=300)
            plt.close()


def save_patient_states_plots(
    data: PatientData,
    theta: np.ndarray,
    out_dir: str,
    *,
    tag: str = "ODE",
    save_csv: bool = True,
    dpi: int = 300,
):
    out_dir = ensure_dir(out_dir)
    pid = str(data.patient)

    S, R, N, r_hat, logca_hat, u_ctx = simulate_states(data, theta)

    if save_csv:
        df = pd.DataFrame(
            {
                "patient": pid,
                "time": data.t.astype(float),
                "context_id": data.context.astype(int),
                "context_name": [data.context_names[i] for i in data.context],
                "S": S.astype(float),
                "R": R.astype(float),
                "N": N.astype(float),
                "r_hat": r_hat.astype(float),
                "ratio_obs": data.ratio.astype(float),
                "logCA_hat": logca_hat.astype(float),
                "logCA_obs": data.log_ca125.astype(float),
            }
        )
        df.to_csv(os.path.join(out_dir, f"{pid}_{tag}_states.csv"), index=False)

        # canonical u_ctx slice begins at 10
        dfu = pd.DataFrame(
            {
                "patient": pid,
                "context_name": data.context_names,
                "u_ctx": u_ctx.astype(float),
                "logit_u_ctx": theta[10 : 10 + len(data.context_names)].astype(float),
            }
        )
        dfu.to_csv(os.path.join(out_dir, f"{pid}_{tag}_u_ctx.csv"), index=False)

    # Plot 1: states
    fig = plt.figure(figsize=(10, 6))
    plt.plot(data.t, S, label="S(t)")
    plt.plot(data.t, R, label="R(t)")
    plt.plot(data.t, N, label="N(t)=S+R")
    plt.xlabel("Time")
    plt.ylabel("Population (a.u.)")
    plt.title(f"{pid} {tag}: simulated states")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{pid}_{tag}_states.png"), dpi=dpi)
    plt.close(fig)

    # Plot 2: fit
    fig = plt.figure(figsize=(10, 6))
    plt.plot(data.t, data.ratio, marker="o", linestyle="-", label="ratio obs")
    plt.plot(data.t, r_hat, marker="o", linestyle="--", label="ratio pred")
    plt.plot(data.t, data.log_ca125, marker="s", linestyle="-", label="logCA125 obs")
    plt.plot(data.t, logca_hat, marker="s", linestyle="--", label="logCA125 pred")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"{pid} {tag}: observed vs predicted")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{pid}_{tag}_fit.png"), dpi=dpi)
    plt.close(fig)

    return {
        "patient": pid,
        "out_dir": out_dir,
        "states_png": os.path.join(out_dir, f"{pid}_{tag}_states.png"),
        "fit_png": os.path.join(out_dir, f"{pid}_{tag}_fit.png"),
        "states_csv": os.path.join(out_dir, f"{pid}_{tag}_states.csv") if save_csv else None,
        "u_csv": os.path.join(out_dir, f"{pid}_{tag}_u_ctx.csv") if save_csv else None,
    }
