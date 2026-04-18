# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def plot_pde_fit(df_traj: pd.DataFrame, out_path: str, *, title: str = "") -> str:
    ensure_dir(os.path.dirname(out_path) or ".")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(df_traj["time"], df_traj["ratio_obs"], "ko", label="Observed")
    ax[0].plot(df_traj["time"], df_traj["ratio_pred"], "-", linewidth=2, label="PDE")
    ax[0].set_title("Resistant fraction" if not title else f"{title} - fraction")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend()

    ax[1].plot(df_traj["time"], df_traj["logca_obs"], "ko", label="Observed")
    ax[1].plot(df_traj["time"], df_traj["logca_pred"], "-", linewidth=2, label="PDE")
    ax[1].set_title("Log CA125" if not title else f"{title} - logCA125")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_heatmaps(
    x: np.ndarray,
    t: np.ndarray,
    S_mat: np.ndarray,
    R_mat: np.ndarray,
    out_path: str,
    *,
    title: str = "",
) -> str:
    ensure_dir(os.path.dirname(out_path) or ".")
    total = S_mat + R_mat
    frac = np.divide(R_mat, total, out=np.zeros_like(R_mat), where=total > 1e-9)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax[0].imshow(
        total, aspect="auto", origin="lower", extent=[x.min(), x.max(), t.min(), t.max()]
    )
    ax[0].set_title(title + " Total density" if title else "Total density")
    ax[0].set_xlabel("Space (x)")
    ax[0].set_ylabel("Time")
    fig.colorbar(im1, ax=ax[0], label="Cell density")

    im2 = ax[1].imshow(
        frac,
        aspect="auto",
        origin="lower",
        extent=[x.min(), x.max(), t.min(), t.max()],
        vmin=0,
        vmax=1,
    )
    ax[1].set_title(title + " Resistant fraction" if title else "Resistant fraction")
    ax[1].set_xlabel("Space (x)")
    fig.colorbar(im2, ax=ax[1], label="R/(S+R)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
