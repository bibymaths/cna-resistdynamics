#!/usr/bin/env python3
import argparse
import numpy as np

from phi.flow import *
import phi.math as phimath

from pde_common import load_patient_data, pde_observables_from_grid, nll_ratio_ca

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--time_unit", default="months", choices=["months", "days"])
    ap.add_argument("--sample_list", default=None)
    ap.add_argument("--use_ca125_updated", action="store_true")
    ap.add_argument("--drop_failed", action="store_true")
    ap.add_argument("--require_panel_sequenced", action="store_true")
    ap.add_argument("--require_detected_cna", action="store_true")

    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--dt", type=float, default=1e-3)

    ap.add_argument("--aS", type=float, default=0.5)
    ap.add_argument("--aR", type=float, default=0.3)
    ap.add_argument("--dS", type=float, default=0.4)
    ap.add_argument("--dR", type=float, default=0.1)
    ap.add_argument("--K", type=float, default=1.0)
    ap.add_argument("--DS", type=float, default=1e-2)
    ap.add_argument("--DR", type=float, default=1e-2)

    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--ca0", type=float, default=0.0)
    ap.add_argument("--sigma_ca", type=float, default=0.5)
    ap.add_argument("--w_ca", type=float, default=1.0)
    args = ap.parse_args()

    data = load_patient_data(
        args.data, args.patient,
        time_unit=args.time_unit,
        sample_list_path=args.sample_list,
        use_ca125_updated=args.use_ca125_updated,
        drop_failed=args.drop_failed,
        require_panel_sequenced=args.require_panel_sequenced,
        require_detected_cna=args.require_detected_cna,
    )

    r0 = float(data.ratio[0])
    ca125_0 = float(data.ca125[0])
    N0 = max((ca125_0 - args.ca0) / max(args.gamma, 1e-12), 1e-12)
    S0_val = (1.0 - r0) * N0
    R0_val = r0 * N0

    L = float(args.L)
    N = int(args.N)
    dt = float(args.dt)
    T_total = float(np.max(data.t))
    steps = int(np.ceil(T_total / dt))

    grid_bounds = Box(x=L)
    # Neumann BCs (no flux)
    S = CenteredGrid(lambda x: S0_val, extrapolation.ZERO_GRADIENT, x=N, bounds=grid_bounds)
    R = CenteredGrid(lambda x: R0_val, extrapolation.ZERO_GRADIENT, x=N, bounds=grid_bounds)

    def u_of_t(_t):
        return 1.0

    obs_times = list(map(float, data.t))
    obs_idx = 0
    states_at_obs = []
    dx = L / N

    print(f"[PhiFlow] patient={data.patient} T={T_total:.4g} steps={steps} dt={dt}")

    for n in range(steps + 1):
        t = n * dt
        u_val = u_of_t(t)

        # snapshot before stepping if aligned
        if obs_idx < len(obs_times) and abs(t - obs_times[obs_idx]) <= 0.5 * dt:
            S_vals = S.values.numpy("x").copy()
            R_vals = R.values.numpy("x").copy()
            states_at_obs.append((S_vals, R_vals))
            obs_idx += 1

        # evolve
        N_field = S + R
        growth_S = args.aS * S * (1 - N_field / args.K)
        growth_R = args.aR * R * (1 - N_field / args.K)
        death_S = u_val * args.dS * S
        death_R = u_val * args.dR * R

        S = S + (growth_S - death_S) * dt
        R = R + (growth_R - death_R) * dt

        S = diffuse(S, amount=args.DS * dt)
        R = diffuse(R, amount=args.DR * dt)

        S = phimath.maximum(S, 0.0)
        R = phimath.maximum(R, 0.0)

        if obs_idx >= len(obs_times):
            break

    if len(states_at_obs) != len(data.t):
        raise RuntimeError(f"Captured {len(states_at_obs)} obs snapshots but expected {len(data.t)}. "
                           f"Try increasing dt alignment tolerance or reducing dt.")

    ratio_hat, logca_hat = [], []
    for S_vals, R_vals in states_at_obs:
        _, _, r_pred, lc_pred = pde_observables_from_grid(S_vals, R_vals, dx, gamma=args.gamma, ca0=args.ca0)
        ratio_hat.append(r_pred)
        logca_hat.append(lc_pred)

    ratio_hat = np.asarray(ratio_hat)
    logca_hat = np.asarray(logca_hat)

    nll = nll_ratio_ca(
        ratio_obs=data.ratio,
        se_logit_ratio=data.se_logit_ratio,
        logca_obs=data.log_ca125,
        ratio_hat=ratio_hat,
        logca_hat=logca_hat,
        sigma_ca=args.sigma_ca,
        w_ca=args.w_ca,
    )

    print("\n--- PhiFlow NLL ---")
    print(f"patient={data.patient}")
    print(f"NLL={nll:.6g}")
    print(f"RMSE_ratio={np.sqrt(np.mean((data.ratio - ratio_hat)**2)):.6g}")
    print(f"RMSE_logCA={np.sqrt(np.mean((data.log_ca125 - logca_hat)**2)):.6g}")

if __name__ == "__main__":
    main()
