#!/usr/bin/env python3
import argparse
import numpy as np
import deepxde as dde
import tensorflow as tf

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

    # PDE / obs params
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--Nx", type=int, default=256, help="spatial quadrature points for integrals")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)

    # baseline model params
    ap.add_argument("--aS", type=float, default=0.5)
    ap.add_argument("--aR", type=float, default=0.3)
    ap.add_argument("--dS", type=float, default=0.4)
    ap.add_argument("--dR", type=float, default=0.1)
    ap.add_argument("--K", type=float, default=1.0)
    ap.add_argument("--DS", type=float, default=1e-2)
    ap.add_argument("--DR", type=float, default=1e-2)

    # CA mapping
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

    # infer uniform IC from first datapoint
    r0 = float(data.ratio[0])
    ca125_0 = float(data.ca125[0])
    N0 = max((ca125_0 - args.ca0) / max(args.gamma, 1e-12), 1e-12)
    S0_val = (1.0 - r0) * N0
    R0_val = r0 * N0

    L = float(args.L)
    T_total = float(np.max(data.t))

    # constant therapy baseline
    def u_of_t(t):
        return tf.ones_like(t)  # u(t)=1

    # PDE system
    def pde_system(x, y):
        # x: [x, t]
        S_val = y[:, 0:1]
        R_val = y[:, 1:2]

        dS_dt = dde.grad.jacobian(y, x, i=0, j=1)
        dR_dt = dde.grad.jacobian(y, x, i=1, j=1)

        d2S_dx2 = dde.grad.hessian(y, x, component=0, i=0, j=0)
        d2R_dx2 = dde.grad.hessian(y, x, component=1, i=0, j=0)

        u_val = u_of_t(x[:, 1:2])

        N = S_val + R_val
        growth_S = args.aS * S_val * (1 - N / args.K)
        growth_R = args.aR * R_val * (1 - N / args.K)
        death_S  = u_val * args.dS * S_val
        death_R  = u_val * args.dR * R_val

        eq_S = dS_dt - (args.DS * d2S_dx2 + growth_S - death_S)
        eq_R = dR_dt - (args.DR * d2R_dx2 + growth_R - death_R)
        return [eq_S, eq_R]

    geom = dde.geometry.Interval(0.0, L)
    timedomain = dde.geometry.TimeDomain(0.0, T_total)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def initial_S_func(x):
        return np.full((len(x), 1), S0_val)

    def initial_R_func(x):
        return np.full((len(x), 1), R0_val)

    ic_S = dde.icbc.IC(geomtime, initial_S_func, lambda _, on_initial: on_initial, component=0)
    ic_R = dde.icbc.IC(geomtime, initial_R_func, lambda _, on_initial: on_initial, component=1)

    def boundary(_, on_boundary):
        return on_boundary

    def neumann(component):
        return lambda x, y, _: dde.grad.jacobian(y, x, i=component, j=0)

    bc_S = dde.icbc.OperatorBC(geomtime, neumann(0), boundary)
    bc_R = dde.icbc.OperatorBC(geomtime, neumann(1), boundary)

    net = dde.maps.FNN([2] + [64] * 3 + [2], "tanh", "glorot_uniform")
    dde_data = dde.data.TimePDE(
        geomtime, pde_system, [ic_S, ic_R, bc_S, bc_R],
        num_domain=2000, num_boundary=200, num_initial=200,
    )

    model = dde.Model(dde_data, net)
    model.compile("adam", lr=args.lr)
    print(f"[DeepXDE] training for patient={data.patient} T={T_total:.4g} epochs={args.epochs}")
    model.train(epochs=args.epochs)

    # Evaluate on observed times: integrate via quadrature over x
    xs = np.linspace(0.0, L, args.Nx)
    dx = L / (args.Nx - 1)

    ratio_hat = []
    logca_hat = []
    for ti in data.t:
        X = np.column_stack([xs, np.full_like(xs, float(ti))])
        pred = model.predict(X)
        S_vals = pred[:, 0]
        R_vals = pred[:, 1]
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

    print("\n--- DeepXDE NLL ---")
    print(f"patient={data.patient}")
    print(f"NLL={nll:.6g}")
    print(f"RMSE_ratio={np.sqrt(np.mean((data.ratio - ratio_hat)**2)):.6g}")
    print(f"RMSE_logCA={np.sqrt(np.mean((data.log_ca125 - logca_hat)**2)):.6g}")

if __name__ == "__main__":
    main()
