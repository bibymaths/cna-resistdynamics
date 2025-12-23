import numpy as np
import pandas as pd

# ---------- Helpers ----------
def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))

def invlogit(z):
    return 1 / (1 + np.exp(-z))

# ---------- Load data ----------
path = "data/liquidCNA_results/Subclonal_ratio_estimates.extended.txt"
df = pd.read_csv(path, sep="\t")

# Keep only usable rows
df = df[df["Accept_estimate"].isin(["yes", "maybe"])].copy()

# Required columns: Patient, Time, context, ratio, ratio_min95, ratio_max95, CA125
required = ["Patient", "Time", "context", "ratio", "ratio_min95", "ratio_max95", "CA125", "Accept_estimate"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ---------- Preprocess ----------
# Sort within patient by time
df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
df = df.dropna(subset=["Time", "ratio", "ratio_min95", "ratio_max95", "CA125"]).copy()
df = df.sort_values(["Patient", "Time"]).reset_index(drop=True)

# Encode patient IDs
patients = df["Patient"].unique().tolist()
P = len(patients)
patient_id = df["Patient"].map({p:i for i,p in enumerate(patients)}).to_numpy()

# Encode contexts (use the context of the *interval*, i.e., previous sample's context)
# We'll store prev_context per row; for the first row of each patient, set to -1
contexts = df["context"].astype(str).unique().tolist()
C = len(contexts)
context_map = {c:i for i,c in enumerate(contexts)}
context_id = df["context"].map(context_map).to_numpy()

# Build previous index within patient and delta-t
prev_idx = np.full(len(df), -1, dtype=int)
dt = np.full(len(df), np.nan, dtype=float)
prev_context = np.full(len(df), -1, dtype=int)

for p in range(P):
    idx = np.where(patient_id == p)[0]
    idx = idx[np.argsort(df.loc[idx, "Time"].to_numpy())]
    for j in range(1, len(idx)):
        cur = idx[j]
        prev = idx[j-1]
        prev_idx[cur] = prev
        dt[cur] = float(df.loc[cur, "Time"] - df.loc[prev, "Time"])
        prev_context[cur] = int(context_id[prev])  # interval context = previous sample context

# Observation for ratio on logit scale with SE from CI in logit space
ratio = np.clip(df["ratio"].to_numpy(), 1e-6, 1-1e-6)
rL = np.clip(df["ratio_min95"].to_numpy(), 1e-6, 1-1e-6)
rU = np.clip(df["ratio_max95"].to_numpy(), 1e-6, 1-1e-6)

y = logit(ratio)
yL = logit(rL)
yU = logit(rU)
se_y = (yU - yL) / 3.92  # approx SE from 95% CI

# If Accept_estimate == "maybe", inflate measurement error
maybe = (df["Accept_estimate"].to_numpy() == "maybe")
se_y = np.where(maybe, se_y * 2.0, se_y)

# Guardrails
se_y = np.clip(se_y, 1e-3, 5.0)

# CA125 (log scale), add epsilon for safety
ca125 = df["CA125"].to_numpy().astype(float)
log_ca125 = np.log(np.clip(ca125, 1e-3, None))

# For Stan-like "first row per patient" indicator
is_first = (prev_idx == -1)

# ---------- PyMC model ----------
import pymc as pm
import pytensor.tensor as pt

N = len(df)

with pm.Model() as model:
    # Hyperpriors for context drift
    mu_context = pm.Normal("mu_context", mu=0.0, sigma=0.5, shape=C)          # drift per day (logit units/day)
    tau_context = pm.HalfNormal("tau_context", sigma=0.5, shape=C)            # variability across patients

    # Patient-specific drift per context
    mu_pc = pm.Normal("mu_pc", mu=mu_context, sigma=tau_context, shape=(P, C))

    # Patient diffusion (process noise)
    sigma_p = pm.HalfNormal("sigma_p", sigma=0.5, shape=P)

    # Initial state per patient
    z0 = pm.Normal("z0", mu=0.0, sigma=2.0, shape=P)

    # Latent state per observation
    z = pm.Normal("z", mu=0.0, sigma=5.0, shape=N)

    # Transition likelihood indices
    idx = np.where(~is_first)[0]
    prev = prev_idx[idx]
    dt_i = dt[idx]
    p_i = patient_id[idx]
    c_prev = prev_context[idx]

    mean_i = z[prev] + mu_pc[p_i, c_prev] * dt_i
    sd_i = sigma_p[p_i] * np.sqrt(dt_i)

    # First obs per patient
    first_idx = np.where(is_first)[0]

    # Add evolution likelihood (cannot use observed=z[...] in PyMC)
    trans_lp = pm.logp(pm.Normal.dist(mu=mean_i, sigma=sd_i), z[idx]).sum()
    init_lp  = pm.logp(pm.Normal.dist(mu=z0[patient_id[first_idx]], sigma=1.0), z[first_idx]).sum()
    pm.Potential("state_evolution", trans_lp + init_lp)

    # CA125 coupling: log(CA125) ~ N(alpha_p + beta*z, sigma_ca)
    alpha_p = pm.Normal("alpha_p", mu=np.mean(log_ca125), sigma=2.0, shape=P)
    beta = pm.Normal("beta", mu=0.0, sigma=1.0)
    sigma_ca = pm.HalfNormal("sigma_ca", sigma=1.0)

    pm.Normal("ca125_obs", mu=alpha_p[patient_id] + beta * z, sigma=sigma_ca, observed=log_ca125)

    # Sample
    idata = pm.sample(
        draws=2000, tune=2000, chains=4, target_accept=0.9, random_seed=42
    )

print(idata)
