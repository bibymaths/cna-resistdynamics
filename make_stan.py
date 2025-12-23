import numpy as np
import pandas as pd
import json

def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x/(1-x))

path = "data/liquidCNA_results/Subclonal_ratio_estimates.extended.txt"
df = pd.read_csv(path, sep="\t")
df = df[df["Accept_estimate"].isin(["yes","maybe"])].copy()

df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
df = df.dropna(subset=["Time", "ratio", "ratio_min95", "ratio_max95", "CA125"]).copy()
df = df.sort_values(["Patient", "Time"]).reset_index(drop=True)

patients = df["Patient"].unique().tolist()
contexts = df["context"].astype(str).unique().tolist()
P, C = len(patients), len(contexts)

pid = df["Patient"].map({p:i+1 for i,p in enumerate(patients)}).to_numpy()   # 1-based
cid = df["context"].map({c:i+1 for i,c in enumerate(contexts)}).to_numpy()   # 1-based

N = len(df)
prev = np.zeros(N, dtype=int)
dt = np.zeros(N, dtype=float)
cprev = np.zeros(N, dtype=int)

for p in range(1, P+1):
    idx = np.where(pid == p)[0]
    idx = idx[np.argsort(df.loc[idx, "Time"].to_numpy())]
    for j in range(len(idx)):
        cur = idx[j]
        if j == 0:
            prev[cur] = 0
            dt[cur] = 0.0
            cprev[cur] = 0
        else:
            pr = idx[j-1]
            prev[cur] = pr + 1  # 1-based index into z
            dt[cur] = float(df.loc[cur, "Time"] - df.loc[pr, "Time"])
            cprev[cur] = int(cid[pr])  # previous sample context

ratio = np.clip(df["ratio"].to_numpy(), 1e-6, 1-1e-6)
rL = np.clip(df["ratio_min95"].to_numpy(), 1e-6, 1-1e-6)
rU = np.clip(df["ratio_max95"].to_numpy(), 1e-6, 1-1e-6)

y = logit(ratio)
yL = logit(rL)
yU = logit(rU)
se_y = (yU - yL) / 3.92

maybe = (df["Accept_estimate"].to_numpy() == "maybe")
se_y = np.where(maybe, se_y * 2.0, se_y)
se_y = np.clip(se_y, 1e-3, 5.0)

ca125 = df["CA125"].to_numpy().astype(float)
log_ca125 = np.log(np.clip(ca125, 1e-3, None))

stan_data = {
    "N": int(N),
    "P": int(P),
    "C": int(C),
    "pid": pid.tolist(),
    "prev": prev.tolist(),
    "dt": dt.tolist(),
    "cprev": cprev.tolist(),
    "y": y.tolist(),
    "se_y": se_y.tolist(),
    "log_ca125": log_ca125.tolist(),
}

with open("stan_data.json", "w") as f:
    json.dump(stan_data, f, indent=2)

print("Wrote stan_data.json")
print("Patients:", patients)
print("Contexts:", contexts)
