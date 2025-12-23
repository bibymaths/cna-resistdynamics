data {
  int<lower=1> N;                 // observations
  int<lower=1> P;                 // patients
  int<lower=1> C;                 // contexts
  array[N] int<lower=1, upper=P> pid;
  array[N] int<lower=0, upper=N> prev;    // 0 if first in patient, else previous index (1..N)
  array[N] real<lower=0> dt;              // time difference (days)
  array[N] int<lower=0, upper=C> cprev;   // 0 if first, else context of previous observation
  vector[N] y;                      // logit(ratio)
  vector<lower=1e-6>[N] se_y;       // SE from CI (logit scale)
  vector[N] log_ca125;
}

parameters {
  // Context drift hierarchy
  vector[C] mu_context;                 // global drift per context
  vector<lower=0>[C] tau_context;       // across-patient variability per context
  matrix[P, C] mu_pc_raw;               // patient-context drift (std normal)

  // Process noise (diffusion)
  vector<lower=1e-6>[P] sigma_p;

  // Initial state per patient
  vector[P] z0;

  // Non-centered innovations for the latent process
  vector[N] eta;

  // CA125 model
  vector[P] alpha_p;
  real beta;
  real<lower=1e-6> sigma_ca;
}

transformed parameters {
  matrix[P, C] mu_pc;
  vector[N] z;

  // Patient-context drifts
  for (p in 1:P)
    for (c in 1:C)
      mu_pc[p,c] = mu_context[c] + tau_context[c] * mu_pc_raw[p,c];

  // Build latent trajectory (non-centered)
  for (n in 1:N) {
    if (prev[n] == 0) {
      // first observation for a patient: z0 plus unit-scale noise
      z[n] = z0[pid[n]] + 0.5 * eta[n];
    } else {
      // guard against dt==0 in input (should be fixed in preprocessing, but safe here)
      real dt_eff = fmax(dt[n], 1e-6);
      real sd = sigma_p[pid[n]] * sqrt(dt_eff);

      z[n] = z[prev[n]]
             + mu_pc[pid[n], cprev[n]] * dt_eff
             + sd * eta[n];
    }
  }
}

model {
  // Priors (tighter = fewer divergences)
  mu_context ~ normal(0, 0.3);
  tau_context ~ normal(0, 0.3);
  to_vector(mu_pc_raw) ~ normal(0, 1);

  sigma_p ~ normal(0, 0.5);
  z0 ~ normal(0, 1.5);

  // Non-centered innovations
  eta ~ normal(0, 1);

  // CA125
  alpha_p ~ normal(mean(log_ca125), 1.5);
  beta ~ normal(0, 1);
  sigma_ca ~ normal(0, 1);

  // Observations
  y ~ normal(z, se_y);
  log_ca125 ~ normal(alpha_p[pid] + beta * z, sigma_ca);
}

generated quantities {
  vector[N] r;
  for (n in 1:N) {
    r[n] = inv_logit(z[n]);  // posterior resistant fraction
  }
}