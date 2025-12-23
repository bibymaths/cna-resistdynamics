data {
  int<lower=1> N;                 // observations
  int<lower=1> P;                 // patients
  int<lower=1> C;                 // contexts
  array[N] int<lower=1, upper=P> pid;
  array[N] int<lower=0, upper=N> prev;    // 0 if first in patient, else previous index (1..N)
  array[N] real<lower=0> dt;              // time difference (days)
  array[N] int<lower=0, upper=C> cprev;   // 0 if first, else context of previous observation
  vector[N] y;                     // logit(ratio)
  vector<lower=1e-6>[N] se_y;       // SE from CI
  vector[N] log_ca125;
}

parameters {
  vector[C] mu_context;                 // global drift per context
  vector<lower=0>[C] tau_context;       // across-patient variability per context
  matrix[P, C] mu_pc_raw;               // patient-context drift (raw)

  vector<lower=0>[P] sigma_p;           // diffusion per patient
  vector[P] z0;                         // initial state per patient
  vector[N] z;                          // latent state per observation

  vector[P] alpha_p;                    // CA125 baseline per patient
  real beta;                            // CA125 coupling to resistance state
  real<lower=0> sigma_ca;               // CA125 noise
}

transformed parameters {
  matrix[P, C] mu_pc;
  for (p in 1:P)
    for (c in 1:C)
      mu_pc[p,c] = mu_context[c] + tau_context[c] * mu_pc_raw[p,c];
}

model {
  // Priors
  mu_context ~ normal(0, 0.5);
  tau_context ~ normal(0, 0.5);
  to_vector(mu_pc_raw) ~ normal(0, 1);

  sigma_p ~ normal(0, 0.5);
  z0 ~ normal(0, 2);

  alpha_p ~ normal(mean(log_ca125), 2);
  beta ~ normal(0, 1);
  sigma_ca ~ normal(0, 1);

  // State evolution
  for (n in 1:N) {
    if (prev[n] == 0) {
      z[n] ~ normal(z0[pid[n]], 1.0);
    } else {
      z[n] ~ normal(
        z[prev[n]] + mu_pc[pid[n], cprev[n]] * dt[n],
        sigma_p[pid[n]] * sqrt(dt[n] + 1e-12)
      );
    }
  }

  // Ratio observation
  y ~ normal(z, se_y);

  // CA125 observation
  log_ca125 ~ normal(alpha_p[pid] + beta * z, sigma_ca);
}