from cmdstanpy import CmdStanModel

model = CmdStanModel(
    stan_file="stan/resistance_state_space.stan",
    cpp_options={"STAN_THREADS": True},
)
fit = model.sample(
    data="stan_data.json",
    chains=4,
    parallel_chains=4,
    threads_per_chain=8,
    iter_warmup=2000,
    iter_sampling=2000,
    adapt_delta=0.99,
    max_treedepth=15,
    seed=42,
)
print(fit.diagnose())
print(fit.summary())
