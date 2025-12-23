from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file="resistance_state_space.stan")
fit = model.sample(data="stan_data.json", chains=4, iter_warmup=2000, iter_sampling=2000, seed=42)
print(fit.summary())
