import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def contaminated_normal_model(data=None):

    # Priors (matching the paper)
    mu = numpyro.sample("mu", dist.Normal(0, 100))
    precision1 = numpyro.sample("precision1", dist.Gamma(0.0001, 0.0001))
    sigma1 = 1.0 / jnp.sqrt(precision1)

    h_raw = numpyro.sample("h_raw", dist.Exponential(0.1))
    h = numpyro.deterministic("h", h_raw+0.01)

    # Contamination Proportion
    gamma = numpyro.sample("gamma", dist.Beta(1, 9))

    with numpyro.plate("data", len(data) if data is not None else 10):
        z = numpyro.sample("z", dist.Bernoulli(gamma), infer={'enumerate': 'parallel'})

        sigma2 = sigma1 * h
        sigma_effective = sigma1 * (1-z) + sigma2 * z

        # Likelihood
        numpyro.sample("obs", dist.Normal(mu, sigma_effective), obs=data)