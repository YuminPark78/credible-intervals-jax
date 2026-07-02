import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def normal_model(data=None):
    
    # Priors (matching the paper)
    mu = numpyro.sample("mu", dist.Normal(0, 100))
    precision = numpyro.sample("precision", dist.Gamma(0.0001, 0.0001))
    sigma = 1.0 / jnp.sqrt(precision)
    
    # Likelihood
    with numpyro.plate("data", len(data) if data is not None else 10):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=data)