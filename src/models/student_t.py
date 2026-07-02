import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def student_t_model(data):

    # Priors (matching the paper)
    mu = numpyro.sample("mu", dist.Normal(0, 100))
    precision = numpyro.sample("precision", dist.Gamma(0.0001, 0.0001))
    sigma = 1.0 / jnp.sqrt(precision)
    df_raw = numpyro.sample("df_raw", dist.Exponential(1.0))
    df = numpyro.deterministic("df", df_raw + 0.001)

    # Likelihood
    with numpyro.plate("data", len(data) if data is not None else 10):
        numpyro.sample("obs", dist.StudentT(df, mu, sigma), obs = data)