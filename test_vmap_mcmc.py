import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(data):
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    numpyro.sample("obs", dist.Normal(mu, 1), obs=data)

def run_mcmc(key, data):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=100, num_samples=100, progress_bar=False)
    mcmc.run(key, data=data)
    return mcmc.get_samples()["mu"]

batched_data = jnp.zeros((10, 5))
keys = jax.random.split(jax.random.PRNGKey(0), 10)

try:
    vmap_run = jax.vmap(run_mcmc)
    res = vmap_run(keys, batched_data)
    print("VMAP SUCCESS:", res.shape)
except Exception as e:
    print("VMAP ERROR:", e)
