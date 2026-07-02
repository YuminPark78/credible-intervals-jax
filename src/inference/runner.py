import jax
import numpyro
from numpyro.infer import MCMC, NUTS

def run_parametric_mcmc(model, data, num_samples=4000, num_warmup=1000, seed=0):
    rng_key = jax.random.PRNGKey(seed)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, data=data)
    return mcmc.get_samples()