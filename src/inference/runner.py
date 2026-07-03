import jax
import jax.numpy as jsonpatch
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from joblib import Parallel, delayed

# For notebooks '01_toy_data_violins.ipynb' and '02_toy_data_plots.ipynb'

def run_parametric_mcmc(model, data, num_samples=4000, num_warmup=1000, seed=0):
    rng_key = jax.random.PRNGKey(seed)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, data=data)
    return mcmc.get_samples()

# For notebook '03_sim_data_plots.ipynb'
# 1. Parametric Runner

def run_parametric_parallel_mcmc(model, batched_data, num_samples=1000, num_warmup=500, n_jobs=-1, seed=42):
    """
    Runs MCMC on an array of datasets concurrently using JAX hardware vectorization.

    Args:
        model: The NumPyro model
        batched_data: Array of shape (n_simulations, n_samples)
        num_samples: Number of samples per MCMC run
        num_warmup: Number of warmup steps
        n_jobs: Ignored (kept for compatibility)
        seed: Random seed
    
    Returns:
        np.ndarray: Posterior samples of 'mu' with shape (n_simulations, num_samples).
    """
    n_simulations = batched_data.shape[0]

    print(f"Vectorizing MCMC over {n_simulations} datasets...")

    base_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(base_key, n_simulations)

    def _single_mcmc(key, single_data):
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
        mcmc.run(key, data=single_data)
        return mcmc.get_samples()["mu"]

    vmapped_mcmc = jax.vmap(_single_mcmc, in_axes=(0, 0))
    results = vmapped_mcmc(keys, batched_data)

    return np.array(results)

# 2. Nonparametric Runner

def run_nonparametric_vectorized(model, batched_data, num_samples=1000, seed=42):
    """
    Runs Bayesian Bootstrap on an array of datasets using JAX hardware vectorization.

    Args:
        model: The NumPyro BB model
        batched_data: Array of shape (n_simulations, n_samples)

    Returns:
        np.ndarray: Posterior samples of 'mu' with shape (n_simulations, num_samples)
    """
    n_simulations = batched_data.shape[0]

    # Generated exactly 1000 unique PRNG keys for the 1000 datasets
    base_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(base_key, n_simulations)

    # Define a pure function that takes a SINGLE key and SINGLE dataset
    def _single_predictive(key, single_data):
        pred = Predictive(model, num_samples=num_samples)
        return pred(key, data=single_data)["mu"]
    
    print(f"Vectorizing Predictive over {n_simulations} datasets...")

    vmapped_predictive = jax.vmap(_single_predictive, in_axes=(0,0))

    results = vmapped_predictive(keys, batched_data)

    return np.array(results)