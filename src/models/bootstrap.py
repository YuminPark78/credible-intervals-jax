import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def bayesian_bootstrap_mean(data):
    N = data.shape[0]

    # Procedure adopted from Figure 2 of the original paper
    # Step 1: Generate n-1 uniform random numbers from the interval [0, 1]
    with numpyro.plate("n_minus_1", N-1):
        u = numpyro.sample("u", dist.Uniform(0.0, 1.0))

    # Step 2: Order the numbers such that u_1 < u_2 < ... < u_{n-1}
    u_sorted = jnp.sort(u)
    u_with_bounds = jnp.concatenate([jnp.array[0.0], u_sorted, jnp.array[1.0]])

    # Step 3: The weight w_i assigned to x_i is the difference u_i - u_{i-1}
    w = numpyro.deterministic("w", jnp.diff(u_with_bounds))

    # Step 4: The sampled distribution G is constructed by setting P(x=x_i) = w_i
    # Values not represented in the original data set are assigned probability zero
    mu = numpyro.deterministic("mu", jnp.sum(w*data))

def bayesian_bootstrap_trimmed(data, trim_prop=0.20):
    N = data.shape[0]

    # Step 1: Sort the data
    x_sorted = jnp.sort(data)

    # Step 2: Generate n-1 uniform random numbers from the interval [0, 1]
    with numpyro.plate("n_minus_1", N-1):
        u = numpyro.sample("u", dist.Uniform(0.0, 1.0))
    
    # Step 3: Order the numbers such that u_1 < u_2 < ... < u_{n-1}
    u_sorted = jnp.sort(u)
    u_with_bounds = jnp.concatenate([jnp.array[0.0], u_sorted, jnp.array[1.0]])

    # Step 4: The weight w_i assigned to x_i is the difference u_i - u_{i-1}
    w = numpyro.deterministic("w", jnp.diff(u_with_bounds))

    # Step 5: Calculate the cumulative probability mass
    cum_w = jnp.cumsum(w)
    cum_w_prev = jnp.concatenate([jnp.array([0,0]), cum_w[:-1]])

    # Step 6: Clip the mass to the trimming boundaries
    lower_bound = trim_prop
    upper_bound = 1.0 - trim_prop

    w_trimmed = jnp.maximum(
        0.0,
        jnp.minimum(cum_w, upper_bound) - jnp.maximum(cum_w_prev, lower_bound)
    )

    # Step 7: Normalize the remaining weights 
    w_normalized = w_trimmed / (1.0 - 2.0 * trim_prop)

    # Step 8: Calculat ethe expected value of this trimmed distribution
    mu_trimmed = numpyro.deterministic("mu", jnp.sum(w_normalized * x_sorted))