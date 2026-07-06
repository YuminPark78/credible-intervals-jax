# credible-intervals-jax

Unofficial JAX reproduction of **"Not every credible interval is credible"** (Kennedy et al.), evaluating robust Bayesian methods and credible intervals under data contamination.

This repository re-implements the paper's five Bayesian inference models — Normal, Contaminated Normal, Student-*t*, Bayesian Bootstrap (Mean), and Bayesian Bootstrap (Trimmed Mean) — using [JAX](https://github.com/jax-ml/jax) and [NumPyro](https://github.com/pyro-ppl/numpyro) as the probabilistic programming backend.

## Background

The original paper demonstrates that different Bayesian models can produce dramatically different credible intervals when data contains outliers or contaminants. A 95% credible interval from a non-robust model can be misleadingly narrow and confidently wrong, while robust alternatives maintain valid coverage at the cost of slightly wider intervals (the so-called **"robustness tax"**).

This reproduction validates those findings across three experiments:

| Experiment | Description | Figures Reproduced |
|---|---|---|
| **Toy Data Violins** | Posterior distributions from a 6-point dataset with one outlier (`x = 15`) | Figure 3 |
| **Outlier Extremity & Sample Size** | How credible intervals shift as outlier value or sample size changes | Figures 4, 5 |
| **Large-Scale Simulation** | Coverage probability and CI width across 48 scenarios (4 sample sizes × 6 contamination levels × 2 contamination types × 1000 datasets each) | Figures 7, 8 |

## Repository Structure

```
credible-intervals-jax/
├── README.md
├── LICENSE                         # MIT License
├── .gitignore
│
├── src/
│   ├── models/                     # NumPyro generative models
│   │   ├── __init__.py             # Package exports
│   │   ├── normal.py               # Normal likelihood model
│   │   ├── student_t.py            # Student-t likelihood model
│   │   ├── contaminated.py         # Contaminated Normal mixture model
│   │   └── bootstrap.py            # Bayesian Bootstrap (mean & trimmed mean)
│   │
│   └── inference/
│       └── runner.py               # MCMC & Predictive runner utilities
│
├── data/
│   ├── generate_toy_data.py        # Generates the 6-point toy dataset
│   ├── generate_simulation_data.py # Generates batched simulation datasets
│   ├── toy_dataset.npy             # Pre-generated toy data: [-2, -1, 0, 1, 2, 15]
│   ├── sim_*.npy                   # Pre-generated simulation datasets (gitignored)
│   ├── simulation_results.csv      # Aggregated simulation results
│   └── img/                        # Reproduced figures (PNG)
│       ├── Figure_3.png
│       ├── Figure_4.png
│       ├── Figure_5.png
│       ├── Figure_7.png
│       └── Figure_8.png
│
├── notebooks/
│   ├── 01_toy_data_violins.ipynb   # Experiment 1: Violin plots (Figure 3)
│   ├── 02_toy_data_plots.ipynb     # Experiment 2: Outlier extremity & sample size (Figures 4, 5)
│   └── 03_sim_data_plots.ipynb     # Experiment 3: Large-scale simulation study (Figures 7, 8)
│
└── test_vmap_mcmc.py               # Exploratory test for JAX vmap over MCMC
```

## Models

All models are defined as NumPyro generative functions in `src/models/` and use vague priors matching the original paper.

### Parametric Models (MCMC via NUTS)

| Model | File | Likelihood | Priors |
|---|---|---|---|
| **Normal** | `normal.py` | $\mathcal{N}(\mu, \sigma)$ | $\mu \sim \mathcal{N}(0, 100)$, $\tau \sim \text{Gamma}(0.0001, 0.0001)$ |
| **Student-*t*** | `student_t.py` | $t_\nu(\mu, \sigma)$ | Same as Normal + $\nu_{\text{raw}} \sim \text{Exp}(1)$ |
| **Contaminated Normal** | `contaminated.py` | Mixture of two Normals with shared mean | Adds $h \sim \text{Exp}(0.1)$, $\gamma \sim \text{Beta}(1, 9)$, latent $z_i \sim \text{Bernoulli}(\gamma)$ |

### Nonparametric Models (Bayesian Bootstrap via Predictive)

| Model | File | Description |
|---|---|---|
| **BB Mean** | `bootstrap.py` | Weighted mean with Dirichlet-distributed weights (stick-breaking via uniforms) |
| **BB Trimmed Mean** | `bootstrap.py` | 20% trimmed weighted mean: clips cumulative weight mass to $[0.2, 0.8]$ before averaging |

## Getting Started

### Prerequisites

- **Python** ≥ 3.12
- [**uv**](https://docs.astral.sh/uv/getting-started/installation/) — a fast Python package manager (recommended)
- A working **JAX** installation (CPU is sufficient; GPU/TPU optional)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YuminPark78/credible-intervals-jax.git
   cd credible-intervals-jax
   ```

2. **Install dependencies with uv:**

   ```bash
   uv sync
   ```

   This creates a `.venv`, resolves all dependencies from the lockfile (`uv.lock`), and installs them — guaranteeing the exact same package versions across machines.

   > **Note:** For GPU support, follow the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) to install the appropriate `jaxlib` variant after running `uv sync`.

   <details>
   <summary><strong>Alternative: using pip</strong></summary>

   If you prefer not to use `uv`:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install jax jaxlib numpyro numpy matplotlib seaborn pandas joblib tqdm scipy ipykernel
   ```

   </details>

3. **Register the Jupyter kernel** (if using notebooks outside the venv):

   ```bash
   uv run python -m ipykernel install --user --name=credible-intervals-jax
   ```

## Reproducing the Results

### Step 1 — Generate Data

The toy dataset is already committed (`data/toy_dataset.npy`). The simulation `.npy` files are gitignored due to size — regenerate them with:

```bash
python data/generate_toy_data.py        # Creates data/toy_dataset.npy
python data/generate_simulation_data.py  # Creates 48 sim_*.npy files (~50 MB total)
```

### Step 2 — Run the Notebooks

Launch Jupyter and run the notebooks **in order**:

```bash
jupyter notebook notebooks/
```

| Notebook | Runtime Estimate | Output |
|---|---|---|
| `01_toy_data_violins.ipynb` | ~1 min | Violin plots comparing posterior distributions (Figure 3) |
| `02_toy_data_plots.ipynb` | ~15 min | Credible interval panels for outlier extremity (Figure 4) and sample size scaling (Figure 5) |
| `03_sim_data_plots.ipynb` | ~1–2 hrs (CPU) | Coverage probability (Figure 7) and CI width (Figure 8) across 48 scenarios |

> **Tip:** Notebook 03 runs MCMC on 1000 datasets per scenario. On CPU, expect extended runtimes. JAX's JIT compilation makes the first iteration slower; subsequent ones are faster.

## Key Findings (Reproduced)

1. **Non-robust models are confidently wrong.** The Normal model and BB-Mean produce credible intervals that track the contaminated sample mean. As sample size grows, these models become *more* certain of the wrong answer — not less.

2. **Robust models maintain valid coverage.** The Student-*t*, Contaminated Normal, and BB-Trimmed models all discount outliers, preserving ~95% coverage even at high contamination levels.

3. **Robustness is not free.** Under clean data, robust models produce slightly wider intervals (the "robustness tax"). But under contamination, this premium pays off as non-robust intervals either collapse in coverage or explode in width.

4. **Nonparametric ≠ robust.** The BB-Mean is nonparametric but just as sensitive to outliers as the Normal model. Robustness requires explicit mechanism (trimming, mixture components, or heavy tails) — not just flexibility.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use this reproduction in your work, please cite the original paper:

```bibtex
@article{kennedy2023credible,
  title   = {Not Every Credible Interval is Credible},
  author  = {Kennedy, Lauren and Gelman, Andrew},
  year    = {2023}
}
```
