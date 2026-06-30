import numpy as np
import os

def generate_simulation_data(
    n_samples: int,
    contamination_proportion: float,
    contamination_type: str = "unbiased",
    n_simulations: int = 1000,
    random_state: int = None
):
    """
    Generates data from a target distribution with a specified proportion of contamination.
    
    Args:
        n_samples (int): Total number of samples to generate per simulation.
        contamination_proportion (float): Proportion of samples that are contaminated (0.0 to 0.5).
        contamination_type (str): Type of contamination - "unbiased" or "biased".
        n_simulations (int): Number of simulated datasets to generate.
        random_state (int, optional): Random seed for reproducibility.
        
    Returns:
        np.ndarray: The generated dataset of shape (n_simulations, n_samples).
    """
    if not (0.0 <= contamination_proportion <= 0.5):
        raise ValueError("Contamination proportion must be between 0.0 and 0.5")
    
    if random_state is not None:
        np.random.seed(random_state)
        
    n_contaminated = int(n_samples * contamination_proportion)
    n_target = n_samples - n_contaminated
    
    # Target distribution: Normal(mean=0, std=1)
    target_data = np.random.normal(loc=0.0, scale=1.0, size=(n_simulations, n_target))
    
    # Contaminant distribution
    if contamination_type == "unbiased":
        # Unbiased: Normal(mean=0, std=10)
        contaminant_data = np.random.normal(loc=0.0, scale=10.0, size=(n_simulations, n_contaminated))
    elif contamination_type == "biased":
        # Biased: Normal(mean=10, std=1)
        contaminant_data = np.random.normal(loc=10.0, scale=1.0, size=(n_simulations, n_contaminated))
    elif contamination_type == "none" or contamination_proportion == 0.0:
        contaminant_data = np.zeros((n_simulations, 0))
    else:
        raise ValueError("contamination_type must be 'unbiased' or 'biased'")
        
    # Combine the distributions
    data = np.concatenate([target_data, contaminant_data], axis=1)
    
    # Shuffle the dataset so contaminated points are randomly distributed
    # np.argsort on random numbers gives a random permutation for each row independently
    indices = np.argsort(np.random.rand(n_simulations, n_samples), axis=1)
    data = np.take_along_axis(data, indices, axis=1)
    
    return data

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    sample_sizes = [20, 50, 100, 250]
    proportions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    contamination_types = ["unbiased", "biased"]
    n_simulations = 1000
    
    total_generated = 0
    
    for n in sample_sizes:
        for prop in proportions:
            for c_type in contamination_types:
                data = generate_simulation_data(
                    n_samples=n,
                    contamination_proportion=prop,
                    contamination_type=c_type,
                    n_simulations=n_simulations,
                    random_state=42 + total_generated
                )
                
                # Format: sim_{type}_{prop}_{n}.npy
                filename = f"sim_{c_type}_{int(prop*100)}_{n}.npy"
                path = os.path.join(script_dir, filename)
                np.save(path, data)
                print(f"Saved: {filename} with shape {data.shape}")
                total_generated += 1
                
    print(f"Successfully generated {total_generated} datasets.")
