import numpy as np
import os

def generate_data():
    """Generates the 6-point toy dataset with an outlier."""
    X = np.array([-2, -1, 0, 1, 2, 15], dtype=np.float32)
    return X

if __name__ == "__main__":
    # Ensure it's saved in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "toy_dataset.npy")
    
    X = generate_data()
    np.save(save_path, X)
    print(f"Saved 6-point toy dataset to {save_path}")
