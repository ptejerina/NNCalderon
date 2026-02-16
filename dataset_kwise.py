import numpy as np
import torch
from torch.utils.data import Dataset

class CalderonBoundaryKDataset(Dataset):
    """
    Returns full boundary vectors per boundary condition k:
      f_k: (Nb,), J_k: (Nb,)
    """
    def __init__(self, filepath: str, noise_level: float = 0.0):
        data = np.load(filepath)
        self.filepath = filepath
        self.noise_level = float(noise_level)

        self.boundary_coords = torch.tensor(data["boundary_coords"], dtype=torch.float32)           # (Nb,2)
        self.boundary_potentials = torch.tensor(data["boundary_potentials"], dtype=torch.float32)  # (K,Nb)
        self.num_bcs = int(data["num_bcs"])
        self.grid_N = int(data["grid_N"])

        if self.noise_level == 0.0:
            currents = data["clean_currents"]
        elif self.noise_level == 0.01:
            currents = data["noisy_currents_1pct"]
        elif self.noise_level == 0.05:
            currents = data["noisy_currents_5pct"]
        else:
            raise ValueError("noise_level must be 0.0, 0.01, or 0.05")

        self.currents = torch.tensor(currents, dtype=torch.float32)  # (K,Nb)
        self.gamma_true = torch.tensor(data["gamma_true"], dtype=torch.float32) if "gamma_true" in data else None

    def __len__(self):
        return self.num_bcs

    def __getitem__(self, k: int):
        return {
            "k": torch.tensor(k, dtype=torch.long),
            "f_bnd": self.boundary_potentials[k],  # (Nb,)
            "J_bnd": self.currents[k],             # (Nb,)
        }
