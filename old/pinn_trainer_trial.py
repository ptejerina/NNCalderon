# filename: codebase/pinn_trainer.py
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import Dataset, DataLoader

# Ensure the codebase modules are accessible
from pinn_framework import FourierFeatureEncoder, ConductivityNetwork, PotentialNetwork, CalderonPINN, init_weights
from fdm_forward_solver import FDMForwardSolver

# Set plotting style
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['text.usetex'] = False


class CalderonDataset(Dataset):
    """
    PyTorch Dataset for loading Calderon's problem DtN data.
    """
    def __init__(self, filepath, noise_level):
        """
        Initializes the dataset.

        Args:
            filepath (str): Path to the .npz data file.
            noise_level (float): Noise level to use (0.0, 0.01, or 0.05).
        """
        self.filepath = filepath
        self.noise_level = noise_level

        data = np.load(filepath)
        self.boundary_coords = torch.tensor(data['boundary_coords'], dtype=torch.float32)
        self.boundary_potentials = torch.tensor(data['boundary_potentials'], dtype=torch.float32)
        self.num_bcs = int(data['num_bcs'])
        self.grid_N = int(data['grid_N'])

        if noise_level == 0.0:
            self.currents = torch.tensor(data['clean_currents'], dtype=torch.float32)
        elif noise_level == 0.01:
            self.currents = torch.tensor(data['noisy_currents_1pct'], dtype=torch.float32)
        elif noise_level == 0.05:
            self.currents = torch.tensor(data['noisy_currents_5pct'], dtype=torch.float32)
        else:
            raise ValueError("Unsupported noise level")

        solver = FDMForwardSolver(N=self.grid_N)
        self.normals = torch.tensor(solver.boundary_info['normals'], dtype=torch.float32)

        self.k_indices = torch.arange(self.num_bcs).unsqueeze(1).expand(-1, self.boundary_coords.shape[0])
        
        self.flat_coords = self.boundary_coords.repeat(self.num_bcs, 1)
        self.flat_k_indices = self.k_indices.flatten()
        self.flat_normals = self.normals.repeat(self.num_bcs, 1)
        self.flat_potentials = self.boundary_potentials.flatten().unsqueeze(1)
        self.flat_currents = self.currents.flatten().unsqueeze(1)

    def __len__(self):
        return len(self.flat_k_indices)

    def __getitem__(self, idx):
        # Tensors are kept on CPU here and moved to device in the training loop
        return (
            self.flat_coords[idx],
            self.flat_k_indices[idx],
            self.flat_normals[idx],
            self.flat_potentials[idx],
            self.flat_currents[idx]
        )


class Trainer:
    """
    A class to handle the training of the Calderon PINN.
    """
    def __init__(self, config, num_bcs):
        """
        Initializes the Trainer.

        Args:
            config (dict): A dictionary of hyperparameters.
            num_bcs (int): The number of boundary conditions (K).
        """
        self.config = config
        self.num_bcs = num_bcs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: " + str(self.device))

        self.ffe_encoder = FourierFeatureEncoder(
            input_dims=2,
            mapping_size=config['ffe_mapping_size'],
            scale=config['ffe_scale'],
            device=self.device
        )
        self.gamma_net = ConductivityNetwork(
            ffe_dims=config['ffe_mapping_size'] * 2,
            layers=config['gamma_net_layers'],
            neurons=config['gamma_net_neurons']
        ).to(self.device)
        self.u_net = PotentialNetwork(
            ffe_dims=config['ffe_mapping_size'] * 2,
            num_bcs=self.num_bcs,
            layers=config['u_net_layers'],
            neurons=config['u_net_neurons']
        ).to(self.device)

        self.gamma_net.apply(init_weights)
        self.u_net.apply(init_weights)

        self.pinn_manager = CalderonPINN(
            self.ffe_encoder, self.gamma_net, self.u_net, self.num_bcs, self.device
        )

        params = list(self.gamma_net.parameters()) + list(self.u_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config['lr_decay_gamma']
        )

        self.loss_history = {
            'total': [], 'pde': [], 'data': [], 'bc': [], 'neumann': [], 'tv': []
        }

    def train(self, dataset, case_name, noise_level_str):
        """
        The main training loop.
        """
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size_bnd'], shuffle=True)
        
        print("Starting training for case: " + case_name + ", noise: " + noise_level_str)
        start_time = time.time()

        for epoch in range(1, self.config['epochs'] + 1):
            for bnd_batch_cpu in dataloader:
                self.optimizer.zero_grad()

                # Move batch to the correct device here
                bnd_batch = [t.to(self.device) for t in bnd_batch_cpu]
                xy_bnd, k_bnd, normals_bnd, f_bnd, J_bnd = bnd_batch

                xy_colloc = torch.rand(self.config['batch_size_pde'], 2, device=self.device)
                k_colloc = torch.randint(0, self.num_bcs, (self.config['batch_size_pde'],), device=self.device)
                
                pde_batch = (xy_colloc, k_colloc)

                total_loss, loss_dict = self.pinn_manager.compute_full_loss(
                    pde_batch, bnd_batch, self.config['loss_weights']
                )

                total_loss.backward()
                self.optimizer.step()

            for key in self.loss_history:
                self.loss_history[key].append(loss_dict[key])

            if epoch % self.config['lr_decay_step'] == 0:
                self.scheduler.step()

            if epoch % 1000 == 0:
                # Fix: Enclose multi-line string concatenation in parentheses
                log_str = ("Epoch " + str(epoch) + "/" + str(self.config['epochs']) + 
                           " | Total Loss: " + str(float('%.2e' % loss_dict['total'])) + 
                           " | PDE: " + str(float('%.2e' % loss_dict['pde'])) + 
                           " | Data: " + str(float('%.2e' % loss_dict['data'])) + 
                           " | TV: " + str(float('%.2e' % loss_dict['tv'])) )
                print(log_str)

        end_time = time.time()
        print("Training finished. Total time: " + str(round(end_time - start_time, 2)) + "s")

    def predict_gamma(self, N=256):
        """Predicts the conductivity map on a uniform grid."""
        self.gamma_net.eval()
        x = torch.linspace(0, 1, N, device=self.device)
        y = torch.linspace(0, 1, N, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        xy_grid = torch.stack([yy.flatten(), xx.flatten()], dim=1)
        
        with torch.no_grad():
            xy_ffe = self.ffe_encoder.encode(xy_grid)
            gamma_pred = self.gamma_net(xy_ffe)
        
        return gamma_pred.reshape(N, N).cpu().numpy()

    def save_results(self, case_name, noise_level_str):
        """Saves models, predictions, and loss history."""
        data_folder = "data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            
        base_filename = "pinn_results_" + case_name + "_" + noise_level_str
        
        torch.save(self.gamma_net.state_dict(), os.path.join(data_folder, base_filename + "_gamma_net.pth"))
        
        gamma_pred = self.predict_gamma()
        np.save(os.path.join(data_folder, base_filename + "_gamma_pred.npy"), gamma_pred)
        
        np.savez(os.path.join(data_folder, base_filename + "_loss_history.npz"), **self.loss_history)
        
        print("Saved final models, predictions, and loss history for " + case_name + " (" + noise_level_str + ")")
        
        plot_loss_curves(self.loss_history, case_name, noise_level_str)


def plot_loss_curves(loss_history, case_name, noise_level_str):
    """Plots and saves the loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(loss_history['total']) + 1)
    
    ax.plot(epochs, loss_history['total'], label='Total Loss', color='black', linewidth=2)
    ax.plot(epochs, loss_history['pde'], label='PDE Loss', linestyle='--')
    ax.plot(epochs, loss_history['data'], label='Data Loss', linestyle='--')
    ax.plot(epochs, loss_history['tv'], label='TV Loss', linestyle='--')
    
    ax.set_yscale('log')
    ax.set_title("Training Loss Curves for " + case_name + " (" + noise_level_str + ")")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    fig.tight_layout()
    
    plot_filename = os.path.join("data", "plot_loss_curves_" + case_name + "_" + noise_level_str + "_" + str(int(time.time())) + ".png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    print("Loss curve plot saved to " + plot_filename)
    print("Plot description: Shows the evolution of total, PDE, data, and TV loss components during training.")


def main():
    """Main function to run all training experiments."""
    CONFIG = {
        'epochs': 100,
        'learning_rate': 1e-3,
        'lr_decay_gamma': 0.9,
        'lr_decay_step': 2000,
        'batch_size_pde': 4096,
        'batch_size_bnd': 1024,
        'ffe_mapping_size': 256,
        'ffe_scale': 10.0,
        'gamma_net_layers': 4,
        'gamma_net_neurons': 64,
        'u_net_layers': 6,
        'u_net_neurons': 128,
        'loss_weights': {'pde': 1.0, 'data': 10.0, 'tv': 1e-4}
    }

    CASES = ["single_inclusion", "multiple_inclusions", "checkerboard"]
    NOISE_LEVELS = {
        "0pct": 0.0,
        "1pct": 0.01,
        "5pct": 0.05
    }

    for case in CASES:
        for noise_str, noise_val in NOISE_LEVELS.items():
            print("\n" + "="*60)
            print("RUNNING EXPERIMENT: CASE=" + case + ", NOISE=" + noise_str)
            print("="*60 + "\n")
            
            data_filepath = os.path.join("..", "data", "dtn_data_" + case + ".npz")
            if not os.path.exists(data_filepath):
                print("Data file not found: " + data_filepath + ". Skipping.")
                continue

            dataset = CalderonDataset(data_filepath, noise_val)
            trainer = Trainer(CONFIG, num_bcs=dataset.num_bcs)
            
            trainer.train(dataset, case, noise_str)
            trainer.save_results(case, noise_str)


if __name__ == "__main__":
    # Fix: Set multiprocessing start method to 'spawn' for CUDA safety
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()