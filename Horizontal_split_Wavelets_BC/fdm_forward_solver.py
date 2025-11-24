# filename: codebase/fdm_forward_solver.py
import os
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set plotting style
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['text.usetex'] = False


class FDMForwardSolver:
    """
    A Finite Difference Method (FDM) solver for the 2D Calderon's problem.

    This class solves the elliptic PDE nabla . (gamma * nabla u) = 0 on a
    square domain [0, 1] x [0, 1] with Dirichlet boundary conditions.
    """

    def __init__(self, N):
        """
        Initializes the FDM solver.

        Args:
            N (int): The number of grid points in each dimension.
        """
        self.N = N
        self.h = 1.0 / (N - 1)
        self.x = np.linspace(0, 1, N)
        self.y = np.linspace(0, 1, N)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.boundary_info = self.get_boundary_info()

    def get_boundary_info(self):
        """
        Identifies boundary nodes, coordinates, and normal vectors.

        Returns:
            dict: A dictionary containing:
                'indices': 1D indices of boundary nodes.
                'coords': (x, y) coordinates of boundary nodes.
                'normals': (nx, ny) normal vectors for each boundary node.
        """
        indices = []
        coords = []
        normals = []
        is_boundary = np.zeros((self.N, self.N), dtype=bool)

        for i in range(self.N):
            for j in range(self.N):
                is_bnd = False
                normal = [0, 0]
                if i == 0:  # Bottom
                    is_bnd = True
                    normal[1] = -1.0
                if i == self.N - 1:  # Top
                    is_bnd = True
                    normal[1] = 1.0
                if j == 0:  # Left
                    is_bnd = True
                    normal[0] = -1.0
                if j == self.N - 1:  # Right
                    is_bnd = True
                    normal[0] = 1.0

                if is_bnd:
                    idx_1d = i * self.N + j
                    indices.append(idx_1d)
                    coords.append((self.x[j], self.y[i]))
                    # Normalize the normal vector in case of corners
                    norm_len = np.sqrt(normal[0]**2 + normal[1]**2)
                    normals.append((normal[0]/norm_len, normal[1]/norm_len))
                    is_boundary[i, j] = True

        # Ensure unique nodes for corners
        unique_indices, unique_idx_map = np.unique(indices, return_index=True)
        
        return {
            'indices': unique_indices,
            'coords': np.array(coords)[unique_idx_map],
            'normals': np.array(normals)[unique_idx_map],
            'mask': is_boundary
        }


    def _assemble_system(self, gamma):
        """
        Assembles the sparse matrix A for the linear system Au=b.

        Args:
            gamma (np.ndarray): A N x N array of conductivity values.

        Returns:
            scipy.sparse.csc_matrix: The assembled sparse matrix A.
        """
        num_nodes = self.N * self.N
        A = sp.lil_matrix((num_nodes, num_nodes))
        h2 = self.h * self.h

        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                p = i * self.N + j

                # Harmonic mean for conductivity at interfaces
                gamma_E = 2 * gamma[i, j] * gamma[i, j + 1] / (gamma[i, j] + gamma[i, j + 1])
                gamma_W = 2 * gamma[i, j] * gamma[i, j - 1] / (gamma[i, j] + gamma[i, j - 1])
                gamma_N = 2 * gamma[i, j] * gamma[i + 1, j] / (gamma[i, j] + gamma[i + 1, j])
                gamma_S = 2 * gamma[i, j] * gamma[i - 1, j] / (gamma[i, j] + gamma[i - 1, j])

                A[p, p] = -(gamma_E + gamma_W + gamma_N + gamma_S) / h2
                A[p, p + 1] = gamma_E / h2  # East
                A[p, p - 1] = gamma_W / h2  # West
                A[p, p + self.N] = gamma_N / h2  # North
                A[p, p - self.N] = gamma_S / h2  # South

        return A.tocsc()

    def solve(self, gamma, boundary_values):
        """
        Solves the forward problem for a given gamma and boundary conditions.

        Args:
            gamma (np.ndarray): The N x N conductivity map.
            boundary_values (dict): A dictionary mapping 1D boundary indices
                                    to potential values.

        Returns:
            np.ndarray: A N x N array of the potential u.
        """
        num_nodes = self.N * self.N
        A = self._assemble_system(gamma)
        b = np.zeros(num_nodes)

        # Apply Dirichlet boundary conditions
        for idx, val in boundary_values.items():
            A[idx, :] = 0
            A[idx, idx] = 1
            b[idx] = val

        u_flat = spsolve(A, b)
        return u_flat.reshape((self.N, self.N))

    def compute_normal_current(self, u, gamma):
        """
        Computes the normal current density J = gamma * du/dn on the boundary.

        Args:
            u (np.ndarray): The N x N potential field.
            gamma (np.ndarray): The N x N conductivity map.

        Returns:
            np.ndarray: A 1D array of normal current densities at boundary points.
        """
        bnd_indices = self.boundary_info['indices']
        bnd_normals = self.boundary_info['normals']
        bnd_coords_map = {tuple(self.boundary_info['coords'][i]): i for i in range(len(bnd_indices))}
        
        J = np.zeros(len(bnd_indices))
        
        for i in range(self.N):
            for j in range(self.N):
                if not self.boundary_info['mask'][i, j]:
                    continue

                coord_tuple = (self.x[j], self.y[i])
                if coord_tuple not in bnd_coords_map:
                    continue
                
                map_idx = bnd_coords_map[coord_tuple]
                nx, ny = bnd_normals[map_idx]
                
                # Compute gradients using second-order one-sided differences
                du_dx = 0.0
                du_dy = 0.0

                # Gradient in x
                if j == 0:  # Left boundary, forward difference
                    du_dx = (-3 * u[i, 0] + 4 * u[i, 1] - u[i, 2]) / (2 * self.h)
                elif j == self.N - 1:  # Right boundary, backward difference
                    du_dx = (3 * u[i, -1] - 4 * u[i, -2] + u[i, -3]) / (2 * self.h)
                else: # Interior point for x-gradient (corners on y-boundary)
                    du_dx = (u[i, j + 1] - u[i, j - 1]) / (2 * self.h)

                # Gradient in y
                if i == 0:  # Bottom boundary, forward difference
                    du_dy = (-3 * u[0, j] + 4 * u[1, j] - u[2, j]) / (2 * self.h)
                elif i == self.N - 1:  # Top boundary, backward difference
                    du_dy = (3 * u[-1, j] - 4 * u[-2, j] + u[-3, j]) / (2 * self.h)
                else: # Interior point for y-gradient (corners on x-boundary)
                    du_dy = (u[i + 1, j] - u[i - 1, j]) / (2 * self.h)

                du_dn = du_dx * nx + du_dy * ny
                J[map_idx] = gamma[i, j] * du_dn
                
        return J


def get_gamma_single_inclusion(N):
    """Generates a conductivity map with a single circular inclusion."""
    gamma = np.ones((N, N))
    center_x, center_y, radius = 0.5, 0.5, 0.2
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)
    mask = (xx - center_x)**2 + (yy - center_y)**2 < radius**2
    gamma[mask] = 2.0
    return gamma

def get_gamma_multiple_inclusions(N):
    """Generates a conductivity map with two circular inclusions."""
    gamma = np.ones((N, N))
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)
    mask1 = (xx - 0.3)**2 + (yy - 0.6)**2 < 0.15**2
    gamma[mask1] = 2.0
    mask2 = (xx - 0.7)**2 + (yy - 0.3)**2 < 0.2**2
    gamma[mask2] = 0.5
    return gamma

def get_gamma_checkerboard(N):
    """Generates a checkerboard conductivity map."""
    gamma = np.ones((N, N))
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)
    checker = (np.floor(xx * 4) + np.floor(yy * 4)) % 2
    gamma[checker == 1] = 2.0
    return gamma

def generate_dtn_data(case_name, gamma_true, N, K):
    """
    Generates and saves the DtN map data for a given conductivity.
    """
    print("--- Starting data generation for case: " + case_name + " ---")
    
    solver = FDMForwardSolver(N)
    bnd_info = solver.boundary_info
    bnd_indices = bnd_info['indices']
    bnd_coords = bnd_info['coords']

    # Define K=32 boundary conditions
    freq_pairs = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2),
                  (3, 0), (0, 3), (3, 1), (1, 3), (3, 2), (2, 3), (3, 3), (4, 0)]
    
    boundary_potentials = []
    clean_currents = []

    start_time = time.time()
    for k in range(K):
        m, n = freq_pairs[k // 2]
        is_sin = k % 2 == 1
        
        # Define boundary function f_k
        if is_sin:
            f_k = np.sin(m * np.pi * bnd_coords[:, 0] + n * np.pi * bnd_coords[:, 1])
        else:
            f_k = np.cos(m * np.pi * bnd_coords[:, 0] + n * np.pi * bnd_coords[:, 1])
        
        boundary_values = {idx: val for idx, val in zip(bnd_indices, f_k)}
        
        # Solve for potential u_k
        u_k = solver.solve(gamma_true, boundary_values)
        
        # Compute normal current J_k
        J_k = solver.compute_normal_current(u_k, gamma_true)
        
        boundary_potentials.append(f_k)
        clean_currents.append(J_k)
        
        if (k + 1) % 8 == 0:
            print("Generated data for " + str(k + 1) + "/" + str(K) + " boundary conditions...")

    boundary_potentials = np.array(boundary_potentials)
    clean_currents = np.array(clean_currents)
    
    # Add noise
    noise_levels = [0.01, 0.05]
    noisy_currents = {}
    
    currents_norm = np.linalg.norm(clean_currents)
    if currents_norm == 0:
        print("Warning: Norm of clean currents is zero. Cannot add relative noise.")
        # Handle case with zero norm to avoid division by zero
        for level in noise_levels:
            key = 'noise_' + str(int(level*100)) + 'pct'
            noisy_currents[key] = clean_currents.copy()
    else:
        for level in noise_levels:
            noise = np.random.normal(0, 1, clean_currents.shape)
            noise_norm = np.linalg.norm(noise)
            scaled_noise = noise * (level * currents_norm / noise_norm)
            key = 'noise_' + str(int(level*100)) + 'pct'
            noisy_currents[key] = clean_currents + scaled_noise

    # Save data
    data_folder = "data_pablo_trial"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    filename = os.path.join(data_folder, "dtn_data_" + case_name + ".npz")
    np.savez_compressed(
        filename,
        gamma_true=gamma_true,
        boundary_coords=bnd_coords,
        boundary_potentials=boundary_potentials,
        clean_currents=clean_currents,
        noisy_currents_1pct=noisy_currents.get('noise_1pct', clean_currents),
        noisy_currents_5pct=noisy_currents.get('noise_5pct', clean_currents),
        grid_N=N,
        num_bcs=K
    )
    
    end_time = time.time()
    print("Data generation for " + case_name + " complete. Time elapsed: " + str(round(end_time - start_time, 2)) + "s")
    print("Dataset saved to " + filename)
    
    # Print statistics
    print("\n--- Dataset Statistics (" + case_name + ") ---")
    print("FDM Grid Size: " + str(N) + "x" + str(N))
    print("Number of Boundary Functions (K): " + str(K))
    print("Boundary Points per Function: " + str(bnd_coords.shape[0]))
    print("Min/Max of Boundary Potential (f_k): " + str(round(np.min(boundary_potentials), 2)) + " / " + str(round(np.max(boundary_potentials), 2)))
    print("Mean of Clean Current Density (J_k): " + str(round(np.mean(clean_currents), 4)))
    print("Std. Dev. of Clean Current Density (J_k): " + str(round(np.std(clean_currents), 2)))
    print("Min/Max of Clean Current Density (J_k): " + str(round(np.min(clean_currents), 2)) + " / " + str(round(np.max(clean_currents), 2)))
    
    # Return one example for plotting
    example_u = solver.solve(gamma_true, {idx: val for idx, val in zip(bnd_indices, boundary_potentials[0])})
    return filename, example_u, boundary_potentials[0], clean_currents[0]


def main():
    """Main function to generate all datasets and plots."""
    N = 128  # Grid size
    K = 32   # Number of boundary conditions
    
    cases = {
        "single_inclusion": get_gamma_single_inclusion(N),
        "multiple_inclusions": get_gamma_multiple_inclusions(N),
        "checkerboard": get_gamma_checkerboard(N)
    }
    
    for name, gamma_true in cases.items():
        filepath, u_ex, f_ex, J_ex = generate_dtn_data(name, gamma_true, N, K)
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Ground Truth Gamma
        im0 = axes[0].imshow(gamma_true, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
        axes[0].set_title("Ground Truth Conductivity (gamma)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im0, ax=axes[0], orientation='vertical')
        
        # Plot 2: Example Potential Solution u_k
        im1 = axes[1].imshow(u_ex, origin='lower', extent=[0, 1, 0, 1], cmap='plasma')
        axes[1].set_title("Example Potential Field u_0(x,y)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(im1, ax=axes[1], orientation='vertical')
        
        # Plot 3: Example DtN data pair (f_k, J_k)
        bnd_info = FDMForwardSolver(N).boundary_info
        bnd_coords = bnd_info['coords']
        # Create a perimeter-like coordinate for plotting
        perimeter = np.zeros(len(bnd_coords))
        for i in range(1, len(bnd_coords)):
            perimeter[i] = perimeter[i-1] + np.linalg.norm(bnd_coords[i] - bnd_coords[i-1])
        
        axes[2].plot(perimeter, f_ex, 'b-', label='Potential f_0')
        ax2_twin = axes[2].twinx()
        ax2_twin.plot(perimeter, J_ex, 'r-', label='Current J_0')
        axes[2].set_title("Example DtN Pair on Boundary")
        axes[2].set_xlabel("Perimeter along Boundary")
        axes[2].set_ylabel("Potential (V)", color='b')
        ax2_twin.set_ylabel("Current Density (A/m^2)", color='r')
        axes[2].grid(True)
        
        fig.tight_layout()
        
        plot_filename = os.path.join("data", "plot_summary_" + name + "_" + str(int(time.time())) + ".png")
        plt.savefig(plot_filename, dpi=300)
        print("\nSummary plot saved to " + plot_filename)
        print("Plot description: Shows the ground truth conductivity, a sample potential solution, and an example DtN pair for the '" + name + "' case.")
        plt.close(fig)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()