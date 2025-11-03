

# filename: Calderon_routine.py
import torch
import torch.nn as nn
import numpy as np

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

#from fdm_forward_solver import FDMForwardSolver ##COMMENTED THIS TO SEE IF ISSUE IS RESOLVED
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.interpolate import RegularGridInterpolator

import os
import copy
import time
import matplotlib as mpl
from torch.utils.data import Dataset, DataLoader

# ---------- Matplotlib defaults ----------
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['text.usetex'] = False



########ADDED THIS TO SEE IF ISSUE WITH FDM FORWARD SOLVER IS RESOLVED########

class FDMForwardSolver:
    """
    A Finite Difference Method (FDM) solver for the 2D Calderon's problem.

    This class solves the elliptic PDE ∇·(γ ∇u) = 0 on [0,1]×[0,1]
    with Dirichlet boundary conditions.
    """

    def __init__(self, N):
        self.N = N
        self.h = 1.0 / (N - 1)
        self.x = np.linspace(0, 1, N)
        self.y = np.linspace(0, 1, N)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.boundary_info = self.get_boundary_info()

    def get_boundary_info(self):
        indices = []
        coords = []
        normals = []
        is_boundary = np.zeros((self.N, self.N), dtype=bool)

        for i in range(self.N):
            for j in range(self.N):
                is_bnd = False
                normal = [0, 0]
                if i == 0:           # Bottom
                    is_bnd = True
                    normal[1] = -1.0
                if i == self.N - 1:  # Top
                    is_bnd = True
                    normal[1] = 1.0
                if j == 0:           # Left
                    is_bnd = True
                    normal[0] = -1.0
                if j == self.N - 1:  # Right
                    is_bnd = True
                    normal[0] = 1.0

                if is_bnd:
                    idx_1d = i * self.N + j
                    indices.append(idx_1d)
                    coords.append((self.x[j], self.y[i]))
                    norm_len = np.sqrt(normal[0]**2 + normal[1]**2)
                    normals.append((normal[0]/norm_len, normal[1]/norm_len))
                    is_boundary[i, j] = True

        unique_indices, unique_idx_map = np.unique(indices, return_index=True)
        return {
            'indices': unique_indices,
            'coords': np.array(coords)[unique_idx_map],
            'normals': np.array(normals)[unique_idx_map],
            'mask': is_boundary
        }

    def _assemble_system(self, gamma):
        """
        Assemble the sparse matrix A for Au=b using harmonic means at interfaces.
        gamma is (N,N) array of conductivities.
        """
        num_nodes = self.N * self.N
        A = sp.lil_matrix((num_nodes, num_nodes))
        h2 = self.h * self.h

        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                p = i * self.N + j

                # Harmonic mean for interface conductivities
                gamma_E = 2 * gamma[i, j] * gamma[i, j + 1] / (gamma[i, j] + gamma[i, j + 1])
                gamma_W = 2 * gamma[i, j] * gamma[i, j - 1] / (gamma[i, j] + gamma[i, j - 1])
                gamma_N = 2 * gamma[i, j] * gamma[i + 1, j] / (gamma[i, j] + gamma[i + 1, j])
                gamma_S = 2 * gamma[i, j] * gamma[i - 1, j] / (gamma[i, j] + gamma[i - 1, j])

                A[p, p] = -(gamma_E + gamma_W + gamma_N + gamma_S) / h2
                A[p, p + 1]      =  gamma_E / h2  # East
                A[p, p - 1]      =  gamma_W / h2  # West
                A[p, p + self.N] =  gamma_N / h2  # North
                A[p, p - self.N] =  gamma_S / h2  # South

        return A.tocsc()

    def solve(self, gamma, boundary_values):
        """
        Solve for u given gamma and Dirichlet boundary values (dict idx->value).
        gamma : (N,N) ndarray
        """
        num_nodes = self.N * self.N
        A = self._assemble_system(gamma)
        b = np.zeros(num_nodes)

        for idx, val in boundary_values.items():
            A[idx, :] = 0
            A[idx, idx] = 1
            b[idx] = val

        u_flat = spsolve(A, b)
        return u_flat.reshape((self.N, self.N))

    def compute_normal_current(self, u, gamma):
        """
        Compute boundary normal current J = gamma * (du/dn) using second-order
        one-sided stencils on the boundary.
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
                
                # Gradients (second-order one-sided on boundary)
                if j == 0:      # Left
                    du_dx = (-3 * u[i, 0] + 4 * u[i, 1] - u[i, 2]) / (2 * self.h)
                elif j == self.N - 1:  # Right
                    du_dx = (3 * u[i, -1] - 4 * u[i, -2] + u[i, -3]) / (2 * self.h)
                else:
                    du_dx = (u[i, j + 1] - u[i, j - 1]) / (2 * self.h)

                if i == 0:      # Bottom
                    du_dy = (-3 * u[0, j] + 4 * u[1, j] - u[2, j]) / (2 * self.h)
                elif i == self.N - 1:  # Top
                    du_dy = (3 * u[-1, j] - 4 * u[-2, j] + u[-3, j]) / (2 * self.h)
                else:
                    du_dy = (u[i + 1, j] - u[i - 1, j]) / (2 * self.h)

                du_dn = du_dx * nx + du_dy * ny
                J[map_idx] = gamma[i, j] * du_dn
                
        return J
################################################################################


# ==============================
# Fourier feature encoder (kept for u_net only)
# ==============================
class FourierFeatureEncoder:
    """
    Implements Fourier Feature Encoding for spatial coordinates.
    Maps input coordinates to a higher-dimensional space to help NNs
    learn high-frequency functions.
    """
    def __init__(self, input_dims, mapping_size, scale, device):
        self.input_dims = input_dims
        self.mapping_size = mapping_size
        self.scale = scale
        self.device = device
        self.B = torch.randn((mapping_size, input_dims), device=self.device) * self.scale
        self.B.requires_grad = False

    def encode(self, x):
        x_proj = (2. * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ==============================
# Potential network (unchanged)
# ==============================
class PotentialNetwork(nn.Module):
    """
    Neural network to approximate the potential u_k(x, y).
    Input: Concatenation of Fourier-encoded spatial coordinates and a
           one-hot encoding for the boundary condition index k.
    Output: Scalar potential value.
    """
    def __init__(self, ffe_dims, num_bcs, layers=6, neurons=128):
        super().__init__()
        input_dim = ffe_dims + num_bcs
        net_layers = []
        for _ in range(layers):
            net_layers.append(nn.Linear(input_dim, neurons))
            net_layers.append(nn.SiLU())
            input_dim = neurons
        net_layers.append(nn.Linear(neurons, 1))
        self.network = nn.Sequential(*net_layers)

    def forward(self, x_ffe, c_k):
        net_input = torch.cat([x_ffe, c_k], dim=1)
        return self.network(net_input)


# ==============================
# True gamma interpolator (unchanged helper)
# ==============================
def load_gamma_interpolator(gamma):
    """
    Builds an interpolator callable returning torch.Tensor [N,1]
    from a (N,N) numpy gamma field on [0,1]x[0,1].
    """
    N = gamma.shape[0]
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    interp = RegularGridInterpolator((x, y), gamma, bounds_error=False, fill_value=None)

    def gamma_true_fn(xy_torch):
        xy_np = xy_torch.detach().cpu().numpy()
        gamma_vals = interp(xy_np)
        return torch.tensor(gamma_vals, dtype=xy_torch.dtype, device=xy_torch.device).unsqueeze(1)
    return gamma_true_fn


# ==============================
# NEW: Parametric gamma = sum c_i * phi_i(x,y)
# ==============================
def make_fourier2d_basis(k_list, Lx=1.0, Ly=1.0, include_cos=True, include_sin=True):
    """
    Build a list of basis functions phi_i(x,y) using 2D Fourier modes.
    Each phi returns shape [N,1]. Uses torch ops (autograd-safe).

    Args:
        k_list: list of (kx, ky) integer tuples.
        Lx, Ly: domain lengths (default 1.0, i.e., x,y in [0,1]).
        include_cos/sin: whether to include cos and/or sin parts.

    Returns:
        basis_fns: list of callables phi(xy)->[N,1]
    """
    basis_fns = []
    two_pi = 2.0 * np.pi

    for (kx, ky) in k_list:
        if include_cos:
            def phi_cos(xy, kx=kx, ky=ky, Lx=Lx, Ly=Ly):
                x = xy[:, 0:1]; y = xy[:, 1:2]
                return torch.cos(two_pi * (kx * x / Lx + ky * y / Ly))
            basis_fns.append(phi_cos)
        if include_sin:
            def phi_sin(xy, kx=kx, ky=ky, Lx=Lx, Ly=Ly):
                x = xy[:, 0:1]; y = xy[:, 1:2]
                return torch.sin(two_pi * (kx * x / Lx + ky * y / Ly))
            basis_fns.append(phi_sin)

    return basis_fns


# class ParametricConductivity(nn.Module):
#     """
#     gamma(x,y) = sum_i c_i * phi_i(x,y) with fixed phi_i and trainable c_i.
#     Optionally enforces positivity or bounds.
#     """
#     def __init__(self, basis_fns, c_init=None, positivity='softplus',
#                  min_gamma=None, max_gamma=None):
#         """
#         Args:
#             basis_fns: list of callables phi(xy)->[N,1] using torch ops
#             c_init: initial coefficients list/ndarray (len M) or None (zeros)
#             positivity: 'none'|'softplus'|'sigmoid_bounds'
#             min_gamma,max_gamma: used if positivity=='sigmoid_bounds'
#         """
#         super().__init__()
#         self.basis_fns = basis_fns
#         M = len(basis_fns)
#         if c_init is None:
#             c_init = np.zeros(M, dtype=np.float32)
#         self.raw_c = nn.Parameter(torch.tensor(c_init, dtype=torch.float32))  # [M,]
#         self.positivity = positivity
#         self.min_gamma = min_gamma
#         self.max_gamma = max_gamma

#         if positivity == 'sigmoid_bounds':
#             if min_gamma is None or max_gamma is None:
#                 raise ValueError("Provide min_gamma and max_gamma for 'sigmoid_bounds'.")

#     def forward(self, xy):
#         """
#         Args:
#             xy: [N,2] raw coordinates in [0,1]^2 (or your domain)
#         Returns:
#             gamma(xy): [N,1]
#         """
#         # Stack all phi_i(xy) into Phi: [N, M]
#         Phi_cols = [phi(xy) for phi in self.basis_fns]  # each [N,1]
#         Phi = torch.cat(Phi_cols, dim=1)                # [N, M]
#         gamma = Phi @ self.raw_c.unsqueeze(-1)          # [N,1]

#         if self.positivity == 'softplus':
#             gamma = torch.nn.functional.softplus(gamma) + 1e-8
#         elif self.positivity == 'sigmoid_bounds':
#             gamma = torch.sigmoid(gamma) * (self.max_gamma - self.min_gamma) + self.min_gamma
#         # else: 'none' -> no constraints

#         return gamma  # [N,1]


##############TRY THIS NEW BUT NOT SURE############

class ParametricConductivity(nn.Module):
    """
    gamma(x,y) = background + sum_i c_i * phi_i(x,y)
    with fixed phi_i and trainable coefficients c_i.
    Optionally enforces positivity or bounds.
    """
    def __init__(self,
                 basis_fns,
                 c_init=None,
                 positivity='none',
                 min_gamma=None,
                 max_gamma=None,
                 background_init=1.0,
                 learn_background=True):
        super().__init__()
        self.basis_fns = basis_fns
        M = len(basis_fns)
        if c_init is None:
            c_init = np.zeros(M, dtype=np.float32)

        self.raw_c = nn.Parameter(torch.tensor(c_init, dtype=torch.float32))  # [M,]
        self.background = nn.Parameter(torch.tensor(float(background_init), dtype=torch.float32),
                                       requires_grad=learn_background)

        self.positivity = positivity
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        if positivity == 'sigmoid_bounds' and (min_gamma is None or max_gamma is None):
            raise ValueError("Provide min_gamma and max_gamma for 'sigmoid_bounds'.")

    def forward(self, xy):
        # Stack all phi_i(xy) into Phi: [N, M]
        Phi_cols = [phi(xy) for phi in self.basis_fns]   # each [N,1]
        Phi = torch.cat(Phi_cols, dim=1) if len(Phi_cols) > 0 else None

        # background + linear combo
        if Phi is None:   # degenerate case: no basis functions
            gamma = self.background.expand(xy.shape[0], 1)
        else:
            gamma = self.background + (Phi @ self.raw_c.unsqueeze(-1))  # [N,1]

        if self.positivity == 'softplus':
            gamma = torch.nn.functional.softplus(gamma) + 1e-8
        elif self.positivity == 'sigmoid_bounds':
            gamma = torch.sigmoid(gamma) * (self.max_gamma - self.min_gamma) + self.min_gamma
        # 'none' -> no constraints
        return gamma  # [N,1]



# ==============================
# Calderon PINN (updated to call gamma_model(xy))
# ==============================
class CalderonPINN:
    """
    Encapsulates the PINN framework for Calderon's inverse problem.
    Manages networks, loss computations, and automatic differentiation.
    """
    def __init__(self, ffe_encoder, gamma_model, u_net, num_bcs, device, synthetic_data=None):
        self.ffe_encoder = ffe_encoder
        self.gamma_model = gamma_model     # <--- changed name/type
        self.u_net = u_net
        self.num_bcs = num_bcs
        self.device = device

        if synthetic_data is not None:
            self.data = np.load(synthetic_data)
            gamma_true = self.data["gamma_true"]
            self.gamma_true = load_gamma_interpolator(gamma_true)
        self.FLAG = True

    def _to_one_hot(self, k_indices, num_classes):
        one_hot = torch.zeros(k_indices.size(0), num_classes, device=self.device)
        one_hot.scatter_(1, k_indices.unsqueeze(1), 1)
        return one_hot

    def compute_pde_residual(self, xy_colloc, k_indices):
        """Computes the PDE residual: ∇·(γ ∇u) at collocation points."""
        xy_colloc.requires_grad_(True)

        # u-net input still uses FFE + one-hot BC
        xy_ffe = self.ffe_encoder.encode(xy_colloc)
        k_one_hot = self._to_one_hot(k_indices, self.num_bcs)
        u = self.u_net(xy_ffe, k_one_hot)

        # gamma is now parametric and depends on raw xy
        gamma = self.gamma_model(xy_colloc)

        # ∇u
        grad_u = torch.autograd.grad(u, xy_colloc, torch.ones_like(u), create_graph=True)[0]
        du_dx = grad_u[:, 0:1]
        du_dy = grad_u[:, 1:2]

        # γ∇u
        flux_x = gamma * du_dx
        flux_y = gamma * du_dy

        # div(γ∇u)
        div_flux_x = torch.autograd.grad(flux_x, xy_colloc, torch.ones_like(flux_x), create_graph=True)[0][:, 0:1]
        div_flux_y = torch.autograd.grad(flux_y, xy_colloc, torch.ones_like(flux_y), create_graph=True)[0][:, 1:2]
        return div_flux_x + div_flux_y

    def force_gamma_true_addloss(self, xy_colloc):
        """MSE between ground-truth gamma and parametric gamma."""
        xy_colloc.requires_grad_(True)
        gamma_nn = self.gamma_model(xy_colloc)           # [N,1]
        gamma_num = self.gamma_true(xy_colloc)           # [N,1]
        addloss = torch.mean((gamma_num - gamma_nn) ** 2)
        if self.FLAG:
            print('Forcing True gamma with AddLoss!')
            self.FLAG = False
        return addloss

    def compute_data_predictions(self, xy_bnd, k_indices, normals):
        """Predicts potential u and normal current J on the boundary."""
        xy_bnd.requires_grad_(True)

        # u prediction
        xy_ffe = self.ffe_encoder.encode(xy_bnd)
        k_one_hot = self._to_one_hot(k_indices, self.num_bcs)
        u_pred = self.u_net(xy_ffe, k_one_hot)

        # ∂u/∂n
        grad_u = torch.autograd.grad(u_pred, xy_bnd, torch.ones_like(u_pred), create_graph=True)[0]
        du_dn_pred = torch.sum(grad_u * normals, dim=1, keepdim=True)

        # γ on boundary (parametric)
        gamma_pred = self.gamma_model(xy_bnd)

        # J = γ ∂u/∂n
        J_pred = gamma_pred * du_dn_pred
        return u_pred, J_pred

    def compute_full_loss(self, pde_batch, bnd_batch, weights):
        xy_colloc, k_colloc = pde_batch
        xy_bnd, k_bnd, normals_bnd, f_bnd, J_bnd = bnd_batch

        # PDE
        pde_residual = self.compute_pde_residual(xy_colloc, k_colloc)
        loss_pde = torch.mean(pde_residual ** 2)

        # Data (f and J on boundary)
        u_pred, J_pred = self.compute_data_predictions(xy_bnd, k_bnd, normals_bnd)
        loss_bc = torch.mean((u_pred - f_bnd) ** 2)
        loss_nd = torch.mean((J_pred - J_bnd) ** 2)

        # Optional: force true gamma
        if weights.get('force_true_gamma', 0.0) != 0.0:
            loss_force_gamma = self.force_gamma_true_addloss(xy_colloc=xy_colloc)
        else:
            loss_force_gamma = torch.tensor([0.0], device=self.device)

        total_loss = (weights['pde'] * loss_pde
                      + weights['dirichlet_bc'] * loss_bc
                      + weights['neumann_bc'] * loss_nd
                      + weights.get('force_true_gamma', 0.0) * loss_force_gamma)

        loss_dict = {
            'total': total_loss.detach().item(),
            'pde': loss_pde.detach().item(),
            'data': (loss_bc + loss_nd).detach().item(),
            'bc': loss_bc.detach().item(),
            'neumann': loss_nd.detach().item(),
            'force_true_gamma': loss_force_gamma.detach().item(),
        }
        return total_loss, loss_dict

    def predict_gamma(self, N=256):
        """Predicts γ on a uniform grid via the parametric model."""
        self.u_net.eval()           # (u_net eval; gamma_model has only c's, no dropout anyway)
        self.gamma_model.eval()

        x = torch.linspace(0, 1, N, device=self.device)
        y = torch.linspace(0, 1, N, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        xy_grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        with torch.no_grad():
            gamma_pred = self.gamma_model(xy_grid)
        return gamma_pred.reshape(N, N).cpu().numpy()

    # -------- Visualization helpers (unchanged except gamma_model name) --------
    def plot_u_field(self, k_idx=0, N=200, plot_fig=True):
        device = self.device
        ffe_encoder = self.ffe_encoder
        u_net = self.u_net
        num_bcs = self.num_bcs

        x = torch.linspace(0, 1, N, device=device)
        y = torch.linspace(0, 1, N, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        xy_grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        xy_ffe = ffe_encoder.encode(xy_grid)

        k_one_hot = torch.zeros(xy_grid.shape[0], num_bcs, device=device)
        k_one_hot[:, k_idx] = 1.0

        with torch.no_grad():
            u_pred = u_net(xy_ffe, k_one_hot)

        u_pred_img = u_pred.reshape(N, N).cpu().numpy()

        if plot_fig:
            fig = plt.figure(figsize=(5,5))
            plt.imshow(u_pred_img, extent=[0,1,0,1], origin="lower", cmap="plasma")
            plt.colorbar(label=r"$u_{NN}(x,y)$")
            plt.title(f"Induced u(x,y) for k={k_idx}")
            plt.xlabel("x"); plt.ylabel("y")
            plt.tight_layout()
            plt.show()
        return u_pred_img

    def plot_boundary_predictions(self, dataset, k_idx=0, use_gamma_th=False):
        device = self.device
        ffe_encoder = self.ffe_encoder
        u_net = self.u_net

        xy_bnd = dataset.boundary_coords.to(device)
        bnd_coords = xy_bnd
        normals_bnd = dataset.normals.to(device)

        f_true = dataset.boundary_potentials[k_idx].unsqueeze(1).to(device)
        J_true = dataset.currents[k_idx].unsqueeze(1).to(device)

        xy_bnd.requires_grad_(True)
        xy_ffe = ffe_encoder.encode(xy_bnd)

        k_one_hot = torch.zeros(xy_bnd.shape[0], dataset.num_bcs, device=device)
        k_one_hot[:, k_idx] = 1.0

        u_pred = u_net(xy_ffe, k_one_hot)

        grad_u = torch.autograd.grad(u_pred, xy_bnd, torch.ones_like(u_pred), create_graph=True)[0]
        du_dn_pred = torch.sum(grad_u * normals_bnd, dim=1, keepdim=True)

        if not use_gamma_th:
            gamma_pred = self.gamma_model(xy_bnd)
        else:
            gamma_pred = self.gamma_true(xy_bnd)

        f_pred = u_pred
        J_pred = gamma_pred * du_dn_pred

        return bnd_coords, f_pred, J_pred


# ==============================
# Dataset (unchanged)
# ==============================
class CalderonDataset(Dataset):
    """
    PyTorch Dataset for loading Calderon's problem DtN data.
    """
    def __init__(self, filepath, noise_level):
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
        return (
            self.flat_coords[idx],
            self.flat_k_indices[idx],
            self.flat_normals[idx],
            self.flat_potentials[idx],
            self.flat_currents[idx]
        )


# ==============================
# Trainer (updated to use ParametricConductivity)
# ==============================
class Trainer:
    """
    Handles training of the Calderon PINN with parametric gamma.
    """
    def __init__(self, config, num_bcs, synthetic_data=None, saving_path=None):
        self.config = config
        self.num_bcs = num_bcs

        self.path = saving_path if saving_path is not None else os.getcwd() + "/new_training/"
        if os.path.isdir(self.path):
            print("Saving directory exists:", self.path)
        else:
            os.mkdir(self.path)
            print("Saving directory created:", self.path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # --- u-net encoder (stays the same) ---
        self.ffe_encoder = FourierFeatureEncoder(
            input_dims=2,
            mapping_size=config['ffe_mapping_size'],
            scale=config['ffe_scale'],
            device=self.device
        )

        # --- u-net (stays the same) ---
        self.u_net = PotentialNetwork(
            ffe_dims=config['ffe_mapping_size'] * 2,
            num_bcs=self.num_bcs,
            layers=config['u_net_layers'],
            neurons=config['u_net_neurons']
        ).to(self.device)

        # --- NEW: build parametric gamma model ---
        # Example: a small set of 2D Fourier modes. You can change k_list freely.
        gb = config.get('gamma_basis', {})
        k_list = gb.get('k_list', [(1,0),(0,1),(1,1),(2,0),(0,2),(2,2),(1,2),(2,1)])  # example modes
        Lx = gb.get('Lx', 1.0)
        Ly = gb.get('Ly', 1.0)
        include_cos = gb.get('include_cos', True)
        include_sin = gb.get('include_sin', True)
        basis_fns = make_fourier2d_basis(k_list, Lx=Lx, Ly=Ly, include_cos=include_cos, include_sin=include_sin)

        c_init = config.get('gamma_coeff_init', None)
        positivity = config.get('gamma_positivity', 'softplus')  # 'none'|'softplus'|'sigmoid_bounds'
        min_gamma = config.get('gamma_min', None)
        max_gamma = config.get('gamma_max', None)

        # self.gamma_model = ParametricConductivity(
        #     basis_fns=basis_fns,
        #     c_init=c_init,
        #     positivity=positivity,
        #     min_gamma=min_gamma,
        #     max_gamma=max_gamma
        # ).to(self.device)
        
        ##############ADD THIS NEW NOT SURE##############
        
        self.gamma_model = ParametricConductivity(
            basis_fns=basis_fns,
            c_init=c_init,
            positivity=positivity,
            min_gamma=min_gamma,
            max_gamma=max_gamma,
            background_init=config.get('gamma_background_init', 1.0),
            learn_background=config.get('gamma_learn_background', True)
        ).to(self.device)


        self.synthetic_data = synthetic_data

        # --- PINN manager ---
        self.pinn_manager = CalderonPINN(
            self.ffe_encoder, self.gamma_model, self.u_net, self.num_bcs, self.device, synthetic_data=synthetic_data
        )

        # --- Optimizer (train u_net + c's only) ---
        params = list(self.gamma_model.parameters()) + list(self.u_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config['lr_decay_gamma']
        )

        self.loss_history = {
            'total': [], 'pde': [], 'data': [], 'bc': [], 'neumann': [], 'force_true_gamma': []
        }

    def train(self, dataset, case_name, noise_level_str, train_epochs=None, disable_progress_bar=True):
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size_bnd'], shuffle=True)
        print(f"Starting training for case: {case_name}, noise: {noise_level_str}")
        start_time = time.time()

        epochs = train_epochs if train_epochs is not None else self.config['epochs']

        for epoch in trange(1, epochs + 1, desc="Training epochs", disable=disable_progress_bar):
            for bnd_batch_cpu in dataloader:
                self.optimizer.zero_grad()

                # Move batch to device
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
                log_str = (
                    f"Epoch {epoch}/{epochs} | Total: {loss_dict['total']:.2e} "
                    f"| PDE: {loss_dict['pde']:.2e} | Data: {loss_dict['data']:.2e}"
                )
                print(log_str)

        end_time = time.time()
        print(f"Training finished. Total time: {round(end_time - start_time, 2)}s")

    def predict_gamma(self, N=256):
        return self.pinn_manager.predict_gamma(N=N)

    # -------- Save/Load updated for gamma_model --------
    def save_model(self, path):
        u_net_state = copy.deepcopy(self.u_net.state_dict())
        gamma_model_state = copy.deepcopy(self.gamma_model.state_dict())
        FFE_rand_matrix = self.ffe_encoder.B

        state = {
            'u_net_state': u_net_state,
            'gamma_model_state': gamma_model_state,
            'optimizer_state': copy.deepcopy(self.optimizer.state_dict()),
            'train_loss': self.loss_history['total'],
            'de_loss': self.loss_history['pde'],
            'dirichlet_loss': self.loss_history['bc'],
            'neumann_loss': self.loss_history['neumann'],
            'loss_weights': self.config['loss_weights'],
            'FFE_rand_matrix': FFE_rand_matrix
        }
        torch.save(state, path + '_epochs_' + str(len(self.loss_history['total'])))
        print('Model successfully saved')

    def load_model(self, path):
        device = self.device
        master_dict = torch.load(path, map_location=device)
        self.u_net.load_state_dict(master_dict['u_net_state'])
        self.gamma_model.load_state_dict(master_dict['gamma_model_state'])
        self.optimizer.load_state_dict(master_dict['optimizer_state'])

        self.u_net.to(device)
        self.gamma_model.to(device)
        self.ffe_encoder.B = master_dict['FFE_rand_matrix']

        # Move optimizer states to correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        self.loss_history['total'] = master_dict['train_loss']
        self.loss_history['pde'] = master_dict['de_loss']
        self.loss_history['bc'] = master_dict['dirichlet_loss']
        self.loss_history['neumann'] = master_dict['neumann_loss']
        self.config['loss_weights'] = master_dict['loss_weights']

        print('Model successfully loaded')

    # -------- plotting utilities (adapted only where necessary) --------
    def update_optimizer(self, lr=None):
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('LR updated to:', lr)
        else:
            for param_group in self.optimizer.param_groups:
                print('Current LR is:', param_group['lr'])

    def plot_loss(self, save_fig=False):
        loss = self.loss_history
        w = self.config['loss_weights']

        DE_loss = w['pde'] * np.array(loss['pde'])
        f_bc_loss = w['dirichlet_bc'] * np.array(loss['bc'])
        J_bc_loss = w['neumann_bc'] * np.array(loss['neumann'])
        force_gamma_addloss = w.get('force_true_gamma', 0.0) * np.array(loss['force_true_gamma'])
        tot_loss = np.array(loss['total'])

        fig = plt.figure(figsize=(5,4))
        plt.plot(tot_loss, label='Tot')
        plt.plot(DE_loss, label='DE')
        plt.plot(f_bc_loss, label=r'Dirichlet $f_k$')
        plt.plot(J_bc_loss, label=r'Neumann $J_k$')
        if w.get('force_true_gamma', 0.0) != 0.0:
            plt.plot(force_gamma_addloss, label=r'Force $\gamma_{true}$')

        plt.yscale('log'); plt.legend(loc=(1.01,0.35))
        if save_fig:
            fig.savefig(f'{self.path}/all_losses_epoch_{len(DE_loss)}.pdf')
        plt.show()

        print('Min DE loss', np.min(DE_loss))
        print('Min f loss', np.min(f_bc_loss))
        print('Min J loss', np.min(J_bc_loss))
    
    
    ####ADDED THIS NOT TO PLOT LOG SCALE LOSS NOT SURE####
    # def plot_loss(self, save_fig=False, log_scale=False):  # default now: linear
    #     loss = self.loss_history
    #     w = self.config['loss_weights']

    #     DE_loss = w['pde'] * np.array(loss['pde'])
    #     f_bc_loss = w['dirichlet_bc'] * np.array(loss['bc'])
    #     J_bc_loss = w['neumann_bc'] * np.array(loss['neumann'])
    #     force_gamma_addloss = w.get('force_true_gamma', 0.0) * np.array(loss['force_true_gamma'])
    #     tot_loss = np.array(loss['total'])

    #     fig = plt.figure(figsize=(6,4))
    #     plt.plot(tot_loss, label='Total', color='C0', linewidth=2.0)
    #     plt.plot(DE_loss, label='PDE', color='C1', linewidth=1.6)
    #     plt.plot(f_bc_loss, label='Dirichlet $f_k$', color='C2', linewidth=1.6)
    #     plt.plot(J_bc_loss, label='Neumann $J_k$', color='C3', linewidth=1.6)
    #     if w.get('force_true_gamma', 0.0) != 0.0:
    #         plt.plot(force_gamma_addloss, label=r'Force $\gamma_{true}$')

    #     if log_scale:
    #         plt.yscale('log')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss (linear)' if not log_scale else 'Loss (log scale)')
    #     plt.legend(loc='upper right', frameon=True, framealpha=0.9)

    #     # plt.legend(loc='best')
    #     plt.tight_layout()

    #     if save_fig:
    #         suffix = "_linear" if not log_scale else "_log"
    #         fig.savefig(f'{self.path}/all_losses_epoch_{len(DE_loss)}{suffix}.pdf')
    #     plt.show()

    #     print('Min PDE loss', np.min(DE_loss))
    #     print('Min Dirichlet loss', np.min(f_bc_loss))
    #     print('Min Neumann loss', np.min(J_bc_loss))


    def plot_gamma_nn(self, N=128, save_fig=False):
        data = np.load(self.synthetic_data)
        gamma_nn = self.predict_gamma(N=N)

        x = torch.linspace(0, 1, N)
        y = torch.linspace(0, 1, N)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        xy_grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        gamma_true = load_gamma_interpolator(data['gamma_true'])(xy_grid).reshape(N, N).cpu().detach().numpy()

        eps = 1e-12
        rel_error = np.log10(np.abs((gamma_true - gamma_nn) / (gamma_true + eps)))

        fig, ax = plt.subplots(1,3, figsize=(15,4))
        im0 = ax[0].imshow(gamma_nn, origin='lower', extent=[0, 1, 0, 1], cmap="viridis")
        fig.colorbar(im0, ax=ax[0], label=r"NN $\gamma(x,y)$")
        ax[0].set_title(r"$\gamma^{param}(x,y)$")

        im1 = ax[1].imshow(gamma_true, origin='lower', extent=[0, 1, 0, 1], cmap="viridis")
        fig.colorbar(im1, ax=ax[1], label=r"True $\gamma(x,y)$")
        ax[1].set_title(r"True $\gamma(x,y)$")

        im2 = ax[2].imshow(rel_error, origin='lower', extent=[0, 1, 0, 1], cmap="viridis")
        fig.colorbar(im2, ax=ax[2], label=r"Log(rel error)")
        ax[2].set_title(r"$\log_{10}$ relative error")

        plt.tight_layout()
        if save_fig:
            fig.savefig(f'{self.path}/gamma_recovered_epoch_{len(np.array(self.loss_history["pde"]))}.pdf')
        plt.show()

    def plot_residuals(self, save_fig=False, show_fig=False):
        K = int(self.num_bcs)
        n_epochs = len(np.array(self.loss_history['pde']))

        for k_choice in range(K):
            N = 64
            x = np.linspace(0, 1, N)
            y = np.linspace(0, 1, N)
            xx, yy = np.meshgrid(x, y)
            xy = np.stack([xx.flatten(), yy.flatten()], axis=1)
            xy_torch = torch.tensor(xy, dtype=torch.float32, device=self.pinn_manager.device)

            k_indices = torch.full((xy_torch.shape[0],), k_choice, dtype=torch.long, device=self.pinn_manager.device)
            residuals = self.pinn_manager.compute_pde_residual(xy_torch.clone().requires_grad_(True), k_indices)
            residuals_np = residuals.cpu().detach().numpy().reshape(N, N)

            fig, ax = plt.subplots(1, 2, figsize=(9, 3))
            im0 = ax[0].imshow(residuals_np, origin="lower", extent=[0,1,0,1],
                               cmap="plasma", vmin=-np.abs(residuals_np).max(), vmax=np.abs(residuals_np).max())
            fig.colorbar(im0, ax=ax[0], label=r"PDE residual $\nabla \cdot (\gamma \nabla u)$")
            ax[0].set_title(fr"PDE Residual (Linear) for BC k={k_choice}")
            ax[0].set_xlabel("x"); ax[0].set_ylabel("y")

            eps = 1e-8
            im1 = ax[1].imshow(np.abs(residuals_np), origin="lower", extent=[0,1,0,1],
                               cmap="viridis", norm=colors.LogNorm(vmin=eps, vmax=np.abs(residuals_np).max()))
            fig.colorbar(im1, ax=ax[1], label=r"$|\nabla \cdot (\gamma \nabla u)|$ (log scale)")
            ax[1].set_title(r"Log-scaled PDE Residual"); ax[1].set_xlabel("x"); ax[1].set_ylabel("y")

            if save_fig:
                fig.savefig(f'{self.path}/residual_k_{k_choice}_epochs_{n_epochs}.pdf')

            plt.tight_layout()
            if show_fig:
                plt.show()
            else:
                plt.close(fig)

    def plot_boundary_data(self, dataset, save_fig=False, show_fig=True):
        n_epochs = len(np.array(self.loss_history['pde']))

        K = int(self.num_bcs)
        k_list = np.arange(K)

        data = np.load(self.synthetic_data)
        gamma_true = data["gamma_true"]
        bnd_coords_true = data["boundary_coords"]
        u_all_true = data["induced_potentials"]
        f_all_true = data["boundary_potentials"]
        J_all_true = data["clean_currents"]
        N_true = int(data["grid_N"])

        x_true, y_true = bnd_coords_true[:,0], bnd_coords_true[:,1]
        tol_true = 1e-12
        bottom_idx_true = np.where(np.abs(y_true - 0.0) < tol_true)[0]
        top_idx_true    = np.where(np.abs(y_true - 1.0) < tol_true)[0]
        left_idx_true   = np.where(np.abs(x_true - 0.0) < tol_true)[0]
        right_idx_true  = np.where(np.abs(x_true - 1.0) < tol_true)[0]
        bottom_idx_true = bottom_idx_true[np.argsort(x_true[bottom_idx_true])]
        top_idx_true    = top_idx_true[np.argsort(x_true[top_idx_true])][::-1]
        right_idx_true  = right_idx_true[np.argsort(y_true[right_idx_true])]
        left_idx_true   = left_idx_true[np.argsort(y_true[left_idx_true])][::-1]
        perimeter_idx_true = np.concatenate([bottom_idx_true, right_idx_true, top_idx_true, left_idx_true])
        afine_true = np.arange(4*(N_true))

        for k in k_list:
            f_true = f_all_true[k, perimeter_idx_true]
            J_true = J_all_true[k, perimeter_idx_true]

            bnd_coords, f_pred, J_pred = self.pinn_manager.plot_boundary_predictions(dataset, k_idx=k)

            N = int(bnd_coords.shape[0] / 4)
            x, y = bnd_coords[:,0].cpu().detach().numpy(), bnd_coords[:,1].cpu().detach().numpy()

            tol = 1e-12
            bottom_idx = np.where(np.abs(y - 0.0) < tol)[0]
            top_idx    = np.where(np.abs(y - 1.0) < tol)[0]
            left_idx   = np.where(np.abs(x - 0.0) < tol)[0]
            right_idx  = np.where(np.abs(x - 1.0) < tol)[0]

            bottom_idx = bottom_idx[np.argsort(x[bottom_idx])]
            top_idx    = top_idx[np.argsort(x[top_idx])][::-1]
            right_idx  = right_idx[np.argsort(y[right_idx])]
            left_idx   = left_idx[np.argsort(y[left_idx])][::-1]

            perimeter_idx = np.concatenate([bottom_idx[:-1], right_idx[:-1], top_idx[:-1], left_idx[:-1]])
            f_nn = f_pred.cpu().detach().numpy()[perimeter_idx]
            J_nn = J_pred.cpu().detach().numpy()[perimeter_idx]
            afine = np.arange(4*(N))

            fig, ax = plt.subplots(1,2, figsize=(10,2))
            ax[0].plot(afine, f_nn, 'b-', label=r"NN $f(x,y)$")
            ax[0].plot(afine_true, f_true, 'b', linestyle='dashed', label=r"True $f(x,y)$")
            ax[1].plot(afine, J_nn, 'r-', label=r"NN $J(x,y)$")
            ax[1].plot(afine_true, J_true, 'r', linestyle='dashed', label=r"True $J(x,y)$")
            for i in range(1,4):
                ax[0].axvline(i*N, color='k', linestyle='--', linewidth=0.5)
                ax[1].axvline(i*N, color='k', linestyle='--', linewidth=0.5)
            ax[0].set_xlabel('Perimeter'); ax[1].set_xlabel('Perimeter')
            ax[0].set_ylabel('f(x,y)');    ax[1].set_ylabel('J(x,y)')
            ax[0].set_xlim(left=0, right=4*N-1)
            ax[1].set_xlim(left=0, right=4*N-1)
            plt.suptitle(f'k={k}; Bottom - Right - Top - Left Boundary')

            if save_fig:
                fig.savefig(f'{self.path}/BC_f_and_J_recovered_k_{k}_epoch_{n_epochs}.pdf')

            if show_fig:
                plt.show()
            else:
                plt.close(fig)

    def plot_induced_potential_u(self, save_fig=False, show_fig=True):
        n_epochs = len(np.array(self.loss_history['pde']))

        K = int(self.num_bcs)
        data = np.load(self.synthetic_data)
        u_all_true = data["induced_potentials"]

        for k in range(K):
            u_num = u_all_true[k, :, :]
            u_nn = self.pinn_manager.plot_u_field(k_idx=k, N=int(u_all_true.shape[-1]), plot_fig=False).T

            rel_error_u = np.log10(np.abs((u_num - u_nn)))
            fig, ax = plt.subplots(1,3, figsize=(14,3))
            im0 = ax[0].imshow(u_nn, origin='lower', extent=[0, 1, 0, 1], cmap='plasma')
            fig.colorbar(im0, ax=ax[0], label=r"NN $u(x,y)$")
            im1 = ax[1].imshow(u_num, origin='lower', extent=[0, 1, 0, 1], cmap='plasma')
            fig.colorbar(im1, ax=ax[1], label=r"True $u(x,y)$")
            im2 = ax[2].imshow(rel_error_u, origin='lower', extent=[0, 1, 0, 1], cmap='plasma')
            fig.colorbar(im2, ax=ax[2], label=r"Log rel. error")
            ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
            ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
            ax[2].set_xlabel("x"); ax[2].set_ylabel("y")
            plt.suptitle(rf"k={k}")

            if save_fig:
                fig.savefig(f'{self.path}/u_induced_k_{k}_compared_epoch_{n_epochs}.pdf')

            if show_fig:
                plt.show()
            else:
                plt.close(fig)
    
    #ADDED LATER TO SAVE c VECTOR


    def get_learned_c(self):
        """Return the current learned c vector as a NumPy array."""
        return self.gamma_model.raw_c.detach().cpu().numpy()

    def save_learned_c(self, path_prefix):
        """Save the c vector to both .npy and .txt for convenience."""
        c = self.get_learned_c()
        np.save(path_prefix + "_c.npy", c)
        with open(path_prefix + "_c.txt", "w") as f:
            f.write(", ".join([f"{v:.8g}" for v in c]))
        print("Saved learned c to:", path_prefix + "_c.npy", "and", path_prefix + "_c.txt")

    ###ADDED THESE NOT SURE####
    
    def get_learned_background(self):
        return float(self.gamma_model.background.detach().cpu().item())

    def save_learned_gamma_params(self, path_prefix):
        c = self.get_learned_c()
        bg = self.get_learned_background()
        np.save(path_prefix + "_c.npy", c)
        with open(path_prefix + "_c.txt", "w") as f:
            f.write(", ".join([f"{v:.8g}" for v in c]))
        with open(path_prefix + "_background.txt", "w") as f:
            f.write(f"{bg:.8g}\n")
        print("Saved learned params to:", path_prefix + "_c.npy/.txt and _background.txt")
