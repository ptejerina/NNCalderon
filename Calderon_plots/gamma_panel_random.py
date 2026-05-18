# filename: make_reconstruction_panel.py

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.size': 10,
#     'axes.labelsize': 10,
#     'axes.titlesize': 11,
#     'legend.fontsize': 9,
#     'xtick.labelsize': 8,
#     'ytick.labelsize': 8,
#     'mathtext.fontset': 'cm',
# })

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'mathtext.fontset': 'dejavusans'
})

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ============================================================
# 1. Minimal versions of your encoder and gamma network
# ============================================================

class FourierFeatureEncoder:
    def __init__(self, B, device):
        """
        B must be the saved FFE random matrix from the checkpoint.
        Shape: (mapping_size, 2)
        """
        self.B = B.to(device)
        self.B.requires_grad = False
        self.device = device

    def encode(self, x):
        x_proj = (2.0 * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class IdentityEncoder:
    def __init__(self, device):
        self.device = device

    def encode(self, x):
        return x


class ConductivityNetworkFromCheckpoint(nn.Module):
    """
    Builds a gamma_net with the same layer indexing as your original
    ConductivityNetwork, so that load_state_dict works.

    Original structure:
        Linear, SiLU, Linear, SiLU, ..., Linear, Sigmoid
    """
    def __init__(self, gamma_state_dict, actv_last_layer=True,
                 gamma_min=0.2, gamma_max=2.5):
        super().__init__()

        # Find all Linear layers from saved state dict
        weight_keys = sorted(
            [k for k in gamma_state_dict.keys() if k.endswith(".weight")],
            key=lambda s: int(s.split(".")[1])
        )

        linear_shapes = [gamma_state_dict[k].shape for k in weight_keys]

        net_layers = []

        # Hidden layers
        for i, shape in enumerate(linear_shapes):
            out_dim, in_dim = shape

            is_last_linear = (i == len(linear_shapes) - 1)

            net_layers.append(nn.Linear(in_dim, out_dim))

            if not is_last_linear:
                net_layers.append(nn.SiLU())

        if actv_last_layer:
            net_layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*net_layers)

        self.min_gamma = gamma_min
        self.max_gamma = gamma_max

    def forward(self, x_encoded):
        raw_output = self.network(x_encoded)
        return self.min_gamma + (self.max_gamma - self.min_gamma) * raw_output


# ============================================================
# 2. Loading model + prediction utilities
# ============================================================

def load_gamma_model_from_checkpoint(
    checkpoint_path,
    device,
    gamma_min=0.2,
    gamma_max=2.5,
    actv_last_layer=True,
):
    """
    Loads only gamma_net from your saved checkpoint.

    Your checkpoint is expected to contain:
        - gamma_net_state
        - use_fourier_features
        - FFE_rand_matrix, if FFE was used
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    gamma_state = ckpt["gamma_net_state"]

    use_ffe = ckpt.get("use_fourier_features", True)

    gamma_net = ConductivityNetworkFromCheckpoint(
        gamma_state_dict=gamma_state,
        actv_last_layer=actv_last_layer,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    ).to(device)

    gamma_net.load_state_dict(gamma_state)
    gamma_net.eval()

    if use_ffe:
        B = ckpt.get("FFE_rand_matrix", None)
        if B is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} says use_fourier_features=True "
                "but does not contain FFE_rand_matrix. "
                "You need the saved B matrix to reproduce the FFE result."
            )
        encoder = FourierFeatureEncoder(B=B, device=device)
    else:
        encoder = IdentityEncoder(device=device)

    return gamma_net, encoder, ckpt


def predict_gamma_from_checkpoint(
    checkpoint_path,
    N=128,
    device=None,
    gamma_min=0.2,
    gamma_max=2.5,
    actv_last_layer=True,
):
    """
    Predict gamma on an N x N grid.

    This follows your original Trainer.predict_gamma convention:
        xy_grid = torch.stack([yy.flatten(), xx.flatten()], dim=1)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gamma_net, encoder, ckpt = load_gamma_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        actv_last_layer=actv_last_layer,
    )

    x = torch.linspace(0, 1, N, device=device)
    y = torch.linspace(0, 1, N, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    # Same coordinate convention as your original predict_gamma()
    xy_grid = torch.stack([yy.flatten(), xx.flatten()], dim=1)

    with torch.no_grad():
        xy_encoded = encoder.encode(xy_grid)
        gamma_pred = gamma_net(xy_encoded)

    gamma_pred = gamma_pred.reshape(N, N).detach().cpu().numpy()

    return gamma_pred, ckpt


def load_true_gamma(data_path, N=128):
    """
    Loads gamma_true from your .npz file and interpolates it to N x N.

    This follows your original plot_gamma_nn convention.
    """
    data = np.load(data_path)
    gamma_true_grid = data["gamma_true"]

    grid_N = gamma_true_grid.shape[0]
    x_old = np.linspace(0, 1, grid_N)
    y_old = np.linspace(0, 1, grid_N)

    interp = RegularGridInterpolator(
        (x_old, y_old),
        gamma_true_grid,
        bounds_error=False,
        fill_value=None,
    )

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
    gamma_true = interp(xy).reshape(N, N)

    return gamma_true


def mean_relative_error_percent(gamma_pred, gamma_true, eps=1e-12):
    rel = np.abs(gamma_pred - gamma_true) / (np.abs(gamma_true) + eps)
    return 100.0 * np.mean(rel)


# ============================================================
# 3. Main paper-style panel
# ============================================================

def plot_reconstruction_panel(
    experiments,
    output_path,
    N=128,
    figsize_per_row=2.4,
    cmap="viridis",
    device=None,
    use_true_scale=False,
    show_metrics_in_titles=True,
):
    """
    experiments: list of dictionaries. Each dictionary must contain:

        {
            "label": "Single inclusion, $R=0.2$",
            "data": "/path/to/dtn_data_single_inclusion.npz",
            "ffe_ckpt": "/path/to/ffe/NN_epochs_291000",
            "noffe_ckpt": "/path/to/noffe/NN_epochs_291000",
            "gamma_min": 0.5,
            "gamma_max": 2.5,
        }

    gamma_min and gamma_max must match what you used during training.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nrows = len(experiments)
    ncols = 3

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8.2, figsize_per_row * nrows),
        squeeze=False,
    )

    col_titles = [
        r"$\gamma^{\rm true}$",
        r"$\gamma^{\rm NN}_{\rm FFE}$",
        r"$\gamma^{\rm NN}_{\rm no\,FFE}$",
    ]

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=16)

    for i, exp in enumerate(experiments):
        label = exp["label"]
        data_path = exp["data"]
        ffe_ckpt = exp["ffe_ckpt"]
        noffe_ckpt = exp["noffe_ckpt"]

        gamma_min_ffe = exp.get("gamma_min_ffe", exp.get("gamma_min", 0.2))
        gamma_max_ffe = exp.get("gamma_max_ffe", exp.get("gamma_max", 2.5))

        gamma_min_noffe = exp.get("gamma_min_noffe", exp.get("gamma_min", 0.2))
        gamma_max_noffe = exp.get("gamma_max_noffe", exp.get("gamma_max", 2.5))

        actv_last_layer_ffe = exp.get("actv_last_layer_ffe", exp.get("actv_last_layer", True))
        actv_last_layer_noffe = exp.get("actv_last_layer_noffe", exp.get("actv_last_layer", True))

        gamma_true = load_true_gamma(data_path, N=N)

        gamma_ffe, _ = predict_gamma_from_checkpoint(
            ffe_ckpt,
            N=N,
            device=device,
            gamma_min=gamma_min_ffe,
            gamma_max=gamma_max_ffe,
            actv_last_layer=actv_last_layer_ffe,
        )

        gamma_noffe, _ = predict_gamma_from_checkpoint(
            noffe_ckpt,
            N=N,
            device=device,
            gamma_min=gamma_min_noffe,
            gamma_max=gamma_max_noffe,
            actv_last_layer=actv_last_layer_noffe,
        )

        re_ffe = mean_relative_error_percent(gamma_ffe, gamma_true)
        re_noffe = mean_relative_error_percent(gamma_noffe, gamma_true)

        if use_true_scale:
            vmin = np.nanmin(gamma_true)
            vmax = np.nanmax(gamma_true)
        else:
            vmin = min(
                np.nanmin(gamma_true),
                np.nanmin(gamma_ffe),
                np.nanmin(gamma_noffe),
            )
            vmax = max(
                np.nanmax(gamma_true),
                np.nanmax(gamma_ffe),
                np.nanmax(gamma_noffe),
            )

        panels = [gamma_true, gamma_ffe, gamma_noffe]

        for j in range(ncols):
            ax = axes[i, j]

            im = ax.imshow(
                panels[j],
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )

            ax.set_aspect("equal")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])

            if i != nrows - 1:
                ax.set_xticklabels([])

            if j != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(label, fontsize=12, labelpad=25)

            if i == nrows - 1:
                ax.set_xlabel("")

            if show_metrics_in_titles and j == 1:
                ax.set_title(rf"FFE, RE={re_ffe:.2f}\%", fontsize=12)
            elif show_metrics_in_titles and j == 2:
                ax.set_title(rf"No FFE, RE={re_noffe:.2f}\%", fontsize=12)

            # One colorbar per row, attached to last panel
            # divider = make_axes_locatable(axes[i, -1])
            # cax = divider.append_axes("right", size="4%", pad=0.04)
            # cb = fig.colorbar(im, cax=cax)
            # cb.ax.tick_params(labelsize=9)
        divider = make_axes_locatable(axes[i, -1])
        cax = divider.append_axes("right", size="4%", pad=0.04)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=11)
        cb.locator = MaxNLocator(nbins=6)
        cb.update_ticks()

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved panel to: {output_path}")

    plt.show()


# ============================================================
# 4. Optional appendix-style error panel
# ============================================================

# def plot_error_panel(
#     experiments,
#     output_path,
#     N=256,
#     figsize_per_row=2.4,
#     cmap="viridis",
#     device=None,
#     error_vmin=-5,
#     error_vmax=0,
# ):
#     """
#     Creates appendix figure:

#         FFE log10 relative error | no-FFE log10 relative error
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     nrows = len(experiments)
#     ncols = 2

#     fig, axes = plt.subplots(
#         nrows=nrows,
#         ncols=ncols,
#         figsize=(6.2, figsize_per_row * nrows),
#         squeeze=False,
#     )

#     axes[0, 0].set_title(r"$\log_{10}$ relative error, FFE")
#     axes[0, 1].set_title(r"$\log_{10}$ relative error, no FFE")

#     eps = 1e-12

#     for i, exp in enumerate(experiments):
#         label = exp["label"]
#         data_path = exp["data"]
#         ffe_ckpt = exp["ffe_ckpt"]
#         noffe_ckpt = exp["noffe_ckpt"]

#         gamma_min = exp.get("gamma_min", 0.2)
#         gamma_max = exp.get("gamma_max", 2.5)
#         actv_last_layer = exp.get("actv_last_layer", True)

#         gamma_true = load_true_gamma(data_path, N=N)

#         gamma_ffe, _ = predict_gamma_from_checkpoint(
#             ffe_ckpt,
#             N=N,
#             device=device,
#             gamma_min=gamma_min,
#             gamma_max=gamma_max,
#             actv_last_layer=actv_last_layer,
#         )

#         gamma_noffe, _ = predict_gamma_from_checkpoint(
#             noffe_ckpt,
#             N=N,
#             device=device,
#             gamma_min=gamma_min,
#             gamma_max=gamma_max,
#             actv_last_layer=actv_last_layer,
#         )

#         err_ffe = np.log10(np.abs(gamma_ffe - gamma_true) / (np.abs(gamma_true) + eps) + eps)
#         err_noffe = np.log10(np.abs(gamma_noffe - gamma_true) / (np.abs(gamma_true) + eps) + eps)

#         panels = [err_ffe, err_noffe]

#         for j in range(ncols):
#             ax = axes[i, j]

#             im = ax.imshow(
#                 panels[j],
#                 origin="lower",
#                 extent=[0, 1, 0, 1],
#                 cmap=cmap,
#                 vmin=error_vmin,
#                 vmax=error_vmax,
#                 interpolation="nearest",
#             )

#             ax.set_aspect("equal")
#             ax.set_xticks([0, 0.5, 1])
#             ax.set_yticks([0, 0.5, 1])

#             if i != nrows - 1:
#                 ax.set_xticklabels([])

#             if j != 0:
#                 ax.set_yticklabels([])
#             else:
#                 ax.set_ylabel(label, fontsize=12, labelpad=25)

#             if i == nrows - 1:
#                 ax.set_xlabel("")

#         divider = make_axes_locatable(axes[i, -1])
#         cax = divider.append_axes("right", size="4%", pad=0.04)
#         cb = fig.colorbar(im, cax=cax)
#         cb.ax.tick_params(labelsize=12)

#     fig.tight_layout()

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     fig.savefig(output_path, bbox_inches="tight", dpi=300)
#     print(f"Saved error panel to: {output_path}")

#     plt.show()


# ============================================================
# 5. Example usage
# ============================================================

if __name__ == "__main__":

    # Edit these paths.
    # IMPORTANT:
    # gamma_min and gamma_max must match the values used in training.
    # If you changed trainer.gamma_net.min_gamma/max_gamma manually,
    # put the same values here.

    experiments_inclusions = [
        {
            
            "label": r"Blobs_1",
            "data": r"blobs_825/data_random_blobs_BC_wavelets/dtn_data_random_blobs.npz",
            "ffe_ckpt": r"blobs_825/results_ffe/NN_epochs_400110",
            "noffe_ckpt": r"blobs_825/results_no_ffe/NN_epochs_210100",

            "gamma_min_ffe": 0.5,
            "gamma_max_ffe": 10.0,

            "gamma_min_noffe": 0.5,
            "gamma_max_noffe": 10.0,

        },
       
        {
            "label": r"Blobs_2",
            "data": "blobs_886/data_random_blobs_BC_wavelets/dtn_data_random_blobs.npz",
            "ffe_ckpt": "blobs_886/results_ffe/NN_epochs_400200",
            "noffe_ckpt": "blobs_886/results_no_ffe/NN_epochs_210100",
            "gamma_min_ffe": 0.5,
            "gamma_max_ffe": 10.0,

            "gamma_min_noffe": 0.5,
            "gamma_max_noffe": 10.0,
        },
        {
            "label": r"Clouds_1",
            "data": "turbulent_119/data_turbulent_BC_wavelets/dtn_data_random_turbulent.npz",
            "ffe_ckpt": "turbulent_119/results_ffe/NN_epochs_390030",
            "noffe_ckpt": "turbulent_119/results_no_ffe/NN_epochs_240100",
            "gamma_min": 0.5,
            "gamma_max": 10.0,
        },
        {
            "label": r"Clouds_2",
            "data": "turbulent_541/data_turbulent_BC_wavelets/dtn_data_random_turbulent.npz",
            "ffe_ckpt": "turbulent_541/results_ffe/NN_epochs_400210",
            "noffe_ckpt": "turbulent_541/results_no_ffe/NN_epochs_210100",
            "gamma_min": 0.5,
            "gamma_max": 10.0,
        }
    ]

    plot_reconstruction_panel(
        experiments=experiments_inclusions,
        output_path="paper_panels_random/random_reconstructions_fixed_scaling.pdf",
        N=128,
        use_true_scale=False,
        show_metrics_in_titles=False,
    )

    # plot_error_panel(
    #     experiments=experiments_inclusions,
    #     output_path="paper_panels/error_maps_inclusions_ffe_vs_noffe.pdf",
    #     N=128,
    #     error_vmin=-5,
    #     error_vmax=0,
    # )