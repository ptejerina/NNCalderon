import os
import matplotlib
matplotlib.use("Agg")
import torch.nn as nn

from dataset_kwise import CalderonBoundaryKDataset
from trainer import InverseDtNTrainer

# --- paths ---
current_dir = os.getcwd()
data_filepath = os.path.join(current_dir, "data_single_inclusion_R3_BC_wavelets_orig", "dtn_data_single_inclusion.npz")

ckpt_path = os.path.join(
    os.path.dirname(data_filepath),
    "train_runs_DtN_full_batch",          # change if your ckpt is elsewhere
    "gammaNN_DtN_steps_1016750.pth"
)

# --- load dataset (for gamma_true) ---
dataset = CalderonBoundaryKDataset(data_filepath, noise_level=0.0)

# --- config must match architecture used in training ---
CONFIG = {
    "epochs": 1,
    "learning_rate": 1e-4,
    "batch_size_k": dataset.num_bcs,
    "full_batch": True,
    "log_every": 10,

    "ffe_mapping_size": 128,
    "ffe_scale": 10.0,
    "gamma_net_layers": 4,
    "gamma_net_neurons": 128,
    "gamma_activation": nn.SiLU(),
    "gamma_actv_last_layer": True,
    "min_gamma": 0.5,
    "max_gamma": 2.5,

    "use_scheduler": False,
    "linear_solver": "dense",
    "cg_max_iter": 80,
    "cg_tol": 1e-8,
    "lambda_reg": 0.0,
}

saving_path = os.path.dirname(ckpt_path)

trainer = InverseDtNTrainer(CONFIG, grid_N=dataset.grid_N, num_bcs=dataset.num_bcs, saving_path=saving_path)
trainer.load_model(ckpt_path)

# now just plot at higher resolution
trainer.plot_gamma_highres(dataset, N_plot=256, save_fig=True)