# file: plot_calderon_loss.py
import os
import numpy as np

# 1) Paths — adjust CASE or base_dir if yours differ
CASE = "fourier_linear_L1"
base_dir = os.path.expanduser("/home/akalout/NEWAPPROACH3/NEW_ALI_TRIAL_5")
saving_path = os.path.join(base_dir, "train_runs", CASE)
ckpt_path = os.path.join(saving_path, "NN_epochs_90000")
data_filepath = os.path.join(base_dir, f"dtn_data_{CASE}.npz")

# 2) Read num_bcs from the dataset (must match the checkpoint)
if not os.path.exists(data_filepath):
    raise FileNotFoundError(f"Dataset not found: {data_filepath}")
npz = np.load(data_filepath)
num_bcs = int(npz["num_bcs"])

# 3) Minimal CONFIG just to instantiate the networks (values don’t have to match training exactly)
CONFIG = {
    "epochs": 1,
    "learning_rate": 1e-3,
    "lr_decay_gamma": 0.9,
    "lr_decay_step": 999999,   # won't matter for plotting
    "batch_size_pde": 64,
    "batch_size_bnd": 64,

    # u_net
    "ffe_mapping_size": 256,
    "ffe_scale": 10.0,
    "u_net_layers": 6,
    "u_net_neurons": 128,

    # gamma basis (keep same length as in training; values won’t affect loading)
    "gamma_basis": {
        "k_list": [(1,0),(0,1),(1,1),(2,0),(0,2),(2,2),(1,2),(2,1)],
        "Lx": 1.0, "Ly": 1.0, "include_cos": True, "include_sin": True,
    },
    "gamma_coeff_init": [0.0]*16,       # 8 modes × (cos+sin) = 16
    "gamma_positivity": "none",
    "gamma_min": None,
    "gamma_max": None,
    "gamma_background_init": 1.0,
    "gamma_learn_background": True,

    "loss_weights": { "pde": 1.0, "dirichlet_bc": 10.0, "neumann_bc": 1.0, "force_true_gamma": 0.0 }
}

# 4) Import Trainer from your routine and instantiate
from Calderon_routine_2 import Trainer

trainer = Trainer(
    CONFIG,
    num_bcs=num_bcs,
    synthetic_data=data_filepath,
    saving_path=saving_path,   # so the PDF saves next to your runs
)

# 5) Load the checkpoint
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
trainer.load_model(ckpt_path)

# 6) (Optional) ensure no zeros before log-scale plotting
for k in ["total", "pde", "bc", "neumann", "force_true_gamma", "data"]:
    trainer.loss_history[k] = [max(float(v), 1e-12) for v in trainer.loss_history[k]]

# 7) Plot in log scale and save
trainer.plot_loss(save_fig=True, log_scale=True)
print("Saved loss plot to:", os.path.join(saving_path, f"all_losses_epoch_{len(trainer.loss_history['pde'])}_log.pdf"))
