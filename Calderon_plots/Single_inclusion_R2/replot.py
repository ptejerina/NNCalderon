from Calderon_routine_saved_scheduler import *
import os
import torch.nn as nn
import matplotlib.pyplot as plt

case = "single_inclusion"
noise_str = "0pct"
noise_val = 0.0

current_dir = os.getcwd()
data_filepath = os.path.join(
    current_dir,
    "data_single_inclusion_R2_BC_wavelets_orig",
    f"dtn_data_{case}.npz"
)

saving_path = os.path.join(
    current_dir,'data_single_inclusion_R2_BC_wavelets_orig', 'neww')


# saving_path = os.path.join(
#     os.path.dirname(data_filepath),
#     "train_runs_Silu_rescaled_better"
# )

checkpoint_path = os.path.join(current_dir, "data_single_inclusion_R2_BC_wavelets_orig", "NN_epochs_537000")

print("Data:", data_filepath)
print("Checkpoint:", checkpoint_path)
print("Saving to:", saving_path)

dataset = CalderonDataset(data_filepath, noise_val)

CONFIG = {
    'epochs': 1000,
    'learning_rate': 1e-4,
    'lr_decay_gamma': 0.95,
    'lr_decay_step': 5000,
    'batch_size_pde': 4096,
    'batch_size_bnd': 1024,
    'ffe_mapping_size': 256,
    'ffe_scale': 10.0,
    'gamma_net_layers': 4,
    'gamma_net_neurons': 128,
    'u_net_layers': 6,
    'u_net_neurons': 256,
    'loss_weights': {
        'pde': 1.0,
        'dirichlet_bc': 10.0,
        'neumann_bc': 10.0,
        'force_true_gamma': 0.0
    },
    'density_factor': 10.0,
    'center_bounds': (0.25, 0.75),
    'sampling_method': 'uniform',
}

trainer = Trainer(
    CONFIG,
    num_bcs=dataset.num_bcs,
    synthetic_data=data_filepath,
    saving_path=saving_path,
    gamma_activ=nn.SiLU(),
    gamma_actv_last_layer=True
)

trainer.load_model(checkpoint_path)

# Replot only
# trainer.plot_loss(save_fig=True, show_fig=False)
trainer.plot_gamma_nn(save_fig=True)
plt.close()

# trainer.plot_boundary_fit(
#     dataset=dataset,
#     k_list=[0, 10, 20, 30, 40, 49],
#     save_fig=True,
#     show_fig=False
# )
# trainer.plot_boundary_data(dataset=dataset, save_fig=[0,10,20,30,40,49], show_fig=False)

print("Replotting done.")
