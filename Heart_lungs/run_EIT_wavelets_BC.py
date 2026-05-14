import time
t = time.process_time()

from Calderon_routine_saved_scheduler import *


###### Choose the cases for the conductivity gamma to be solved, and the folder with the desidered BCs #######

# CASES = ["single_inclusion", "multiple_inclusions", "checkerboard"]
# CASES = ["gaussian_inclusion"]
CASES = ["EIT"]

NOISE_LEVELS = {
    "0pct": 0.0,
    # "1pct": 0.01,
    # "5pct": 0.05
}

for case in CASES:
    for noise_str, noise_val in NOISE_LEVELS.items():
        # print("\n" + "="*60)
        # # print("RUNNING EXPERIMENT: CASE=" + case + ", NOISE=" + noise_str)
        # print("="*60 + "\n")

        current_dir = f'{os.getcwd()}'
        
        data_filepath = os.path.join(f"{current_dir}", "data_EIT_BC_wavelets", "dtn_data_" + case + ".npz")
        print(data_filepath)



### Choose saving path: e.g. same path where data is ###
saving_path = os.path.dirname(data_filepath) + '/train_runs_Silu_ffe_20_Dirichlet_100_both'


dataset = CalderonDataset(data_filepath, noise_val)

print('grid size', dataset.grid_N)
print('number of BC functions', dataset.k_indices.shape[0])

#### Choose the architecture and hyperparameters for the pinn training ###

CONFIG = {
    'epochs': 1000,
    'learning_rate': 1e-4,
    'lr_decay_gamma': 0.95,
    'lr_decay_step': 5000,
    'batch_size_pde': 4096,
    'batch_size_bnd': 1024,
    #'use_fourier_features': False,
    'ffe_mapping_size': 256,
    'ffe_scale': 20.0,
    'gamma_net_layers': 4,
    'gamma_net_neurons': 128,
    'u_net_layers': 6,
    'u_net_neurons': 256,
    'loss_weights': {'pde': 1.0, 'dirichlet_bc': 100.0, 'neumann_bc': 10.0, 'force_true_gamma': 0.0},
    'density_factor': 10.0,
    'center_bounds': (0.25, 0.75),
    'sampling_method': 'uniform',

}


### Initialize the Trainer/Solver ###

trainer = Trainer(CONFIG, num_bcs=dataset.num_bcs, synthetic_data=data_filepath, saving_path=saving_path,\
                  gamma_activ = nn.SiLU(), gamma_actv_last_layer = True)


### Train for the selected amount of epochs  (defaults to CONFIG) ###

train_epochs = 10_000
check_epochs = 1_000
reps = 45

# train_epochs = 10
# check_epochs = 3
# reps = 2

for i in range(reps):

    if i==0:
        trainer.train(dataset, case, noise_str, train_epochs=check_epochs, disable_progress_bar=False)

    else:
        trainer.train(dataset, case, noise_str, train_epochs=train_epochs, disable_progress_bar=True)


    ### Save and Visualize results ###
    trainer.save_model(path=f'{saving_path}/NN')

    trainer.plot_loss(save_fig=True)
    plt.close()
    trainer.plot_gamma_nn(save_fig=True)
    plt.close()
    if i % 2 == 0 or i == reps - 1:
        trainer.plot_boundary_data(dataset=dataset, save_fig=[0,2,4,8,10,12], show_fig=False)
        plt.close()
        trainer.plot_boundary_fit(dataset=dataset, k_list=[0, 2,4,8,10,12], save_fig=True, show_fig=False)
        plt.close()
        trainer.plot_induced_potential_u(save_fig=[0,2,4,8,10,12], show_fig=False)
        plt.close()
        trainer.plot_residuals(save_fig=[0,2,4,6], show_fig=False)
    if i ==0:
        trainer.plot_sampling_heatmap(batch_size = 50000, bins=200, save_fig = True)

###################
elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' hours' )
