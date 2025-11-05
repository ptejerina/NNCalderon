import time
t = time.process_time()

from NNCalderon_rutine import *


###### Choose the cases for the conductivity gamma to be solved, and the folder with the desidered BCs #######

# CASES = ["single_inclusion", "multiple_inclusions", "checkerboard"]
# CASES = ["gaussian_inclusion"]
CASES = ["checkerboard"]

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
        
        data_filepath = os.path.join(f"{current_dir}", "data_checkerboard_dipoles_and_segments_BC_10", "dtn_data_" + case + ".npz")
        print(data_filepath)



### Choose saving path: e.g. same path where data is ###
saving_path = os.path.dirname(data_filepath) + '/test_1'


dataset = CalderonDataset(data_filepath, noise_val)

print('grid size', dataset.grid_N)
print('number of BC functions', dataset.k_indices.shape[0])

#### Choose the architecture and hyperparameters for the pinn training ###

CONFIG = {
    'epochs': 1000,
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
    'loss_weights': {'pde': 1.0, 'dirichlet_bc': 10.0, 'neumann_bc': 10.0, 'force_true_gamma': 0.0} #, 'tv': 1e-4}
}


### Initialize the Trainer/Solver ###

trainer = Trainer(CONFIG, num_bcs=dataset.num_bcs, synthetic_data=data_filepath, saving_path=saving_path)


### Train for the selected amount of epochs  (defaults to CONFIG) ###

train_epochs = 20_000
reps = 5

for _ in range(reps):

    trainer.train(dataset, case, noise_str, train_epochs=train_epochs)


    ### Save and Visualize results ###
    trainer.save_model(path=f'{saving_path}/NN')

    trainer.plot_loss(save_fig=True)
    plt.close()
    trainer.plot_gamma_nn(save_fig=True)
    plt.close()
    trainer.plot_boundary_data(dataset=dataset, save_fig=[0,2,4,6,8], show_fig=False)
    plt.close()
    trainer.plot_induced_potential_u(save_fig=[0,2,4,6,8], show_fig=False)
    plt.close()
    trainer.plot_residuals(save_fig=[0], show_fig=False)

###################
elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' hours' )
