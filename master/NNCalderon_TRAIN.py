import time
t = time.process_time()

from NNCalderon_rutine import *


###### Choose the cases for the conductivity gamma to be solved, and the folder with the desidered BCs #######

CASES = [
    'single_inclusion',
    # 'checkerboard',
    # 'fourierboard',
    # 'random_blobs',
    # 'random_turbulent'
    ]

NOISE_LEVELS = {
    "0pct": 0.0,
}



### Choose the number of BCs, the type of BCs (must match the BCs in the data folder) ###
num_BC = 140
BC_type = 'wavelets'





### Import the data (DtN pairs at the boundary) ###

for case in CASES:

    current_dir = f'{os.getcwd()}'

    # Path to the data folder
    data_filepath = os.path.join(f"{current_dir}", f"single_inclusion_r_02", "dtn_data_" + case + ".npz")
    print(data_filepath)


dataset = CalderonDataset(data_filepath)

print('grid size', dataset.grid_N)
print('number of BC functions', dataset.k_indices.shape[0])




### Choose the architecture and hyperparameters for the pinn training ###

CONFIG = {
    # Training hyperparameters:
    'epochs': 1000,  # Only used if train_epochs is not specified in the training loop
    'learning_rate': 1e-4,
    'lr_decay_gamma': 0.95,
    'lr_decay_step': 5000,
    'batch_size_pde': 50_000,
    'batch_size_bnd': 12_500,

    # FFE (Fourier Feature Encoding) specifications:
    'use_fourier_features': True,  # whether to use FFE for the input coordinates
    'ffe_mapping_size': 256,  # number of frequency features for the FFE mapping
    'ffe_scale': 10.0,        # standard deviation for the random Gaussian matrix B in FFE.

    # NN architecture specifications:
    'gamma_net_layers': 4,    
    'gamma_net_neurons': 64,
    'u_net_layers': 4,
    'u_net_neurons': 64,

    # Loss hyperparameters:
    'loss_weights': {'pde': 1.0, 'dirichlet_bc': 10.0, 'neumann_bc': 10.0, 'force_true_gamma': 0.0},

    # Sampling method for the PDE points:
    'sampling_method': 'uniform',  # options: 'uniform', 'square_dense_center', 'gaussian_dense_center', or personalized
    'density_factor': 10.0,  # only used if sampling_method is not 'uniform'
    'center_bounds': (0.25, 0.75), # only used if sampling_method is 'square_dense_center' 

}



### Choose saving path  ###
saving_path = os.path.dirname(data_filepath) + f'/test_1_results_{case}_{num_BC}_BC_{BC_type}'



### Initialize the Trainer Class ###

trainer = Trainer(CONFIG, num_bcs=dataset.num_bcs, synthetic_data=data_filepath, saving_path=saving_path, device=None, \
                  gamma_activ = nn.SiLU(), gamma_actv_last_layer = True, gamma_min=0.5, gamma_max=10.0)





########## (Optional) Load pre-trained model ##########
# loop_index = 686000   # epoch number of the pre-trained model to be loaded
# trainer.load_model(path=f'{saving_path}/NN_epochs_{loop_index}')


## Update the lr of the optimizer if needed (optional)
# trainer.update_optimizer(lr=5e-6)


### Train for the selected amount of epochs  (defaults to CONFIG) ###

train_epochs = 10_000
check_epochs = 2
reps = 2



for i in range(reps):

    if i==0:
        trainer.train(dataset, case, train_epochs=check_epochs, disable_progress_bar=False)

    else:      
        # if i==1:   
        #     trainer.update_optimizer(lr=2.5e-6)
   
        trainer.train(dataset, case, train_epochs=train_epochs, disable_progress_bar=False)



    ### Save and Visualize results ###
    trainer.save_model(path=f'{saving_path}/NN')

    trainer.plot_loss(save_fig=True)
    plt.close()
    trainer.plot_gamma_nn(save_fig=True)
    plt.close()

    if i % 2 == 0 or i == reps - 1:
        trainer.plot_boundary_data(dataset=dataset, save_fig=[0,10], show_fig=False)
        plt.close()
        trainer.plot_induced_potential_u(save_fig=[0, 10], show_fig=False)
        plt.close()
        trainer.plot_residuals(save_fig=[0,10,20,49], show_fig=False)

###################


elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' hours' )
