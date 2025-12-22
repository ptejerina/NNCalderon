import time
from pathlib import Path
t = time.process_time()

from NNCalderon_rutine_BATCHING_BC import *


###### Choose the cases for the conductivity gamma to be solved, and the folder with the desidered BCs #######

CASES = [
    # 'single_inclusion',
    # 'checkerboard',
    # 'fourierboard',
    'random_blobs',
    # 'random_turbulent'
    ]

NOISE_LEVELS = {
    "0pct": 0.0,
    # "1pct": 0.01,
    # "5pct": 0.05
}

num_BC = 140
BC_type = 'wavelets'
seed = 886

for case in CASES:
    for noise_str, noise_val in NOISE_LEVELS.items():
        # print("\n" + "="*60)
        # # print("RUNNING EXPERIMENT: CASE=" + case + ", NOISE=" + noise_str)
        # print("="*60 + "\n")

        if torch.cuda.is_available():# when running in nyx
            print('cuda is available')
            current_dir = f'{os.getcwd()}' 
        else: # when running locally
            current_dir = Path.cwd().parent  # when running locally
            print(current_dir)

        data_filepath = os.path.join(f"{current_dir}", f"data_{case}_{num_BC}_BC_{BC_type}_seed_{seed}", "dtn_data_" + case + ".npz")
        print(data_filepath)


dataset = CalderonDataset(data_filepath, noise_val)

print('grid size', dataset.grid_N)
print('number of BC functions', dataset.k_indices.shape[0])

#### Choose the architecture and hyperparameters for the pinn training ###

CONFIG = {
    'epochs': 1000,
    'learning_rate': 1e-4,
    'lr_decay_gamma': 0.95,
    'lr_decay_step': 5000,
    'batch_size_pde': 'automatic',  #4096,  # if 'automatic', set to K_batch * Nb * 4 where Nb is num boundary points per BC
    'batch_size_bnd': 1024, # this number is not used in the implementation where BC are batched
    'ffe_mapping_size': 256,
    'ffe_scale': 10.0,
    'gamma_net_layers': 4,
    'gamma_net_neurons': 128,
    'u_net_layers': 6,
    'u_net_neurons': 256,
    'loss_weights': {'pde': 1.0, 'dirichlet_bc': 10.0, 'neumann_bc': 10.0, 'force_true_gamma': 0.0},
    'density_factor': 10.0,
    'center_bounds': (0.25, 0.75),
    'sampling_method': 'uniform',
    'batch_size_bc' : 14,   # e.g. 5–10, NOT 140
}


### Choose saving path: e.g. same path where data is ###
saving_path = os.path.dirname(data_filepath) + f'/test_1_BATCH_BC_results_{case}_{num_BC}_BC_{BC_type}_seed_{seed}_samp_{CONFIG["sampling_method"]}'

### Initialize the Trainer/Solver ###

trainer = Trainer(CONFIG, num_bcs=dataset.num_bcs, synthetic_data=data_filepath, saving_path=saving_path, device=None, \
                  gamma_activ = nn.SiLU(), gamma_actv_last_layer = True, gamma_min=0.5, gamma_max=10.0)



########## (Optional) Load pre-trained model ##########
# loop_index = 686000
# trainer.load_model(path=f'{saving_path}/NN_epochs_{loop_index}')

# trainer.update_optimizer(lr=5e-6)


### Train for the selected amount of epochs  (defaults to CONFIG) ###

train_epochs = 20_000
check_epochs = 20
reps = 1

# train_epochs = 10
# check_epochs = 3
# reps = 1

for i in range(reps):

    if i==0:
        trainer.train(dataset, case, noise_str, train_epochs=check_epochs, disable_progress_bar=False)

    else:      
        # if i==1:   
        #     trainer.update_optimizer(lr=2.5e-6)
   
        trainer.train(dataset, case, noise_str, train_epochs=train_epochs, disable_progress_bar=True)


    ### Save and Visualize results ###
    trainer.save_model(path=f'{saving_path}/NN')

    trainer.plot_loss(save_fig=True)
    plt.close()
    trainer.plot_gamma_nn(save_fig=True)
    plt.close()
    if i % 2 == 0 or i == reps - 1:
        trainer.plot_boundary_data(dataset=dataset, save_fig=[0,10,20,30,40,49], show_fig=False)
        plt.close()
        trainer.plot_induced_potential_u(save_fig=[0,10,20,30,40,49], show_fig=False)
        plt.close()
        trainer.plot_residuals(save_fig=[0,10,20,49], show_fig=False)

###################
elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' hours' )
