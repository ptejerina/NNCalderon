import time
t = time.process_time()

from Calderon_routine_saved_scheduler import *

CASES = ["single_inclusion"]

NOISE_LEVELS = {
    "0pct": 0.0,
}

for case in CASES:
    for noise_str, noise_val in NOISE_LEVELS.items():

        current_dir = f"{os.getcwd()}"

        data_filepath = os.path.join(
            current_dir,
            "data_single_inclusion_first_50",
            "dtn_data_" + case + ".npz"
        )

        # Folder where the trained NN checkpoints are stored
        checkpoint_dir = os.path.dirname(data_filepath) + "/train_runs_Silu"

        # Folder where you want to save gamma error txt files
        gamma_error_dir = os.path.dirname(data_filepath) + "/gamma_errors_64_grid"

        if not os.path.isdir(gamma_error_dir):
            os.makedirs(gamma_error_dir)

        dataset = CalderonDataset(data_filepath, noise_val)

        CONFIG = {
            'epochs': 1000,
            'learning_rate': 1e-4,
            'lr_decay_gamma': 0.95,
            'lr_decay_step': 5000,
            'batch_size_pde': 4096,
            'batch_size_bnd': 1024,

            # IMPORTANT:
            # This must match the model that was trained.
            # If NN_epochs_291000 was trained with False, keep False.
            # If it was trained with Fourier features, use True.
            #'use_fourier_features': True,

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

            # This controls where plot_gamma_nn saves the txt/pdf outputs
            saving_path=gamma_error_dir,

            gamma_activ=nn.SiLU(),
            gamma_actv_last_layer=True
        )

        checkpoint_epoch = 291000

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"NN_epochs_{checkpoint_epoch}"
        )

        print("Loading checkpoint from:")
        print(checkpoint_path)

        trainer.load_model(checkpoint_path)

        # This will save:
        # gamma_errors_epoch_291000.txt
        # inside gamma_error_dir
        trainer.plot_gamma_nn(N=64, save_fig=True)

elapsed_time = time.process_time() - t
print(str(elapsed_time / 3600) + " hours")