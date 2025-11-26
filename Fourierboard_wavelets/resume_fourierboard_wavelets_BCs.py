import time
t = time.process_time()

from Calderon_routine import *
import os
import torch.nn as nn
import matplotlib.pyplot as plt

###### Same case + noise as before ######
CASES = ["fourierboard_wavelets"]

NOISE_LEVELS = {
    "0pct": 0.0,
}

for case in CASES:
    for noise_str, noise_val in NOISE_LEVELS.items():

        current_dir = f'{os.getcwd()}'
        data_filepath = os.path.join(
            current_dir,
            "data_fourierboard_BC_wavelets",
            "dtn_data_" + case + ".npz"
        )
        print("Data path:", data_filepath)

        # Same saving path as before
        saving_path = os.path.dirname(data_filepath) + '/train_runs_Silu'
        print("Saving path:", saving_path)

        # Dataset
        dataset = CalderonDataset(data_filepath, noise_val)

        print('grid size', dataset.grid_N)
        print('number of BC functions', dataset.k_indices.shape[0])

        #### Same CONFIG as the first run (you can tweak if you want) ####
        CONFIG = {
            'epochs': 1000,   # not actually used if you pass train_epochs
            'learning_rate': 1e-3,
            'lr_decay_gamma': 0.95,
            'lr_decay_step': 2000,
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
            'center_bounds': (0.2, 0.8),
            'sampling_method': 'uniform',
        }

        # === 1) Build Trainer exactly as before ===
        trainer = Trainer(
            CONFIG,
            num_bcs=dataset.num_bcs,
            synthetic_data=data_filepath,
            saving_path=saving_path,
            gamma_activ=nn.SiLU(),
            gamma_actv_last_layer=True
        )

        # === 2) Load checkpoint NN_epochs_161000 ===
        checkpoint_path = f"{saving_path}/NN_epochs_141000"
        print("Loading checkpoint:", checkpoint_path)
        trainer.load_model(checkpoint_path)
        trainer.update_optimizer(lr=1e-3)  # Optionally adjust learning rate
        
        # Sanity check: how many epochs already trained?
        already_trained = len(trainer.loss_history['total'])
        print("Already trained epochs:", already_trained)

        # === 3) Decide how far you want to go ===
        TARGET_TOTAL_EPOCHS = 700_000
        remaining = TARGET_TOTAL_EPOCHS - already_trained

        if remaining <= 0:
            print("Nothing to do, already at or beyond target epochs.")
        else:
            print("Remaining epochs to train:", remaining)

            # Train in chunks to save/check periodically
            chunk_epochs = 60_000
            reps = remaining// chunk_epochs  #would be 9 in this case
            #extra_epochs = remaining % chunk_epochs

            #print(f"Planning {reps} × {chunk_epochs} + {extra_epochs} epochs")

            for i in range(reps):
                print(f"\n==== Chunk {i+1}/{reps} ====")
                trainer.train(
                    dataset,
                    case,
                    noise_str,
                    train_epochs=chunk_epochs,
                    disable_progress_bar=True
                )

                # Save + plots after each chunk
                trainer.save_model(path=f'{saving_path}/NN')
                trainer.plot_loss(save_fig=True)
                plt.close()
                trainer.plot_gamma_nn(save_fig=True)
                plt.close()

                if i % 2 == 0 or i == reps - 1:
                    trainer.plot_boundary_data(
                        dataset=dataset,
                        save_fig=[0,10,20,30,40,49,50,60,70,80],
                        show_fig=False
                    )
                    plt.close()
                    trainer.plot_induced_potential_u(
                        save_fig=[0,10,20,30,40,49,50,60,70,80],
                        show_fig=False
                    )
                    plt.close()
                    trainer.plot_residuals(
                        save_fig=[0,10,20,49],
                        show_fig=False
                    )

            # if extra_epochs > 0:
            #     print(f"\n==== Final extra chunk: {extra_epochs} epochs ====")
            #     trainer.train(
            #         dataset,
            #         case,
            #         noise_str,
            #         train_epochs=extra_epochs,
            #         disable_progress_bar=True
            #     )
            #     trainer.save_model(path=f'{saving_path}/NN')
            #     trainer.plot_loss(save_fig=True)
            #     plt.close()
            #     trainer.plot_gamma_nn(save_fig=True)
            #     plt.close()

elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' hours' )
