

# import time, os
# t = time.process_time()

# # ✅ Use the new routine that has the parametric gamma model
# from Calderon_routine import *
# import matplotlib.pyplot as plt

# # ---- Which dataset(s) to train on ----
# # IMPORTANT: 'case' must match the saved filename dtn_data_{case}.npz
# CASES = ["fourier_parametric_6terms"]   # e.g., produced by your parametric generator
# NOISE_LEVELS = {"0pct": 0.0}

# for case in CASES:
#     for noise_str, noise_val in NOISE_LEVELS.items():

#         # ---- Where the dataset lives ----
#         current_dir = os.getcwd()
#         data_dir = os.path.join(current_dir, "Ali_NEW_APPROACH_TRIAL")   # adjust if you used another folder
#         data_filepath = os.path.join(data_dir, f"dtn_data_{case}.npz")
#         if not os.path.exists(data_filepath):
#             raise FileNotFoundError(f"Dataset not found: {data_filepath}")
#         print(f"Using data file: {data_filepath}")

#         # ---- Where to save training outputs ----
#         saving_path = os.path.join(os.path.dirname(data_filepath), "train_runs", case)
#         os.makedirs(saving_path, exist_ok=True)

#         # ---- Dataset (API unchanged) ----
#         dataset = CalderonDataset(data_filepath, noise_val)
#         print('Grid size:', dataset.grid_N)
#         print('Number of BC functions:', dataset.num_bcs)

#         # ---- Trainer config: parametric gamma (no gamma_net_* keys) ----
#         CONFIG = {
#             'epochs': 10,
#             'learning_rate': 1e-3,
#             'lr_decay_gamma': 0.9,
#             'lr_decay_step': 2000,
#             'batch_size_pde': 4096,
#             'batch_size_bnd': 1024,

#             # u_net
#             'ffe_mapping_size': 256,
#             'ffe_scale': 10.0,
#             'u_net_layers': 6,
#             'u_net_neurons': 128,

#             # 🔴 Parametric gamma basis MUST be consistent with how you generated the dataset
#             # For k_list=[(1,0),(0,1),(1,1)] with include_cos/sin=True -> 6 basis functions
#             'gamma_basis': {
#                 'k_list': [(1, 0), (0, 1), (1, 1)],
#                 'Lx': 1.0,
#                 'Ly': 1.0,
#                 'include_cos': True,
#                 'include_sin': True,
#             },
#             # Start c at zeros so the model learns the coefficients
#             'gamma_coeff_init': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#             'gamma_positivity': 'none',     # or 'softplus' / 'sigmoid_bounds'
#             'gamma_min': None,
#             'gamma_max': None,

#             'loss_weights': {
#                 'pde': 1.0,
#                 'dirichlet_bc': 10.0,
#                 'neumann_bc': 10.0,
#                 'force_true_gamma': 0.0,    # set >0 to add supervised γ loss vs. ground truth
#             }
#         }

#         # ---- Initialize trainer ----
#         trainer = Trainer(CONFIG, num_bcs=dataset.num_bcs,
#                           synthetic_data=data_filepath, saving_path=saving_path)

#         # ---- (Optional) continue from a checkpoint if present ----
#         ckpt = os.path.join(saving_path, "NN_epochs_98629")
#         if os.path.exists(ckpt):
#             trainer.load_model(path=ckpt)
#             trainer.update_optimizer()
#         else:
#             print("No pretrained checkpoint found — starting fresh.")

#         # ---- Train ----
#         train_epochs = 5
#         reps = 2

#         for _ in range(reps):
#             trainer.train(dataset, case, noise_str, train_epochs=train_epochs)

#             # ---- Save & plots ----
#             trainer.save_model(path=os.path.join(saving_path, "NN"))

#             trainer.plot_loss(save_fig=True);                plt.close()
#             trainer.plot_gamma_nn(save_fig=True);            plt.close()
#             trainer.plot_boundary_data(dataset, save_fig=True, show_fig=False); plt.close()
#             trainer.plot_induced_potential_u(save_fig=True, show_fig=False);    plt.close()
#             trainer.plot_residuals(save_fig=True, show_fig=False);              plt.close()

#             # (Optional) lower LR between reps
#             trainer.update_optimizer(lr=1e-5)

# elapsed_time = time.process_time() - t
# print(f"{elapsed_time/3600:.2f} hours")




import time, os
t = time.process_time()

# ✅ Use the new routine that has the parametric gamma model
from Calderon_routine import *
import matplotlib.pyplot as plt

# ---- Which dataset(s) to train on ----
# IMPORTANT: 'case' must match the saved filename dtn_data_{case}.npz
CASES = ["fourier_parametric_6terms"]   # e.g., produced by your parametric generator
NOISE_LEVELS = {"0pct": 0.0}

for case in CASES:
    for noise_str, noise_val in NOISE_LEVELS.items():

        # ---- Where the dataset lives ----
        current_dir = os.getcwd()
        data_dir = os.path.join(current_dir, "Ali_NEW_APPROACH_TRIAL")   # adjust if you used another folder
        data_filepath = os.path.join(data_dir, f"dtn_data_{case}.npz")
        if not os.path.exists(data_filepath):
            raise FileNotFoundError(f"Dataset not found: {data_filepath}")
        print(f"Using data file: {data_filepath}")

        # ---- Where to save training outputs ----
        saving_path = os.path.join(os.path.dirname(data_filepath), "train_runs_more_epochs_new", case) #it was train_runs
        os.makedirs(saving_path, exist_ok=True)

        # ---- Dataset (API unchanged) ----
        dataset = CalderonDataset(data_filepath, noise_val)
        print('Grid size:', dataset.grid_N)
        print('Number of BC functions:', dataset.num_bcs)

        # ---- Trainer config: parametric gamma (no gamma_net_* keys) ----
        CONFIG = {
            'epochs': 10,
            'learning_rate': 1e-3,
            'lr_decay_gamma': 0.9,
            'lr_decay_step': 2000,
            'batch_size_pde': 4096,
            'batch_size_bnd': 1024,
            
         

            # u_net
            'ffe_mapping_size': 256,
            'ffe_scale': 10.0,
            'u_net_layers': 6,
            'u_net_neurons': 128,

            # 🔴 Parametric gamma basis MUST be consistent with how you generated the dataset
            # For k_list=[(1,0),(0,1),(1,1)] with include_cos/sin=True -> 6 basis functions
            'gamma_basis': {
                'k_list': [(1, 0), (0, 1), (1, 1)],
                'Lx': 1.0,
                'Ly': 1.0,
                'include_cos': True,
                'include_sin': True,
            },
            # Start c at zeros so the model learns the coefficients
            'gamma_coeff_init': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'gamma_positivity': 'none',     # or 'softplus' / 'sigmoid_bounds'
            'gamma_min': None,
            'gamma_max': None,
            ##THESE TWO NEW NOT SURE##
            'gamma_background_init': 1.0,   # match data generator background
            'gamma_learn_background': True, # allow a small correction during training

            'loss_weights': {
                'pde': 1.0,
                'dirichlet_bc': 10.0,
                'neumann_bc': 1.0,
                'force_true_gamma': 0.0,    # set >0 to add supervised γ loss vs. ground truth
            }
        }

        # ---- Initialize trainer ----
        trainer = Trainer(CONFIG, num_bcs=dataset.num_bcs,
                          synthetic_data=data_filepath, saving_path=saving_path)

        # ---- (Optional) continue from a checkpoint if present ----
        ckpt = os.path.join(saving_path, "NN_epochs_98629")
        if os.path.exists(ckpt):
            trainer.load_model(path=ckpt)
            trainer.update_optimizer()
        else:
            print("No pretrained checkpoint found — starting fresh.")

        # ---- Train ----
        train_epochs = 20000
        reps = 6

        for _ in range(reps):
            trainer.train(dataset, case, noise_str, train_epochs=train_epochs)
            
            tag = f'rep{_+1}_epochs{len(trainer.loss.history['total'])}'
            prefix = os.path.join(saving_path, f"NN_{tag}")
            

            # ---- Save & plots ----
            trainer.save_model(path=os.path.join(saving_path, "NN"))
            
            ###ADDED THESE NOT SURE###
            trainer.save_learned_gamma_params(prefix)
            print("learned background:", trainer.get_learned_background())
            print("learned c:", trainer.get_learned_c())


            trainer.plot_loss(save_fig=True);                plt.close()
            trainer.plot_gamma_nn(save_fig=True);            plt.close()
            trainer.plot_boundary_data(dataset, save_fig=True, show_fig=False); plt.close()
            trainer.plot_induced_potential_u(save_fig=True, show_fig=False);    plt.close()
            trainer.plot_residuals(save_fig=True, show_fig=False);              plt.close()

            # (Optional) lower LR between reps
            #trainer.update_optimizer(lr=1e-5)

elapsed_time = time.process_time() - t
print(f"{elapsed_time/3600:.2f} hours")
























