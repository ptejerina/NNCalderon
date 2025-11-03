

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
from Calderon_routine_2 import *
import matplotlib.pyplot as plt

# ---- Which dataset(s) to train on ----
# IMPORTANT: 'case' must match the saved filename dtn_data_{case}.npz
CASES = ["fourier_linear_L1_new"]   # e.g., produced by your parametric generator
NOISE_LEVELS = {"0pct": 0.0}

for case in CASES:
    for noise_str, noise_val in NOISE_LEVELS.items():

        # ---- Where the dataset lives ----
        current_dir = os.getcwd()
        data_dir = os.path.join(current_dir, "NEW_ALI_TRIAL_6")   # adjust if you used another folder
        data_filepath = os.path.join(data_dir, f"dtn_data_{case}.npz")
        if not os.path.exists(data_filepath):
            raise FileNotFoundError(f"Dataset not found: {data_filepath}")
        print(f"Using data file: {data_filepath}")

        # ---- Where to save training outputs ----
        saving_path = os.path.join(os.path.dirname(data_filepath), "train_runs", case) #it was train_runs
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
                'k_list': [(1,0),(0,1),(1,1),(2,0),(0,2),(2,2),(1,2),(2,1), (3,0),(0,3),(3,1),(1,3),(3,2),(2,3),(3,3),(4,0)],
                'Lx': 1.0,
                'Ly': 1.0,
                'include_cos': True,
                'include_sin': True,
            },
            # Start c at zeros so the model learns the coefficients
            'gamma_coeff_init': np.random.uniform(-1,1,32).tolist(), #or 32 zeros [0.0]*32
            'gamma_positivity': 'none',     # or 'softplus' / 'sigmoid_bounds'
            'gamma_min': None,
            'gamma_max': None,
            ##THESE TWO NEW NOT SURE##
            'gamma_background_init': 1.0,   # match data generator background
            'gamma_learn_background': True, # allow a small correction during training

            'loss_weights': {
                'pde': 2.0,
                'dirichlet_bc': 8.0,
                'neumann_bc': 2.0,
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

        # for _ in range(reps):
        #     trainer.train(dataset, case, noise_str, train_epochs=train_epochs)

        #     # ---- Save & plots ----
        #     trainer.save_model(path=os.path.join(saving_path, "NN"))
            
        #     ###ADDED THESE NOT SURE###
        #     trainer.save_learned_gamma_params(os.path.join(saving_path, "NN"))
        #     print("learned background:", trainer.get_learned_background())
        #     print("learned c:", trainer.get_learned_c())


        #     trainer.plot_loss(save_fig=True);                plt.close()
        #     trainer.plot_gamma_nn(save_fig=True);            plt.close()
        #     trainer.plot_boundary_data(dataset, save_fig=True, show_fig=False); plt.close()
        #     trainer.plot_induced_potential_u(save_fig=True, show_fig=False);    plt.close()
        #     trainer.plot_residuals(save_fig=True, show_fig=False);              plt.close()

            # (Optional) lower LR between reps
            #trainer.update_optimizer(lr=1e-5)

#THIS TO SAVE C WHILE TRAINING:
        for rep_idx in range(1, reps + 1):
            trainer.train(dataset, case, noise_str, train_epochs=train_epochs)

            # How many epochs have been trained in total so far:
            epochs_done = len(trainer.loss_history['total'])

            # ---- Save model (your current save_model already appends _epochs_<count>)
            trainer.save_model(path=os.path.join(saving_path, "NN"))

            # ---- Save *unique* c and background for this rep
            prefix = os.path.join(saving_path, f"NN_epochs_{epochs_done}_rep_{rep_idx}")
            trainer.save_learned_gamma_params(prefix)

            print("learned background:", trainer.get_learned_background())
            print("learned c:", trainer.get_learned_c())

            #ADDED THIS TO COMPARE GAMMAS:
            # ---- Compute and log gamma error
            errs = trainer.gamma_error(N=256)   # choose N
            print(f"[rep {rep_idx}] gamma MAE={errs['mae']:.3e}, MSE={errs['mse']:.3e}, relL2={errs['rel_l2']:.3e}")

            # Append to CSV
            metric_path = os.path.join(saving_path, "metrics_gamma.csv")
            header_needed = not os.path.exists(metric_path)
            with open(metric_path, "a") as f:
                if header_needed:
                    f.write("epochs,rep,mae,mse,rel_l2\n")
                f.write(f"{epochs_done},{rep_idx},{errs['mae']},{errs['mse']},{errs['rel_l2']}\n")

            # ---- Plots (optionally also make them unique)
            trainer.plot_loss(save_fig=True);                plt.close()
            trainer.plot_gamma_nn(save_fig=True);            plt.close()
            trainer.plot_boundary_data(dataset, save_fig=True, show_fig=False); plt.close()
            trainer.plot_induced_potential_u(save_fig=True, show_fig=False);    plt.close()
            trainer.plot_residuals(save_fig=True, show_fig=False);              plt.close()


elapsed_time = time.process_time() - t
print(f"{elapsed_time/3600:.2f} hours")
























