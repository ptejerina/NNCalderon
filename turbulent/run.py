# run.py  (robust DtN inversion runner — keeps YOUR style, adds:
#   (1) boundary ordering auto-fix
#   (2) solver="auto" option for dense/cg selection
#   (3) cluster-safe headless plotting fallback)

import os
import time
import numpy as np

# Cluster-safe: if your sbatch already sets MPLBACKEND=Agg, this is redundant but harmless.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch.nn as nn

from dataset_kwise import CalderonBoundaryKDataset
from trainer import InverseDtNTrainer

t0 = time.process_time()


# =========================
# Boundary ordering alignment
# =========================
def _align_dataset_boundary_to_solver(dataset, trainer, tol=1e-7):
    """
    If dataset.boundary_coords ordering differs from solver.boundary.coords ordering,
    reorder dataset.boundary_coords, dataset.boundary_potentials, dataset.currents in-place.
    """
    coords_data = dataset.boundary_coords.detach().cpu().numpy()
    coords_solver = trainer.solver.boundary.coords.detach().cpu().numpy()

    if coords_data.shape != coords_solver.shape:
        raise RuntimeError(
            f"Boundary coord shapes differ: data={coords_data.shape}, solver={coords_solver.shape}"
        )

    max_diff_direct = float(np.max(np.abs(coords_data - coords_solver)))
    print(f"Max |boundary_coords(data) - boundary_coords(solver)| = {max_diff_direct:.3e}")

    # Already aligned
    if max_diff_direct <= tol:
        print("✅ Boundary ordering already matches solver.")
        return

    # Build permutation perm such that coords_data[perm] == coords_solver
    # Use a robust hash key for float coords on [0,1] grid.
    def keyify(xy):
        return (int(round(float(xy[0]) * 1e9)), int(round(float(xy[1]) * 1e9)))

    lookup = {keyify(coords_data[i]): i for i in range(coords_data.shape[0])}

    try:
        perm = np.array([lookup[keyify(coords_solver[i])] for i in range(coords_solver.shape[0])], dtype=np.int64)
    except KeyError as e:
        raise RuntimeError(
            "Failed to align boundary coords: could not match some solver coords in dataset coords.\n"
            "This usually means coords are not exactly the same set (different discretization / rounding)."
        ) from e

    # Apply permutation consistently
    dataset.boundary_coords = dataset.boundary_coords[perm]
    dataset.boundary_potentials = dataset.boundary_potentials[:, perm]
    dataset.currents = dataset.currents[:, perm]

    # Re-check
    coords_data2 = dataset.boundary_coords.detach().cpu().numpy()
    max_diff2 = float(np.max(np.abs(coords_data2 - coords_solver)))
    print(f"After reordering: Max diff = {max_diff2:.3e}")
    if max_diff2 > tol:
        raise RuntimeError("Boundary alignment attempted but still mismatched (unexpected).")

    print("✅ Dataset boundary ordering aligned to solver.")


# =========================
# Solver selection
# =========================
def _choose_linear_solver(grid_N: int, prefer: str = "auto") -> str:
    prefer = (prefer or "auto").lower()
    if prefer in ("dense", "cg"):
        return prefer
    # auto policy:
    # - dense OK around N<=64 (maybe <=80 depending on memory)
    # - cg for bigger N
    return "dense" if grid_N <= 64 else "cg"


# =========================
# Choose cases + noise
# =========================
CASES = ["random_turbulent"]

NOISE_LEVELS = {
    "0pct": 0.0,
    # "1pct": 0.01,
    # "5pct": 0.05,
}

# =========================
# Training schedule (your style)
# =========================
check_epochs = 50
train_epochs = 5000
reps = 50

plot_every = 2
k_plot_list = [0, 10, 20, 30, 40]  # will be clipped automatically

# Prefer solver: "auto" recommended (dense for 64, cg otherwise)
SOLVER_PREF = "auto"

for case in CASES:
    for noise_str, noise_val in NOISE_LEVELS.items():

        current_dir = os.getcwd()

        # ---- data path (edit folder name if needed) ----
        data_filepath = os.path.join(
            current_dir,
            "data_random_turbulent_140_BC_wavelets_seed_886",
            f"dtn_data_{case}.npz"
        )

        print("\n" + "=" * 60)
        print("RUNNING EXPERIMENT:", "CASE =", case, "| NOISE =", noise_str)
        print("Data:", data_filepath)
        print("=" * 60 + "\n")

        # ---- saving path (your style) ----
        saving_path = os.path.join(os.path.dirname(data_filepath), "train_runs_DtN_full_batch")
        os.makedirs(saving_path, exist_ok=True)

        # ---- dataset ----
        dataset = CalderonBoundaryKDataset(data_filepath, noise_level=noise_val)

        print("grid size:", dataset.grid_N)
        print("number of BC functions (K):", dataset.num_bcs)
        print("boundary coords shape:", tuple(dataset.boundary_coords.shape))

        # Info-only check about dense feasibility
        Nb = int(dataset.boundary_coords.shape[0])
        if Nb == 252:
            print("Boundary points = 252 -> implies grid_N=64 (since 4N-4=252). Dense solve OK.")
        else:
            print("Boundary points Nb =", Nb, "(dense typically only OK around N~64).")

        # Decide solver for this dataset
        linear_solver = _choose_linear_solver(dataset.grid_N, prefer=SOLVER_PREF)
        print(f"Selected linear solver: {linear_solver} (pref={SOLVER_PREF})")

        # =========================
        # CONFIG (DtN)
        # =========================
        CONFIG = {
            # training
            "epochs": 999999,          # not used directly; we pass epochs explicitly
            "learning_rate": 1e-4,
            "batch_size_k": dataset.num_bcs,
            "full_batch": True,
            "log_every": 20,

            # gamma NN
            "ffe_mapping_size": 128,
            "ffe_scale": 10.0,
            "gamma_net_layers": 4,
            "gamma_net_neurons": 128,
            "gamma_activation": nn.SiLU(),
            "gamma_actv_last_layer": True,
            "min_gamma": 0.5,
            "max_gamma": 2.5,

            "use_scheduler": True,  #can use False if i dont want scheduler
            "lr_decay_gamma": 0.95,
            "lr_decay_step": 5000,
            "lr_decay_unit": "epoch",   # NEW: "epoch" matches your old code; "step" to be used if i wanna update per optimizer step (many steps per epoch)

            # differentiable PDE solver
            "linear_solver": linear_solver,  # "dense" or "cg"
            "cg_max_iter": 80,
            "cg_tol": 1e-8,

            # regularization (start 0 for sanity)
            "lambda_reg": 0.0,
        }

        # =========================
        # Initialize Trainer
        # =========================
        trainer = InverseDtNTrainer(
            CONFIG,
            grid_N=dataset.grid_N,
            num_bcs=dataset.num_bcs,
            saving_path=saving_path,
        )

        # =========================
        # Boundary ordering: auto-fix if needed
        # =========================
        _align_dataset_boundary_to_solver(dataset, trainer, tol=1e-7)

        # Clip k list to available range
        k_ok = [k for k in k_plot_list if k < dataset.num_bcs]
        if len(k_ok) == 0:
            k_ok = [0]

        # =========================
        # Train in chunks (your reps/checkpoints style)
        # =========================
        for i in range(reps):

            if i == 0:
                print(f"\n[rep {i+1}/{reps}] sanity training for {check_epochs} epochs")
                trainer.train(dataset, epochs=check_epochs)
            else:
                print(f"\n[rep {i+1}/{reps}] training for {train_epochs} epochs")
                trainer.train(dataset, epochs=train_epochs)

            # ---- Save checkpoint each rep ----
            trainer.save_model(path_prefix=os.path.join(saving_path, "gammaNN_DtN"))

            # ---- Always save loss + gamma plots each rep ----
            plt.close("all")
            trainer.plot_loss(save_fig=True)
            plt.close("all")
            trainer.plot_gamma(dataset, save_fig=True)
            plt.close("all")

            # ---- Plot boundary/u diagnostics every N reps (and last) ----
            do_plot = (i % plot_every == 0) or (i == reps - 1)
            if do_plot:
                trainer.plot_boundary_fit(dataset, k_list=tuple(k_ok), save_fig=True, show_fig=False)
                plt.close("all")

                trainer.plot_u_field(dataset, k_idx=k_ok[0], save_fig=True)
                plt.close("all")

        gamma_hat = trainer.predict_gamma_numpy()
        print("Final gamma_hat shape:", gamma_hat.shape)

elapsed_time = time.process_time() - t0
print("\nTotal elapsed:", elapsed_time / 3600, "hours")
