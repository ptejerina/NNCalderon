import torch
import os

path = "results_no_ffe\\NN_epochs_231000"
ckpt = torch.load(path, map_location="cpu")

total_hist = ckpt["train_loss"]
pde_hist = ckpt["de_loss"]
dirichlet_hist = ckpt["dirichlet_loss"]
neumann_hist = ckpt["neumann_loss"]
weights = ckpt["loss_weights"]

total_last = total_hist[-1]
pde_last = pde_hist[-1]
dirichlet_last = dirichlet_hist[-1]
neumann_last = neumann_hist[-1]

weighted_pde = weights["pde"] * pde_last
weighted_dirichlet = weights["dirichlet_bc"] * dirichlet_last
weighted_neumann = weights["neumann_bc"] * neumann_last

reconstructed_total = weighted_pde + weighted_dirichlet + weighted_neumann

# txt output path
txt_path = os.path.join(os.path.dirname(path), "losses_epoch_231000.txt")

with open(txt_path, "w") as f:
    f.write("Loss values for NN_epochs_231000\n")
    f.write("=" * 40 + "\n\n")

    f.write(f"Epoch: {len(total_hist)}\n\n")

    f.write("Saved raw losses:\n")
    f.write(f"total_loss_saved: {total_last}\n")
    f.write(f"pde_loss_raw: {pde_last}\n")
    f.write(f"dirichlet_loss_raw: {dirichlet_last}\n")
    f.write(f"neumann_loss_raw: {neumann_last}\n\n")

    f.write("Loss weights used in training:\n")
    f.write(f"pde_weight: {weights['pde']}\n")
    f.write(f"dirichlet_weight: {weights['dirichlet_bc']}\n")
    f.write(f"neumann_weight: {weights['neumann_bc']}\n")
    f.write(f"force_true_gamma_weight: {weights['force_true_gamma']}\n\n")

    f.write("Weighted contributions:\n")
    f.write(f"pde_loss_weighted: {weighted_pde}\n")
    f.write(f"dirichlet_loss_weighted: {weighted_dirichlet}\n")
    f.write(f"neumann_loss_weighted: {weighted_neumann}\n\n")

    f.write("Check:\n")
    f.write(f"reconstructed_total_from_components: {reconstructed_total}\n")
    f.write(f"saved_total_loss: {total_last}\n")

print(f"Saved losses to: {txt_path}")