import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------
# Settings
# ---------------------------------------------------

checkpoint_path = "NN_epochs_321000"

# First epoch to display.
# For example:
#   20  -> ignore epochs 1 to 19
#   100 -> ignore epochs 1 to 99
#   1000 -> ignore epochs 1 to 999
start_epoch = 100

output_path = (
    f"all_losses_epoch_321000_from_epoch_{start_epoch}.pdf"
)


# ---------------------------------------------------
# Load checkpoint
# ---------------------------------------------------

if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

weights = checkpoint["loss_weights"]

total_loss = np.asarray(
    checkpoint["train_loss"],
    dtype=float
)

pde_loss = (
    weights["pde"]
    * np.asarray(checkpoint["de_loss"], dtype=float)
)

dirichlet_loss = (
    weights["dirichlet_bc"]
    * np.asarray(checkpoint["dirichlet_loss"], dtype=float)
)

neumann_loss = (
    weights["neumann_bc"]
    * np.asarray(checkpoint["neumann_loss"], dtype=float)
)


# ---------------------------------------------------
# Truncate the displayed histories
# ---------------------------------------------------

if start_epoch < 1:
    raise ValueError("start_epoch must be at least 1.")

if start_epoch > len(total_loss):
    raise ValueError(
        f"start_epoch={start_epoch} is larger than the "
        f"available number of epochs: {len(total_loss)}"
    )

# Python uses zero-based indexing
start_index = start_epoch - 1

total_plot = total_loss[start_index:]
pde_plot = pde_loss[start_index:]
dirichlet_plot = dirichlet_loss[start_index:]
neumann_plot = neumann_loss[start_index:]

# Preserve the real epoch numbers
epochs = np.arange(start_epoch, len(total_loss) + 1)
epochs_thousands = epochs / 1000.0


# ---------------------------------------------------
# Plot
# ---------------------------------------------------

fig, ax = plt.subplots(figsize=(7.2, 4.2))

ax.plot(
    epochs_thousands,
    total_plot,
    label="Total",
    linewidth=2.0
)

ax.plot(
    epochs_thousands,
    pde_plot,
    label="PDE",
    linewidth=1.5
)

ax.plot(
    epochs_thousands,
    dirichlet_plot,
    label=r"Dirichlet $f_k$",
    linewidth=1.5
)

ax.plot(
    epochs_thousands,
    neumann_plot,
    label=r"Neumann $J_k$",
    linewidth=1.5
)

ax.set_yscale("log")
ax.set_xlabel(r"Epoch ($\times 10^3$)")
ax.set_ylabel("Weighted loss")
ax.set_title("Training losses")

ax.grid(
    True,
    which="both",
    alpha=0.25
)

ax.legend(
    loc="upper right",
    frameon=True,
    fontsize=9
)

fig.tight_layout()

fig.savefig(
    output_path,
    bbox_inches="tight"
)

output_pdf = f"all_losses_from_epoch_{start_epoch}.pdf"

fig.savefig(
    output_pdf,
    format="pdf",
    bbox_inches="tight"
)

print(f"Saved PDF as: {os.path.abspath(output_pdf)}")

plt.show()

print(f"Loaded {len(total_loss)} epochs.")
print(f"Displayed epochs {start_epoch} to {len(total_loss)}.")
print(f"Saved plot to: {os.path.abspath(output_path)}")

print(
    f"Minimum displayed PDE loss: "
    f"{np.min(pde_plot):.6e}"
)
print(
    f"Minimum displayed Dirichlet loss: "
    f"{np.min(dirichlet_plot):.6e}"
)
print(
    f"Minimum displayed Neumann loss: "
    f"{np.min(neumann_plot):.6e}"
)