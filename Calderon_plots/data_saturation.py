# import numpy as np
# import matplotlib.pyplot as plt

# # -----------------------------
# # Data
# # -----------------------------
# K = np.array([50, 90, 140, 190])

# mse_gamma = np.array([
#     4.62687761e-02,
#     7.96469301e-02,
#     2.66038794e-02,
#     3.81260626e-02
# ])

# mean_relative_error = np.array([
#     8.92292595,
#     12.2840424,
#     8.02315903,
#     8.98360920
# ])

# # -----------------------------
# # Matplotlib style
# # -----------------------------
# plt.rcParams.update({
#     "font.size": 13,
#     "axes.labelsize": 15,
#     "axes.titlesize": 15,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     "legend.fontsize": 12,
#     "figure.dpi": 150,
#     "savefig.dpi": 300,
#     "axes.linewidth": 1.2,
#     "lines.linewidth": 2.2,
#     "lines.markersize": 7,
#     "font.family": "serif",
#     "mathtext.fontset": "dejavuserif",
# })

# # -----------------------------
# # Figure: two-panel plot
# # -----------------------------
# fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# # --- MSE plot ---
# axes[0].plot(K, mse_gamma, marker="o")
# axes[0].set_xlabel(r"Number of wavelet boundary conditions, $K$")
# axes[0].set_ylabel(r"$\frac{1}{N}\sum_i(\gamma_i^{\mathrm{NN}}-\gamma_i^{\mathrm{true}})^2$")
# axes[0].set_title(r"Conductivity MSE vs. $K$")
# axes[0].grid(True, alpha=0.3)

# # Mark best point
# best_idx = np.argmin(mse_gamma)
# axes[0].scatter(K[best_idx], mse_gamma[best_idx], s=90, zorder=3)
# axes[0].annotate(
#     rf"best: $K={K[best_idx]}$",
#     xy=(K[best_idx], mse_gamma[best_idx]),
#     xytext=(K[best_idx] + 8, mse_gamma[best_idx] + 0.008),
#     arrowprops=dict(arrowstyle="->", lw=1.0),
# )

# # --- Mean relative error plot ---
# axes[1].plot(K, mean_relative_error, marker="o")
# axes[1].set_xlabel(r"Number of wavelet boundary conditions, $K$")
# axes[1].set_ylabel(r"$\frac{100}{N}\sum_i\left|"
#     r"\frac{\gamma_i^{\mathrm{NN}}-\gamma_i^{\mathrm{true}}}"
#     r"{\gamma_i^{\mathrm{true}}}\right|$")
# axes[1].set_title(r"Relative Error vs. $K$")
# axes[1].grid(True, alpha=0.3)

# # Mark best point
# best_idx_rel = np.argmin(mean_relative_error)
# axes[1].scatter(K[best_idx_rel], mean_relative_error[best_idx_rel], s=90, zorder=3)
# axes[1].annotate(
#     rf"best: $K={K[best_idx_rel]}$",
#     xy=(K[best_idx_rel], mean_relative_error[best_idx_rel]),
#     xytext=(K[best_idx_rel] + 8, mean_relative_error[best_idx_rel] + 0.8),
#     arrowprops=dict(arrowstyle="->", lw=1.0),
# )

# # Shared formatting
# for ax in axes:
#     ax.set_xticks([50, 90, 140, 155, 175, 190])
#     ax.tick_params(direction="in", length=5, width=1.1)
#     ax.spines["top"].set_visible(True)
#     ax.spines["right"].set_visible(True)

# fig.suptitle(
#     r"Boundary-condition ablation study for the single-inclusion case",
#     fontsize=16,
#     y=1.03
# )

# fig.tight_layout()

# # Save figure
# plt.savefig("wavelet_ablation_single_inclusion.png", bbox_inches="tight")
# plt.savefig("wavelet_ablation_single_inclusion.pdf", bbox_inches="tight")

# plt.show()

import numpy as np
import matplotlib.pyplot as plt

K = np.array([50, 90, 140, 190])

mse_gamma = np.array([
    4.62687761e-02,
    7.96469301e-02,
    2.66038794e-02,
    3.81260626e-02
])

mean_relative_error = np.array([
    8.92292595,
    12.2840424,
    8.02315903,
    8.98360920
])

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 1.1,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
})

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

axes[0].plot(K, mse_gamma, marker="o")
axes[0].set_xlabel(r"# of BCs")
axes[0].set_ylabel(
    r"$\frac{1}{N}\sum_i(\gamma_i^{\mathrm{NN}}-\gamma_i^{\mathrm{true}})^2$"
)
axes[0].set_title(r"MSE of $\gamma$")
axes[0].grid(True, alpha=0.3)

axes[1].plot(K, mean_relative_error, marker="o")
axes[1].set_xlabel(r"# of BCs")
axes[1].set_ylabel(
    r"$\frac{100}{N}\sum_i\left|"
    r"\frac{\gamma_i^{\mathrm{NN}}-\gamma_i^{\mathrm{true}}}"
    r"{\gamma_i^{\mathrm{true}}}\right|$"
)
axes[1].set_title(r"Mean relative error of $\gamma$ (%)")
axes[1].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xticks([50, 90, 140, 155, 175, 190])
    ax.tick_params(direction="in", length=5, width=1.0)

plt.savefig("wavelet_ablation_single_inclusion.png", bbox_inches="tight")
plt.savefig("wavelet_ablation_single_inclusion.pdf", bbox_inches="tight")

plt.show()