import os, time, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pinn_framework import FourierFeatureEncoder, ConductivityNetwork
from torch_fdm_differentiable import DifferentiableFDMForwardSolverTorch
from plot_utils import perimeter_order

def smoothness_l2(gamma_grid: torch.Tensor) -> torch.Tensor:
    dx = gamma_grid[:, 1:] - gamma_grid[:, :-1]
    dy = gamma_grid[1:, :] - gamma_grid[:-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()

class InverseDtNTrainer:
    """
    Train ONLY gamma_net using DtN mismatch:
      loss = ||J_pred(gamma,f) - J_meas||^2 + lambda_reg * R(gamma)
    """
    def __init__(self, config: dict, grid_N: int, num_bcs: int, saving_path: str | None = None):
        self.config = config
        self.grid_N = int(grid_N)
        self.num_bcs = int(num_bcs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        self.path = saving_path or (os.getcwd() + "/train_runs_dtn/")
        os.makedirs(self.path, exist_ok=True)
        print("Using device:", self.device)
        print("Saving to:", self.path)

        # grid coords (x,y) for evaluating gamma_net on PDE mesh
        N = self.grid_N
        x = torch.linspace(0.0, 1.0, N, device=self.device, dtype=self.dtype)
        y = torch.linspace(0.0, 1.0, N, device=self.device, dtype=self.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")   # rows=y, cols=x
        xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # (N^2,2)
        self.xy_grid = xy

        # gamma NN
        self.ffe = FourierFeatureEncoder(
            input_dims=2,
            mapping_size=config["ffe_mapping_size"],
            scale=config["ffe_scale"],
            device=self.device
        )

        self.gamma_net = ConductivityNetwork(
            ffe_dims=config["ffe_mapping_size"] * 2,
            layers=config["gamma_net_layers"],
            neurons=config["gamma_net_neurons"],
            actv=config.get("gamma_activation", nn.SiLU()),
            actv_last_layer=config.get("gamma_actv_last_layer", True),
            min_gamma=config.get("min_gamma", 0.5),
            max_gamma=config.get("max_gamma", 2.5),
        ).to(self.device)

        # cache Fourier grid features
        with torch.no_grad():
            self.xy_ffe_grid = self.ffe.encode(self.xy_grid)

        # forward solver
        self.solver = DifferentiableFDMForwardSolverTorch(N=N, device=self.device, dtype=self.dtype)

        self.optimizer = torch.optim.Adam(self.gamma_net.parameters(), lr=config["learning_rate"])

        # =========================
        # NEW: optional LR scheduler (like your old PINN routine)
        # =========================
        self.scheduler = None
        if self.config.get("use_scheduler", False):
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config["lr_decay_gamma"]
            )

        self.loss_history = {"total": [], "data": [], "reg": []}

        # =========================
        # NEW: keep a global step counter (counts optimizer updates)
        # This is helpful if you want scheduler stepping by "step"
        # =========================
        self.global_step = 0

    def gamma_on_grid(self) -> torch.Tensor:
        g = self.gamma_net(self.xy_ffe_grid)  # (N^2,1)
        return g.view(self.grid_N, self.grid_N)

    @torch.no_grad()
    def predict_gamma_numpy(self) -> np.ndarray:
        self.gamma_net.eval()
        return self.gamma_on_grid().detach().cpu().numpy()

    def train(self, dataset, epochs: int | None = None):
        epochs = int(epochs or self.config["epochs"])
        dl = DataLoader(dataset, batch_size=self.config["batch_size_k"], shuffle=True, drop_last=True)

        solver_type = self.config.get("linear_solver", "dense")
        cg_iters = int(self.config.get("cg_max_iter", 60))
        cg_tol = float(self.config.get("cg_tol", 1e-8))
        lam_reg = float(self.config.get("lambda_reg", 0.0))
        log_every = int(self.config.get("log_every", 10))

        # NEW: scheduler stepping mode
        # "epoch" matches your old code exactly
        # "step" is usually better here because you take many optimizer steps per epoch
        lr_unit = self.config.get("lr_decay_unit", "epoch")
        lr_step_every = int(self.config.get("lr_decay_step", 5000))

        if solver_type == "dense":
            M = (self.grid_N - 2) ** 2
            approx_mb = (M * M * 4) / (1024**2)
            print(f"[dense] interior unknowns={M}, dense A ~ {approx_mb:.1f} MB (float32)")
            if self.grid_N > 80:
                print("WARNING: dense solve will be huge for N>80. Consider solver='cg'.")

        print(f"Training {epochs} epochs | batch_k={self.config['batch_size_k']} | solver={solver_type}")

        t0 = time.time()
        for ep in range(1, epochs + 1):
            for batch in dl:
                self.optimizer.zero_grad(set_to_none=True)

                f_bnd = batch["f_bnd"].to(self.device)   # (B,Nb)
                J_meas = batch["J_bnd"].to(self.device)  # (B,Nb)

                gamma_grid = self.gamma_on_grid()        # (N,N)

                _, J_pred = self.solver.predict_currents(
                    gamma_grid, f_bnd,
                    solver=solver_type,
                    cg_max_iter=cg_iters,
                    cg_tol=cg_tol
                )

                loss_data = torch.mean((J_pred - J_meas) ** 2)

                loss_reg = torch.tensor(0.0, device=self.device)
                if lam_reg != 0.0:
                    loss_reg = smoothness_l2(gamma_grid)

                loss = loss_data + lam_reg * loss_reg
                loss.backward()
                self.optimizer.step()
                # NEW: count optimizer updates
                self.global_step += 1
                # =========================
                # NEW: scheduler stepping option
                # =========================
                if self.scheduler is not None and lr_unit == "step":
                    if (self.global_step % lr_step_every) == 0:
                        self.scheduler.step()

                self.loss_history["total"].append(loss.detach().item())
                self.loss_history["data"].append(loss_data.detach().item())
                self.loss_history["reg"].append(loss_reg.detach().item())
            
            # =========================
            # NEW: epoch-based scheduler stepping (matches your old code)
            # =========================
            if self.scheduler is not None and lr_unit == "epoch":
                if (ep % lr_step_every) == 0:
                    self.scheduler.step()

            if ep % log_every == 0:
                print(f"Epoch {ep}/{epochs} | loss={self.loss_history['total'][-1]:.3e} "
                      f"| data={self.loss_history['data'][-1]:.3e} | reg={self.loss_history['reg'][-1]:.3e}")

        print("Done. Time(s):", round(time.time() - t0, 2))

    def save_model(self, path_prefix: str):
        state = {
            "gamma_net_state": copy.deepcopy(self.gamma_net.state_dict()),
            "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler_state": copy.deepcopy(self.scheduler.state_dict()) if self.scheduler is not None else None,
            "global_step": int(self.global_step),
            "config": copy.deepcopy(self.config),
            "loss_history": copy.deepcopy(self.loss_history),
            "ffe_B": self.ffe.B.detach().cpu(),
            "grid_N": self.grid_N,
            "num_bcs": self.num_bcs,
        }
        out = path_prefix + f"_steps_{len(self.loss_history['total'])}.pth"
        torch.save(state, out)
        print("Saved:", out)

        # =========================
    # NEW: load_model restores scheduler too (like your old routine)
    # =========================
    def load_model(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.gamma_net.load_state_dict(ckpt["gamma_net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])

        # restore scheduler if both exist
        if self.scheduler is not None and ckpt.get("scheduler_state", None) is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        elif ckpt.get("scheduler_state", None) is not None and self.scheduler is None:
            print("Warning: checkpoint has scheduler_state but trainer has no scheduler. Set use_scheduler=True.")
        elif self.scheduler is not None and ckpt.get("scheduler_state", None) is None:
            print("Warning: trainer has scheduler but checkpoint has none. Scheduler will start fresh.")

        # restore FFE matrix
        if "ffe_B" in ckpt:
            self.ffe.B = ckpt["ffe_B"].to(self.device)
            self.ffe.B.requires_grad = False

        # IMPORTANT: move optimizer tensors to correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.loss_history = ckpt.get("loss_history", self.loss_history)
        self.global_step = int(ckpt.get("global_step", len(self.loss_history["total"])))

        print("Loaded:", ckpt_path)

        
    # ---------------- Plotting ----------------
    def plot_loss(self, save_fig: bool = False):
        loss = self.loss_history
        fig = plt.figure(figsize=(5,4))
        plt.plot(loss["total"], label="total")
        plt.plot(loss["data"], label="data")
        if max(loss["reg"]) > 0:
            plt.plot(loss["reg"], label="reg")
        plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend(loc="best")
        plt.tight_layout()
        if save_fig:
            fig.savefig(os.path.join(self.path, f"loss_steps_{len(loss['total'])}.pdf"))
        plt.show()

    @torch.no_grad()
    def plot_gamma(self, dataset=None, save_fig: bool = False):
        gamma_hat = self.predict_gamma_numpy()

        has_true = (dataset is not None) and (getattr(dataset, "gamma_true", None) is not None)
        if has_true:
            gamma_true = dataset.gamma_true.detach().cpu().numpy()
            eps = 1e-12
            rel = np.log10(np.abs((gamma_true - gamma_hat) / (gamma_true + eps)))
            mean_re = np.mean(np.abs((gamma_true - gamma_hat) / (gamma_true + eps)))
            fig, ax = plt.subplots(1,3, figsize=(15,4))

            #im0 = ax[0].imshow(gamma_hat, origin="lower", extent=[0,1,0,1])
            #fig.colorbar(im0, ax=ax[0]); ax[0].set_title("Recovered γ")

            #im1 = ax[1].imshow(gamma_true, origin="lower", extent=[0,1,0,1])
            #fig.colorbar(im1, ax=ax[1]); ax[1].set_title("True γ")

            # --- ADD THIS right after you compute gamma_hat and gamma_true ---
            vmin = float(np.minimum(gamma_hat.min(), gamma_true.min()))
            vmax = float(np.maximum(gamma_hat.max(), gamma_true.max()))

            # --- REPLACE your im0/im1/colorbar code with this ---
            im0 = ax[0].imshow(gamma_hat, origin="lower", extent=[0,1,0,1], vmin=vmin, vmax=vmax)
            fig.colorbar(im0, ax=ax[0]); ax[0].set_title("Recovered γ")

            im1 = ax[1].imshow(gamma_true, origin="lower", extent=[0,1,0,1], vmin=vmin, vmax=vmax)
            fig.colorbar(im1, ax=ax[1]); ax[1].set_title("True γ")

            # one shared colorbar for BOTH (so it’s guaranteed the same)
            #fig.colorbar(im0, ax=[ax[0], ax[1]])


            im2 = ax[2].imshow(rel, origin="lower", extent=[0,1,0,1])
            fig.colorbar(im2, ax=ax[2]); ax[2].set_title(f"log10(RE), mean={mean_re*100:.2f}%")
        else:
            fig, ax = plt.subplots(1,1, figsize=(5,4))
            im0 = ax.imshow(gamma_hat, origin="lower", extent=[0,1,0,1])
            fig.colorbar(im0, ax=ax); ax.set_title("Recovered γ")

        plt.tight_layout()
        if save_fig:
            fig.savefig(os.path.join(self.path, f"gamma_steps_{len(self.loss_history['total'])}.pdf"))
        plt.show()

    @torch.no_grad()
    def plot_boundary_fit(self, dataset, k_list=(0,), save_fig: bool = False, show_fig: bool = True):
        coords = dataset.boundary_coords.detach().cpu().numpy()
        order = perimeter_order(coords)
        Nb = coords.shape[0]
        afine = np.arange(Nb)
        seg = int(Nb / 4)

        gamma_grid = self.gamma_on_grid()
        solver_type = self.config.get("linear_solver", "dense")
        cg_iters = int(self.config.get("cg_max_iter", 60))
        cg_tol = float(self.config.get("cg_tol", 1e-8))

        for k in k_list:
            f_k = dataset.boundary_potentials[k].unsqueeze(0).to(self.device)
            J_true = dataset.currents[k].detach().cpu().numpy()
            f_true = dataset.boundary_potentials[k].detach().cpu().numpy()

            _, Jp = self.solver.predict_currents(
                gamma_grid, f_k,
                solver=solver_type,
                cg_max_iter=cg_iters,
                cg_tol=cg_tol
            )
            J_pred = Jp.squeeze(0).detach().cpu().numpy()

            # --- residual ---
            res = J_pred - J_true
            rms = np.sqrt(np.mean(res**2))
            rms_ord = np.sqrt(np.mean(res[order]**2))

            # --- plots ---
            # (1) f_k
            fig, ax = plt.subplots(1,3, figsize=(15,2.8))
            ax[0].plot(afine, f_true[order])
            ax[0].set_title(f"Dirichlet f_k (k={k})")
            ax[0].set_xlabel("perimeter index"); ax[0].set_ylabel("f")

            # (2) J_pred vs J_true
            ax[1].plot(afine, J_pred[order], label="pred")
            ax[1].plot(afine, J_true[order], linestyle="dashed", label="meas/true")
            ax[1].set_title(f"Neumann J_k (k={k})")
            ax[1].set_xlabel("perimeter index"); ax[1].set_ylabel("J")
            ax[1].legend(loc="best")

            # (3) residual
            ax[2].plot(afine, res[order], label=rf"$\Delta J$ (RMS={rms_ord:.2e})")
            ax[2].axhline(0.0, linewidth=0.8)
            ax[2].set_title(r"Residual $J_{\mathrm{pred}}-J_{\mathrm{meas}}$")
            ax[2].set_xlabel("perimeter index")
            ax[2].set_ylabel(r"$\Delta J$")
            ax[2].legend(loc="best")

            # vertical lines separating edges
            for i in range(1,4):
                ax[0].axvline(i*seg, linestyle="--", linewidth=0.5)
                ax[1].axvline(i*seg, linestyle="--", linewidth=0.5)
                ax[2].axvline(i * seg, linestyle="--", linewidth=0.5)

            plt.suptitle(f"k={k} | RMS(all)={rms:.2e} | RMS(ordered)={rms_ord:.2e}")
            plt.tight_layout()
            if save_fig:
                fig.savefig(os.path.join(self.path, f"boundary_fit_k{k}_steps_{len(self.loss_history['total'])}.pdf"))
            if show_fig:
                plt.show()
            else:
                plt.close(fig)

    @torch.no_grad()
    def plot_u_field(self, dataset, k_idx: int = 0, save_fig: bool = False):
        solver_type = self.config.get("linear_solver", "dense")
        cg_iters = int(self.config.get("cg_max_iter", 60))
        cg_tol = float(self.config.get("cg_tol", 1e-8))

        gamma_grid = self.gamma_on_grid()
        f_k = dataset.boundary_potentials[k_idx].unsqueeze(0).to(self.device)
        u_full, _ = self.solver.predict_currents(
            gamma_grid, f_k,
            solver=solver_type,
            cg_max_iter=cg_iters,
            cg_tol=cg_tol
        )
        u_img = u_full.squeeze(0).detach().cpu().numpy()

        fig = plt.figure(figsize=(5,4))
        im = plt.imshow(u_img, origin="lower", extent=[0,1,0,1])
        plt.colorbar(im, label="u")
        plt.title(f"u(x,y) from solve (k={k_idx})")
        plt.xlabel("x"); plt.ylabel("y")
        plt.tight_layout()
        if save_fig:
            fig.savefig(os.path.join(self.path, f"u_field_k{k_idx}_steps_{len(self.loss_history['total'])}.pdf"))
        plt.show()
