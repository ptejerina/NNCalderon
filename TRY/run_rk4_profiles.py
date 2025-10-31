# run_rk4_profiles.py
import numpy as np
import matplotlib.pyplot as plt
from rk4_steady_solver import make_gamma, build_dirichlet, rk4_steady, boundary_modes

# --- PARAMETERS ---
N   = 81         # grid points per direction
Lx  = 1.0
Ly  = 1.0
dx  = Lx/(N-1)
dy  = Ly/(N-1)
K   = 6          # number of boundary excitations (cos/sin of first 3 freq pairs)

# gamma params (your exact spec & coefficients)
k_list = [(1,0),(0,1),(1,1)]
c = [0.40, -0.20, 0.30, 0.15, 0.10, 0.40]  # [cos..., sin...]
X, Y, gamma = make_gamma(N=N, Lx=Lx, Ly=Ly, k_list=k_list,
                         include_cos=True, include_sin=True, c=c,
                         background=1.0, min_gamma=1e-3)

# Prepare figures
cols = 3
rows = int(np.ceil(K/cols))
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.8*rows), constrained_layout=True)
axes = np.atleast_2d(axes)

# Solve per boundary mode
for k in range(K):
    mode_idx = k // 2
    is_sin   = (k % 2 == 1)
    m, n = boundary_modes()[mode_idx]
    uB = build_dirichlet(N, Lx, Ly, mode_idx, is_sin)

    # initial guess = boundary extended (zero interior)
    u0 = np.zeros_like(uB)
    u0 = uB.copy()  # start from boundary values everywhere (quick convergence)

    u, steps, res = rk4_steady(u0, gamma, dx, dy, uB, tol=1e-6, max_steps=6000, cfl=0.24)

    r = k // cols
    cax = k % cols
    ax = axes[r, cax]
    im = ax.imshow(u.T, origin='lower', extent=[0,Lx,0,Ly], aspect='equal')
    ax.set_title(f"RK4 steady u: {'sin' if is_sin else 'cos'}({m}πx+{n}πy)\nsteps={steps}, res={res:.1e}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, shrink=0.85)

# hide empty axes if any
for idx in range(K, rows*cols):
    r = idx // cols
    cax = idx % cols
    axes[r, cax].axis('off')

plt.show()
