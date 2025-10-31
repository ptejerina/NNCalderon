# run_fdm_profiles.py
import numpy as np
import matplotlib.pyplot as plt
from fdm_cg_solver import (make_gamma, build_dirichlet, build_A_and_b,
                           assemble_from_interior, boundary_modes, cg)

# --- PARAMETERS ---
N   = 81
Lx  = 1.0
Ly  = 1.0
dx  = Lx/(N-1)
dy  = Ly/(N-1)
K   = 6  # number of boundary excitations (cos/sin of first 3 freq pairs)

# gamma params
k_list = [(1,0),(0,1),(1,1)]
c = [0.40, -0.20, 0.30, 0.15, 0.10, 0.40]
X, Y, gamma = make_gamma(N=N, Lx=Lx, Ly=Ly, k_list=k_list,
                         include_cos=True, include_sin=True, c=c,
                         background=1.0, min_gamma=1e-3)

cols = 3
rows = int(np.ceil(K/cols))
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.8*rows), constrained_layout=True)
axes = np.atleast_2d(axes)

for k in range(K):
    mode_idx = k // 2
    is_sin   = (k % 2 == 1)
    m, n = boundary_modes()[mode_idx]
    uB = build_dirichlet(N, Lx, Ly, mode_idx, is_sin)

    # Matrix-free system Au = b
    A, b, mask, M = build_A_and_b(gamma, dx, dy, uB)

    # Conjugate Gradient
    u_int, iters, res = cg(A, b, x0=None, tol=1e-8, maxiter=4000, M=M)  # uses fdm_cg_solver.cg if imported; else paste it here

    # assemble full field
    u = assemble_from_interior(u_int, mask, uB)

    r = k // cols
    cax = k % cols
    ax = axes[r, cax]
    im = ax.imshow(u.T, origin='lower', extent=[0,Lx,0,Ly], aspect='equal')
    ax.set_title(f"FDM+CG u: {'sin' if is_sin else 'cos'}({m}πx+{n}πy)\niters={iters}, res={res:.1e}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, shrink=0.85)

# hide empty axes if any
for idx in range(K, rows*cols):
    r = idx // cols
    cax = idx % cols
    axes[r, cax].axis('off')

plt.show()
