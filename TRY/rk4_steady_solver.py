# rk4_steady_solver.py
import numpy as np

def make_gamma(N=81, Lx=1.0, Ly=1.0,
               k_list=((1,0),(0,1),(1,1)),
               include_cos=True, include_sin=True,
               c=(0.40, -0.20, 0.30, 0.15, 0.10, 0.40),
               background=1.0, min_gamma=1e-3):
    """
    Build gamma(x,y) = background + sum_i [ c_cos_i cos(2π(kx x + ky y)) + c_sin_i sin(2π(...)) ].
    Coefficient ordering: [all cos for k_list..., then all sin for k_list...].
    """
    x = np.linspace(0.0, Lx, N)
    y = np.linspace(0.0, Ly, N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    gamma = np.full((N, N), float(background))
    K = len(k_list)
    c = np.asarray(c, dtype=float)
    assert c.size == (int(include_cos) + int(include_sin)) * K, \
        "Coefficient vector length does not match basis"

    idx = 0
    if include_cos:
        for (kx, ky) in k_list:
            phase = 2*np.pi*(kx*X/Lx + ky*Y/Ly)
            gamma += c[idx] * np.cos(phase)
            idx += 1
    if include_sin:
        for (kx, ky) in k_list:
            phase = 2*np.pi*(kx*X/Lx + ky*Y/Ly)
            gamma += c[idx] * np.sin(phase)
            idx += 1

    # ensure strictly positive
    gamma = np.maximum(gamma, min_gamma)
    return X, Y, gamma

def boundary_modes():
    """Your frequency pairs list (cos/sin per pair)."""
    return [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2),
            (3, 0), (0, 3), (3, 1), (1, 3), (3, 2), (2, 3), (3, 3), (4, 0)]

def build_dirichlet(N, Lx, Ly, mode_index, is_sin):
    """
    Build Dirichlet boundary array f on the full grid for the selected mode.
    f(x,y) = cos(mπx+nπy) or sin(...). Applied only on boundary nodes.
    """
    x = np.linspace(0, Lx, N)
    y = np.linspace(0, Ly, N)
    m, n = boundary_modes()[mode_index]
    uB = np.zeros((N, N), dtype=float)

    def phi(xx, yy):
        arg = m*np.pi*xx + n*np.pi*yy
        return np.sin(arg) if is_sin else np.cos(arg)

    # bottom (i, j=0)
    uB[:, 0] = phi(x, 0.0)
    # top (i, j=N-1)
    uB[:, -1] = phi(x, Ly)
    # left (i=0, j)
    uB[0, :] = phi(0.0, y)
    # right (i=N-1, j)
    uB[-1, :] = phi(Lx, y)

    # corners get assigned multiple times consistently (same formula).
    return uB

def harmonic_mean(a, b, eps=1e-12):
    return 2.0*a*b / np.maximum(a+b, eps)

def L_operator(u, gamma, dx, dy):
    """
    Discrete L(u) = div(gamma * grad u) using flux form and harmonic averaging at faces.
    Neumann is not used; Dirichlet handled by overwriting boundary after each explicit step.
    """
    N = u.shape[0]
    # face gammas
    gE = harmonic_mean(gamma[1:, :], gamma[:-1, :])     # between i-1 and i in x
    gN = harmonic_mean(gamma[:, 1:], gamma[:, :-1])     # between j-1 and j in y

    # fluxes (east-west)
    fx = np.zeros_like(u)
    fy = np.zeros_like(u)

    # internal x-flux differences
    fx[1:-1, :] = ( gE[1:, :] * (u[2:, :] - u[1:-1, :]) - gE[:-1, :] * (u[1:-1, :] - u[:-2, :]) ) / (dx*dx)

    # internal y-flux differences
    fy[:, 1:-1] = ( gN[:, 1:] * (u[:, 2:] - u[:, 1:-1]) - gN[:, :-1] * (u[:, 1:-1] - u[:, :-2]) ) / (dy*dy)

    return fx + fy

def impose_dirichlet(u, uB):
    """Overwrite boundary nodes with prescribed values uB."""
    u[0, :]  = uB[0, :]
    u[-1, :] = uB[-1, :]
    u[:, 0]  = uB[:, 0]
    u[:, -1] = uB[:, -1]
    return u

def rk4_steady(u0, gamma, dx, dy, uB, tol=1e-6, max_steps=5000, cfl=0.24):
    """
    Pseudo-time explicit RK4 integration of u_t = L(u) toward steady state L(u)=0.
    Boundary is kept Dirichlet (u = uB on ∂Ω) after each substage.
    """
    u = u0.copy()
    gmax = float(np.max(gamma))
    dt = cfl * min(dx, dy)**2 / max(gmax, 1e-6)

    for step in range(1, max_steps+1):
        k1 = L_operator(u, gamma, dx, dy)
        u1 = impose_dirichlet(u + 0.5*dt*k1, uB)

        k2 = L_operator(u1, gamma, dx, dy)
        u2 = impose_dirichlet(u + 0.5*dt*k2, uB)

        k3 = L_operator(u2, gamma, dx, dy)
        u3 = impose_dirichlet(u + dt*k3, uB)

        k4 = L_operator(u3, gamma, dx, dy)

        u = impose_dirichlet(u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4), uB)

        # check interior residual
        R = L_operator(u, gamma, dx, dy)
        res = np.max(np.abs(R[1:-1, 1:-1]))
        if res < tol:
            return u, step, res

    return u, max_steps, res
