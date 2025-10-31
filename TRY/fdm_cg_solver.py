# fdm_cg_solver.py
import numpy as np

def make_gamma(N=81, Lx=1.0, Ly=1.0,
               k_list=((1,0),(0,1),(1,1)),
               include_cos=True, include_sin=True,
               c=(0.40, -0.20, 0.30, 0.15, 0.10, 0.40),
               background=1.0, min_gamma=1e-3):
    x = np.linspace(0.0, Lx, N)
    y = np.linspace(0.0, Ly, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    gamma = np.full((N, N), float(background))
    K = len(k_list)
    c = np.asarray(c, dtype=float)
    assert c.size == (int(include_cos) + int(include_sin)) * K

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

    gamma = np.maximum(gamma, min_gamma)
    return X, Y, gamma

def boundary_modes():
    return [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2),
            (3, 0), (0, 3), (3, 1), (1, 3), (3, 2), (2, 3), (3, 3), (4, 0)]

def build_dirichlet(N, Lx, Ly, mode_index, is_sin):
    x = np.linspace(0, Lx, N)
    y = np.linspace(0, Ly, N)
    m, n = boundary_modes()[mode_index]
    uB = np.zeros((N, N), dtype=float)

    def phi(xx, yy):
        arg = m*np.pi*xx + n*np.pi*yy
        return np.sin(arg) if is_sin else np.cos(arg)

    uB[:, 0]  = phi(x, 0.0)
    uB[:, -1] = phi(x, Ly)
    uB[0, :]  = phi(0.0, y)
    uB[-1, :] = phi(Lx, y)
    return uB

def harmonic_mean(a, b, eps=1e-12):
    return 2.0*a*b / np.maximum(a+b, eps)

def L_operator(u, gamma, dx, dy):
    N = u.shape[0]
    gE = harmonic_mean(gamma[1:, :], gamma[:-1, :])
    gN = harmonic_mean(gamma[:, 1:], gamma[:, :-1])
    fx = np.zeros_like(u)
    fy = np.zeros_like(u)
    fx[1:-1, :] = ( gE[1:, :] * (u[2:, :] - u[1:-1, :]) - gE[:-1, :] * (u[1:-1, :] - u[:-2, :]) ) / (dx*dx)
    fy[:, 1:-1] = ( gN[:, 1:] * (u[:, 2:] - u[:, 1:-1]) - gN[:, :-1] * (u[:, 1:-1] - u[:, :-2]) ) / (dy*dy)
    return fx + fy

def build_interior_mask(N):
    mask = np.zeros((N, N), dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask

def flatten_interior(u, mask):
    return u[mask]

def assemble_from_interior(int_vec, mask, uB):
    u = uB.copy()
    u[mask] = int_vec
    return u

def cg(matvec, b, x0=None, tol=1e-8, maxiter=5000, M=None):
    """
    Matrix-free CG for SPD operator.
    If M is provided, it's a preconditioner: y = M(r) solves approx A y = r.
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - matvec(x)
    z = M(r) if M is not None else r
    p = z.copy()
    rz_old = np.dot(r, z)
    for it in range(1, maxiter+1):
        Ap = matvec(p)
        alpha = rz_old / max(np.dot(p, Ap), 1e-30)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r, ord=np.inf) < tol:
            return x, it, np.linalg.norm(r, ord=np.inf)
        z = M(r) if M is not None else r
        rz_new = np.dot(r, z)
        beta = rz_new / max(rz_old, 1e-30)
        p = z + beta * p
        rz_old = rz_new
    return x, maxiter, np.linalg.norm(r, ord=np.inf)

def build_A_and_b(gamma, dx, dy, uB):
    """
    Build matvec A(v) using L_operator with zero boundary, and RHS b = -L(uB_only) on interior.
    """
    N = gamma.shape[0]
    mask = build_interior_mask(N)

    # RHS: apply L to boundary-only field (interior zero)
    uB_only = np.zeros_like(uB)
    uB_only[0, :]  = uB[0, :]
    uB_only[-1, :] = uB[-1, :]
    uB_only[:, 0]  = uB[:, 0]
    uB_only[:, -1] = uB[:, -1]
    g = L_operator(uB_only, gamma, dx, dy)
    b = -g[mask]  # interior RHS

    def matvec(v_interior):
        u_tmp = np.zeros_like(uB)  # zero boundary by construction
        u_tmp[mask] = v_interior
        w = L_operator(u_tmp, gamma, dx, dy)
        return w[mask]

    # simple Jacobi preconditioner on the fly (approx diagonal)
    # build diagonal via probing unit vectors on interior grid efficiently:
    diag = np.zeros_like(b)
    # apply A to basis-like stencil at once using a trick: approximate diagonal by
    # local 5-point equivalent with averaged gammas
    # (good enough as a preconditioner; keeps code compact)
    N2 = N
    idxs = np.argwhere(mask)
    gx = harmonic_mean(gamma[1:, :], gamma[:-1, :])
    gy = harmonic_mean(gamma[:, 1:], gamma[:, :-1])
    for k, (i, j) in enumerate(idxs):
        d = 0.0
        # neighbors exist because interior
        d += (gx[i, j] + gx[i-1, j])/(dx*dx)
        d += (gy[i, j] + gy[i, j-1])/(dy*dy)
        diag[k] = d
    diag = np.maximum(diag, 1e-12)

    def M(r):
        return r/diag

    return matvec, b, mask, M
