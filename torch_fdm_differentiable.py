import torch
from dataclasses import dataclass
from typing import Tuple, Literal

def _harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (2.0 * a * b) / (a + b + eps)

@dataclass
class BoundaryInfo:
    indices_1d: torch.Tensor   # (Nb,) flattened i*N + j (sorted asc)
    ij: torch.Tensor           # (Nb,2) integer (i,j)
    coords: torch.Tensor       # (Nb,2) float (x,y)
    normals: torch.Tensor      # (Nb,2) float (nx,ny)

class DifferentiableFDMForwardSolverTorch:
    """
    Differentiable FD solver for ∇·(γ∇u)=0 on [0,1]^2 with Dirichlet boundary f.
    Two solver modes:
      - dense: torch.linalg.solve (best for N~64)
      - cg:    unrolled differentiable CG (for bigger N)
    """
    def __init__(self, N: int, device="cpu", dtype=torch.float32):
        self.N = int(N)
        self.device = torch.device(device)
        self.dtype = dtype
        self.h = 1.0 / (self.N - 1)
        self.boundary = self._build_boundary_info()

    def _build_boundary_info(self) -> BoundaryInfo:
        N = self.N
        x = torch.linspace(0.0, 1.0, N, device=self.device, dtype=self.dtype)
        y = torch.linspace(0.0, 1.0, N, device=self.device, dtype=self.dtype)

        indices, ijs, coords, normals = [], [], [], []
        for i in range(N):
            for j in range(N):
                is_bnd = False
                nx, ny = 0.0, 0.0
                if i == 0:      is_bnd = True; ny += -1.0
                if i == N - 1:  is_bnd = True; ny +=  1.0
                if j == 0:      is_bnd = True; nx += -1.0
                if j == N - 1:  is_bnd = True; nx +=  1.0
                if is_bnd:
                    idx = i * N + j
                    nlen = (nx*nx + ny*ny) ** 0.5
                    indices.append(idx)
                    ijs.append((i, j))
                    coords.append((float(x[j]), float(y[i])))
                    normals.append((nx / nlen, ny / nlen))

        indices_t = torch.tensor(indices, device=self.device, dtype=torch.long)
        unique_indices = torch.unique(indices_t, sorted=True)

        first_pos = {}
        for pos, idx in enumerate(indices):
            if idx not in first_pos:
                first_pos[idx] = pos
        order = [first_pos[int(idx.item())] for idx in unique_indices]

        ij_t = torch.tensor([ijs[p] for p in order], device=self.device, dtype=torch.long)
        coords_t = torch.tensor([coords[p] for p in order], device=self.device, dtype=self.dtype)
        normals_t = torch.tensor([normals[p] for p in order], device=self.device, dtype=self.dtype)

        return BoundaryInfo(unique_indices, ij_t, coords_t, normals_t)

    # ---------- matrix-free interior operator ----------
    def _apply_A_interior(self, u_int: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        N = self.N
        h2 = self.h * self.h
        if u_int.dim() == 2:
            u_int = u_int.unsqueeze(0)

        B = u_int.shape[0]
        u_full = torch.zeros((B, N, N), device=u_int.device, dtype=u_int.dtype)
        u_full[:, 1:-1, 1:-1] = u_int

        uc = u_full[:, 1:-1, 1:-1]
        uE = u_full[:, 1:-1, 2:]
        uW = u_full[:, 1:-1, :-2]
        uN = u_full[:, 2:, 1:-1]
        uS = u_full[:, :-2, 1:-1]

        gC = gamma[1:-1, 1:-1]
        gE = _harmonic_mean(gC, gamma[1:-1, 2:]).unsqueeze(0)
        gW = _harmonic_mean(gC, gamma[1:-1, :-2]).unsqueeze(0)
        gN = _harmonic_mean(gC, gamma[2:, 1:-1]).unsqueeze(0)
        gS = _harmonic_mean(gC, gamma[:-2, 1:-1]).unsqueeze(0)

        return (-(gE+gW+gN+gS)*uc + gE*uE + gW*uW + gN*uN + gS*uS) / h2

    def _rhs_from_dirichlet(self, f_full: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        N = self.N
        h2 = self.h * self.h
        if f_full.dim() == 2:
            f_full = f_full.unsqueeze(0)
        B = f_full.shape[0]

        u_full = torch.zeros((B, N, N), device=f_full.device, dtype=f_full.dtype)
        u_full[:, 0, :]  = f_full[:, 0, :]
        u_full[:, -1, :] = f_full[:, -1, :]
        u_full[:, :, 0]  = f_full[:, :, 0]
        u_full[:, :, -1] = f_full[:, :, -1]

        uc = u_full[:, 1:-1, 1:-1]
        uE = u_full[:, 1:-1, 2:]
        uW = u_full[:, 1:-1, :-2]
        uN = u_full[:, 2:, 1:-1]
        uS = u_full[:, :-2, 1:-1]

        gC = gamma[1:-1, 1:-1]
        gE = _harmonic_mean(gC, gamma[1:-1, 2:]).unsqueeze(0)
        gW = _harmonic_mean(gC, gamma[1:-1, :-2]).unsqueeze(0)
        gN = _harmonic_mean(gC, gamma[2:, 1:-1]).unsqueeze(0)
        gS = _harmonic_mean(gC, gamma[:-2, 1:-1]).unsqueeze(0)

        res = (-(gE+gW+gN+gS)*uc + gE*uE + gW*uW + gN*uN + gS*uS) / h2
        return -res

    # ---------- differentiable unrolled CG ----------
    @staticmethod
    def cg_solve_batched(A_fn, b: torch.Tensor, max_iter: int = 60, tol: float = 1e-8) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b - A_fn(x)
        p = r.clone()

        def dot(a, c):
            return (a*c).flatten(1).sum(dim=1, keepdim=True)

        rs_old = dot(r, r)
        for _ in range(max_iter):
            Ap = A_fn(p)
            alpha = rs_old / (dot(p, Ap) + 1e-12)
            a = alpha.view(-1, 1, 1)
            x = x + a * p
            r = r - a * Ap
            rs_new = dot(r, r)
            mask = (rs_new > (tol**2)).to(b.dtype)
            beta = (rs_new / (rs_old + 1e-12)) * mask
            bta = beta.view(-1, 1, 1)
            p = r + bta * p
            rs_old = rs_new
        return x

    def solve_u_full_cg(self, gamma: torch.Tensor, f_bnd: torch.Tensor,
                        cg_max_iter: int = 60, cg_tol: float = 1e-8) -> torch.Tensor:
        N = self.N
        Nb = self.boundary.indices_1d.numel()
        assert f_bnd.dim() == 2 and f_bnd.shape[1] == Nb
        B = f_bnd.shape[0]
        ij = self.boundary.ij

        f_full = torch.zeros((B, N, N), device=gamma.device, dtype=gamma.dtype)
        f_full[:, ij[:, 0], ij[:, 1]] = f_bnd

        b = -self._rhs_from_dirichlet(f_full, gamma)

        def A_fn(x): return -self._apply_A_interior(x, gamma)
        u_int = self.cg_solve_batched(A_fn, b, max_iter=cg_max_iter, tol=cg_tol)

        u_full = torch.zeros((B, N, N), device=gamma.device, dtype=gamma.dtype)
        u_full[:, 1:-1, 1:-1] = u_int
        u_full[:, ij[:, 0], ij[:, 1]] = f_bnd
        return u_full

    # ---------- dense assembly + direct solve ----------
    def _assemble_A_dense(self, gamma: torch.Tensor) -> torch.Tensor:
        N = self.N
        nx = N - 2
        ny = N - 2
        M = nx * ny
        h2 = self.h * self.h

        gC = gamma[1:-1, 1:-1]
        gE = _harmonic_mean(gC, gamma[1:-1, 2:])
        gW = _harmonic_mean(gC, gamma[1:-1, :-2])
        gN = _harmonic_mean(gC, gamma[2:, 1:-1])
        gS = _harmonic_mean(gC, gamma[:-2, 1:-1])

        diag = (-(gE + gW + gN + gS) / h2).reshape(-1)
        vE   = (gE / h2).reshape(-1)
        vW   = (gW / h2).reshape(-1)
        vN   = (gN / h2).reshape(-1)
        vS   = (gS / h2).reshape(-1)

        A = torch.zeros((M, M), device=gamma.device, dtype=gamma.dtype)
        p = torch.arange(M, device=gamma.device)
        col = p % nx
        row = p // nx

        A[p, p] = diag
        east  = col < (nx - 1)
        west  = col > 0
        north = row < (ny - 1)
        south = row > 0

        A[p[east],  (p + 1)[east]]   = vE[east]
        A[p[west],  (p - 1)[west]]   = vW[west]
        A[p[north], (p + nx)[north]] = vN[north]
        A[p[south], (p - nx)[south]] = vS[south]
        return A

    def solve_u_full_dense(self, gamma: torch.Tensor, f_bnd: torch.Tensor) -> torch.Tensor:
        N = self.N
        Nb = self.boundary.indices_1d.numel()
        assert f_bnd.dim() == 2 and f_bnd.shape[1] == Nb
        B = f_bnd.shape[0]
        ij = self.boundary.ij

        f_full = torch.zeros((B, N, N), device=gamma.device, dtype=gamma.dtype)
        f_full[:, ij[:, 0], ij[:, 1]] = f_bnd

        b = self._rhs_from_dirichlet(f_full, gamma)  # (B,ny,nx)
        A = self._assemble_A_dense(gamma)            # (M,M)

        nx = N - 2
        ny = N - 2
        M = nx * ny
        b_flat = b.reshape(B, M)

        u_flat = torch.linalg.solve(A, b_flat.T).T
        u_int = u_flat.reshape(B, ny, nx)

        u_full = torch.zeros((B, N, N), device=gamma.device, dtype=gamma.dtype)
        u_full[:, 1:-1, 1:-1] = u_int
        u_full[:, ij[:, 0], ij[:, 1]] = f_bnd
        return u_full

    # ---------- boundary current ----------
    def boundary_current(self, u_full: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        N = self.N
        h = self.h
        if u_full.dim() == 2:
            u_full = u_full.unsqueeze(0)
        B = u_full.shape[0]
        Nb = self.boundary.indices_1d.numel()

        ij = self.boundary.ij
        normals = self.boundary.normals
        i = ij[:, 0]
        j = ij[:, 1]

        def u_at(ii, jj): return u_full[:, ii, jj]

        du_dx = torch.zeros((B, Nb), device=u_full.device, dtype=u_full.dtype)
        du_dy = torch.zeros((B, Nb), device=u_full.device, dtype=u_full.dtype)

        left = (j == 0)
        right = (j == N-1)
        midx = ~(left | right)
        if left.any():
            ii, jj = i[left], j[left]
            du_dx[:, left] = (-3*u_at(ii,jj) + 4*u_at(ii,jj+1) - u_at(ii,jj+2)) / (2*h)
        if right.any():
            ii, jj = i[right], j[right]
            du_dx[:, right] = (3*u_at(ii,jj) - 4*u_at(ii,jj-1) + u_at(ii,jj-2)) / (2*h)
        if midx.any():
            ii, jj = i[midx], j[midx]
            du_dx[:, midx] = (u_at(ii,jj+1) - u_at(ii,jj-1)) / (2*h)

        bottom = (i == 0)
        top = (i == N-1)
        midy = ~(bottom | top)
        if bottom.any():
            ii, jj = i[bottom], j[bottom]
            du_dy[:, bottom] = (-3*u_at(ii,jj) + 4*u_at(ii+1,jj) - u_at(ii+2,jj)) / (2*h)
        if top.any():
            ii, jj = i[top], j[top]
            du_dy[:, top] = (3*u_at(ii,jj) - 4*u_at(ii-1,jj) + u_at(ii-2,jj)) / (2*h)
        if midy.any():
            ii, jj = i[midy], j[midy]
            du_dy[:, midy] = (u_at(ii+1,jj) - u_at(ii-1,jj)) / (2*h)

        nx = normals[:, 0].view(1, Nb)
        ny = normals[:, 1].view(1, Nb)
        du_dn = du_dx * nx + du_dy * ny

        gamma_bnd = gamma[i, j].view(1, Nb)
        return gamma_bnd * du_dn

    def predict_currents(
        self,
        gamma: torch.Tensor,
        f_bnd: torch.Tensor,
        solver: Literal["dense", "cg"] = "dense",
        cg_max_iter: int = 60,
        cg_tol: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if solver == "dense":
            u_full = self.solve_u_full_dense(gamma, f_bnd)
        else:
            u_full = self.solve_u_full_cg(gamma, f_bnd, cg_max_iter=cg_max_iter, cg_tol=cg_tol)
        J = self.boundary_current(u_full, gamma)
        return u_full, J
