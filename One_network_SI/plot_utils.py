import numpy as np

def perimeter_order(coords: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Perimeter order: bottom(x↑) -> right(y↑) -> top(x↓) -> left(y↓)
    Uses [:-1] to avoid duplicated corners; returns exactly Nb indices.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    bottom = np.where(np.abs(y - 0.0) < tol)[0]
    top    = np.where(np.abs(y - 1.0) < tol)[0]
    left   = np.where(np.abs(x - 0.0) < tol)[0]
    right  = np.where(np.abs(x - 1.0) < tol)[0]

    bottom = bottom[np.argsort(x[bottom])]
    right  = right[np.argsort(y[right])]
    top    = top[np.argsort(x[top])][::-1]
    left   = left[np.argsort(y[left])][::-1]

    return np.concatenate([bottom[:-1], right[:-1], top[:-1], left[:-1]])
