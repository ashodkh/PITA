import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar

def _calculate_best_point_estimate(pdf_vals: np.array, grid: np.array) -> np.array:
    """
    Compute zx for all objects that minimizes ∫ dz P(z) * Loss(zx,z)

    Parameters
    ----------
    pdf_vals : np.ndarray
        Array of shape (N, k), PDF values for N objects evaluated on z_grid
    z_grid : np.ndarray
        1D array of redshift grid of shape (k,)

    Returns
    -------
    zx_array : np.ndarray
        Array of optimal zx values of shape (N,)
    """

    grid = np.array(grid)

    assert isinstance(grid, np.ndarray)

    N = pdf_vals.shape[0]
    zx_array = np.zeros(N)

    for i in range(N):
        pz = pdf_vals[i]

        def risk(zx: np.array) -> float:
            integrand = pz * loss(zx, grid)
            return simpson(integrand, grid)

        def loss(zx: np.array, grid: np.array, gamma: float = 0.15) -> np.ndarray:
            dz = (zx - grid) / (1 + grid)
            return 1 - 1 / (1 + (dz / gamma) ** 2)

        result = minimize_scalar(risk, bounds=(grid[0], grid[-1]), method="bounded")
        zx_array[i] = result.x

    return zx_array

