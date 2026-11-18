import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks


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

def _calculate_best_point_estimate_better(pdf_vals, grid, gamma=0.15):
    """
    Find the minimum-risk redshift for each PDF using grid search
    followed by local refinement.
    """
    grid = np.asarray(grid)
    zx_array = np.zeros(len(pdf_vals))

    for i, pz in enumerate(pdf_vals):

        def risk(zx):
            dz = (zx - grid) / (1 + grid)
            loss = 1 - 1 / (1 + (dz / gamma)**2)
            return simpson(pz * loss, x=grid)

        # Evaluate risk over the entire redshift grid
        risk_vals = np.array([risk(z) for z in grid])

        # Identify local minima
        minima, _ = find_peaks(-risk_vals)

        # Include endpoints
        candidates = [
            (risk_vals[0], grid[0]),
            (risk_vals[-1], grid[-1]),
        ]

        # Refine every detected local minimum
        for j in minima:
            result = minimize_scalar(
                risk,
                bounds=(grid[j-1], grid[j+1]),
                method="bounded",
            )
            if result.success:
                candidates.append((result.fun, result.x))

        # Choose the candidate with the lowest risk
        zx_array[i] = min(candidates)[1]

    return zx_array
