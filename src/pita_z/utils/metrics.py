import numpy as np
import torch
from scipy.spatial import KDTree
from pathlib import Path
from tqdm import tqdm

def bias_nmad_outliers(y, y_pred, outlier_threshold=0.15, remove_outliers_for_bias=False):
    '''
    Calculates bias, nmad, and outlier fraction.

    Arguments
    ---------
        - y (1D array): true y values.
        - y_pred (1D array): predicted y values.
        - outlier_threshold (float): threshold of what is considered an outlier.
        - remove_outliers_for_bias (bool): Boolean whether or not to remove outliers when calculating bias.

    Returns
    -------
        - bias (float)
        - nmad (float)
        - outlier_fraction (float)
        - nmad_err (float): Estimate of uncertainty on NMAD from size of data.
        - outlier_f_error (float): Estimate of uncertainty on outlier_fraction
    '''
    delta = (y_pred - y)/(1+y)
    
    s_outliers = np.abs(delta) > outlier_threshold
    if remove_outliers_for_bias:
        bias = np.mean(delta[np.logical_not(s_outliers)])
    else:
        bias = np.mean(delta)
    nmad = 1.4826*np.median(np.abs(delta - np.median(delta)))
    outlier_fraction = np.sum(s_outliers)/len(y)
    
    nmad_err = nmad/np.sqrt(2*len(y))
    outlier_f_error = np.sqrt(outlier_fraction*(1-outlier_fraction)/len(y))
    return bias, nmad, outlier_fraction, nmad_err, outlier_f_error

def cde_loss(cdes, y_grid, y_true):
    '''
    Approximates CDE loss (up to a constant) using the predicted CDEs and samples from truth (test set).

    Arguments
    ---------
        - cdes (2D array, (N_samples, grid_dimension)): Estimated CDE values on a grid.
        - y_grid (1D array, grid_dimension): Grid on which CDEs are estimated.
        - y_true (1D array, N_samples): N_samples of y_true values from the true distribution. Assumed to be only 1 sample per data point.

    Returns
    -------
        - CDE loss (float).
    '''
    y_true_idx = np.argmin(np.abs(y_grid.reshape(-1,1) - y_true.reshape(-1,1).T), axis=0)
    
    first_term = []
    second_term = []
    
    for i in range(len(y_true)):
        first_term.append(np.trapezoid(cdes[i,:]**2,y_grid))
        second_term.append(cdes[i,y_true_idx[i]])
    
    return np.average(first_term) - 2 * np.average(second_term)

def ks_from_uniform(pits):
    '''
    Calculates the KS statistic comparing given PIT distribution to a uniform one.

    Arguments
    ---------
        - pits (1D array): PIT values.

    Returns
    -------
        - KS statistic (float)
    '''
    if torch.is_tensor(pits):
        pits = pits.cpu().detach().numpy()

    # calculating the CDF of the pit distribution
    pit_grid = np.linspace(0, 1, 1000)
    cdf = np.zeros(len(pit_grid))
    for i in range(len(pit_grid)):
        cdf[i] = np.sum(pits < pit_grid[i])

    cdf[:] = cdf[:] / len(pits)

    # for a uniform distribution, the cdf value at pit_grid is pit_grid (the cdf is linear).
    idx = np.argmax(np.abs(cdf - pit_grid))
    ks_statistic = (cdf - pit_grid)[idx]
    
    return ks_statistic

def neighbor_metrics(pits, directory, file_name, config_file, features, n_neighbors, overwrite=False):
    """
    Calculate local PIT calibration metrics using nearest neighbors in feature space.
    """

    directory = Path(directory)
    neighbor_ks_path = directory / f"{file_name}_{config_file}_neighbor_ks.npy"
    neighbor_mean_path = directory / f"{file_name}_{config_file}_eighbor_mean.npy"

    if neighbor_ks_path.exists() and neighbor_mean_path.exists() and not overwrite:
        neighbor_pit_ks = np.load(neighbor_ks_path)
        neighbor_pit_mean = np.load(neighbor_mean_path)

    else:
        tree = KDTree(features)

        # Shape: (N, n_neighbors)
        _, indices = tree.query(features, k=n_neighbors)

        # Shape: (N, n_neighbors)
        neighbor_pits = pits[indices]

        neighbor_pit_mean = neighbor_pits.mean(axis=1)

        neighbor_pit_ks = np.array([
            ks_from_uniform(x)
            for x in tqdm(neighbor_pits)
        ])

        np.save(neighbor_ks_path, neighbor_pit_ks)
        np.save(neighbor_mean_path, neighbor_pit_mean)

    return neighbor_pit_ks, neighbor_pit_mean

def mag_binned_metrics(pits, mags, mag_bin_edges):
    mag_bin_centers = (mag_bin_edges[:-1] + mag_bin_edges[1:])/2
    mag_bin_ks = []
    mag_bin_mean = []
    for i in range(len(mag_bin_edges)-1):
        s = (mags > mag_bin_edges[i]) * (mags <= mag_bin_edges[i+1])
        pit_samples = pits[s]
        mag_bin_ks.append(ks_from_uniform(pit_samples))
        mag_bin_mean.append(np.mean(pit_samples))

    return mag_bin_ks, mag_bin_mean