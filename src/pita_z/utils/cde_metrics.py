import numpy as np

def cde_loss(cdes, y_grid, y_true):
    y_true_idx = np.argmin(np.abs(y_grid.reshape(-1,1) - y_true.reshape(-1,1).T), axis=0)
    
    first_term = []
    second_term = []
    
    for i in range(len(y_true)):
        first_term.append(np.trapezoid(cdes[i,:]**2,y_grid))
        second_term.append(cdes[i,y_true_idx[i]])
    
    return np.average(first_term) - 2 * np.average(second_term)

def ks_from_uniform(pits):
    pit_grid = np.linspace(0, 1, 1000)
    cdf = np.zeros(len(pit_grid))
    for i in range(len(pit_grid)):
        cdf[i] = np.sum(pits < pit_grid[i])

    cdf[:] = cdf[:] / len(pits)
    return np.max(np.abs(cdf - pit_grid))