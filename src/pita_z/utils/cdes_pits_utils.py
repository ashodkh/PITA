import numpy as np
from scipy.interpolate import PchipInterpolator
import torch
from tqdm import tqdm

def pit_transform(cdfs, grid, samples):
    '''
    Calculate the PITs of samples using the provided CDFs (assuming different CDF for each sample).

    Arguments
    ---------
        - CDFs (2D array, (N_samples,grid_dimension)): CDFs calculated on a grid.
        - grid (1D array, grid_dimension): Grid on which CDFs are calculated.
        - samples (1D array, N_samples): Samples from PDFs that are transformed to PIT variables.

    Returns
    -------
        - PITs (1D array)
    '''

    n_samples = len(samples)
    pits = np.zeros(n_samples)
    for i in range(n_samples):
        cdf = cdfs[i]
        pits[i] = PchipInterpolator(grid, cdf, extrapolate=True)(samples[i])

    return pits

def predict_calpit_cdes_and_cdfs(model, dataset, cde_init, z_grid, alpha_grid, batch_size):
    '''
    Predict calibrated conditional density estimates (CDEs) and cumulative distribution functions (CDFs) in batches.
    
    Arguments
    ---------
        - model (torch.nn.Module): Trained model used to predict CDFs and transform the initial CDEs.
        - dataset (2D array, (N_samples, N_features)): Input features for each sample. Could also be an h5py dataset.
        - cde_init (2D array, (N_samples, grid_dimension)): Initial CDEs evaluated on the redshift grid.
        - z_grid (1D array, grid_dimension): Redshift grid on which the CDEs are evaluated.
        - alpha_grid (1D array or torch.Tensor, grid_dimension): Alpha grid used to evaluate the predicted CDFs.
        - batch_size (int): Number of samples processed in each batch.
    
    Returns
    -------
        - cde_pred (2D array, (N_samples, grid_dimension)): Predicted calibrated CDEs.
        - cdf_pred (2D array, (N_samples, grid_dimension)): Predicted calibrated CDFs.
    '''
    assert len(alpha_grid) == len(z_grid)
    
    n_total = len(dataset)
    n_grid = len(z_grid)

    cde_pred = np.zeros((n_total, n_grid))
    cdf_pred = np.zeros((n_total, n_grid))

    device = model.device
    
    if not isinstance(alpha_grid, torch.Tensor):
        alpha_grid = torch.tensor(alpha_grid, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in tqdm(range(0, n_total, batch_size)):
            i_final = min(i + batch_size, n_total)
            batch_features = torch.tensor(dataset[i:i_final], dtype=torch.float32, device=device)
            batch_features_alpha = torch.cat(
                [
                    torch.repeat_interleave(alpha_grid, i_final-i)[:,None],
                    torch.tile(batch_features, (n_grid,1))
                ],
                dim=-1
            )
            cde_init_batch = torch.tensor(cde_init[i:i_final], dtype=torch.float32, device=device)
            cdf_pred[i:i_final,:] = model.forward(alphas=None, x=batch_features_alpha).cpu().detach().numpy().squeeze().reshape((len(alpha_grid),i_final-i)).T
            cde_pred[i:i_final,:] = model.transform(batch_features,cde_init_batch).cpu().detach().numpy().squeeze()

    return cde_pred, cdf_pred
    
def predict_calpita_cdes_and_cdfs(model, datasets, cde_init, z_grid, alpha_grid, batch_size, transforms, ebvs, R, latent_d, flux_masks):
    '''
    Predict calibrated conditional density estimates (CDEs) and cumulative distribution functions (CDFs) in batches.
    
    Arguments
    ---------
        - model (torch.nn.Module): Trained model used to predict CDFs and transform the initial CDEs.
        - datasets (h5py dataset): Input dataset which contains images and flux masks. Images have shape (N_samples, N_bands, px, py).
        - cde_init (2D array, (N_samples, grid_dimension)): Initial CDEs evaluated on the redshift grid.
        - z_grid (1D array, grid_dimension): Redshift grid on which the CDEs are evaluated.
        - alpha_grid (1D array or torch.Tensor, grid_dimension): Alpha grid used to evaluate the predicted CDFs.
        - batch_size (int): Number of samples processed in each batch.
        - transforms: transforms to apply to the images.
        - ebvs (1D array, N_samples): E(B-V) values for deredenning.
        - R (1D array): R value for each band.
        - latent_d (int): Dimensionality of latent space.
        - flux_masks (bool): whether to use flux masking maps.
    
    Returns
    -------
        - cde_pred (2D array, (N_samples, grid_dimension)): Predicted calibrated CDEs.
        - cdf_pred (2D array, (N_samples, grid_dimension)): Predicted calibrated CDFs.
    '''
    assert len(alpha_grid) == len(z_grid)

    images = datasets["images"]
    n_total = len(images)
    n_grid = len(z_grid)

    cde_pred = np.zeros((n_total, n_grid))
    cdf_pred = np.zeros((n_total, n_grid))
    latent_vectors = np.zeros((n_total, latent_d))
    
    device = model.device
    
    if not isinstance(alpha_grid, torch.Tensor):
        alpha_grid = torch.tensor(alpha_grid, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in tqdm(range(0, n_total, batch_size)):
            i_final = min(i + batch_size, n_total)
            batch_images = transforms(images[i:i_final])
            batch_flux_masks = transforms(datasets["flux_masks"][i:i_final])
            true_ext = ebvs[i:i_final][:,None] * R[None,:]
            dr_images = torch.tensor(batch_images * (10.**(true_ext[:,:,None,None]/2.5)), dtype=torch.float32, device=device)
            if flux_masks:
                dr_images = torch.cat(
                    (
                        dr_images,
                        torch.tensor(batch_flux_masks, dtype=torch.float32, device=device)
                    ),
                    axis=1
                )

            cde_pred[i:i_final,:] = model.transform_cde(dr_images).cpu().detach().numpy().squeeze()
            latent_vectors[i:i_final,:] = model.encoder_mlp(model.encoder(dr_images)).cpu().detach().numpy().squeeze()
            batch_features_alpha = torch.cat(
                [
                    torch.repeat_interleave(alpha_grid, i_final-i)[:,None],
                    torch.tile(torch.tensor(latent_vectors[i:i_final,:], dtype=torch.float32, device=device), (len(alpha_grid),1))
                ],
                dim=-1
            )
            cdf_pred[i:i_final,:] = torch.sigmoid(model.redshift_mlp(batch_features_alpha)).cpu().detach().numpy().squeeze().reshape((len(alpha_grid),i_final-i)).T
                
    return latent_vectors, cde_pred, cdf_pred
    

