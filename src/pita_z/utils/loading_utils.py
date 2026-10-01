import numpy as np
import h5py
from pita_z.models import fully_supervised_model, basic_models
import calpit
import torchvision.models as models
from torchvision.transforms import v2
from torch import nn
from pita_z.models import pita_model

def load_data(directory, file_names, data_type, dataset_names):
    '''
    Loads hdf5 photometric catalog given directory and data type (train, test, val). File_names is a dictionary
    that relates each data type to its exact file name.

    Arguments
    ---------
        - directory (str): directory which contains the files.
        - file_names (dict): dictionary that maps (train, test, val) to the file names of each.
        - data_type (str): specifies one of (train, test, val).
        - dataset_names (list[str]): specifies which datasets to lazy-load.

    Returns
    -------
        - h5py file
        - feature dataset
        - redshift dataset
        - fluxes dataset
    '''

    file_path = directory + file_names[data_type]

    f = h5py.File(file_path, 'r')
    datasets = {name: f[name] for name in dataset_names}
    return f, datasets

def load_calpit_photometry_model(config, checkpoint_path, z_grid):
    '''
    Loads a CalpitPhotometryLightning model given a config file, checkpoint path, and z_grid.

    Arguments
    ---------
        - config (dict): dictionary of hyperparameters used during training.
        - checkpoint_path (str): path to pytorch lightning checkpoint.
        - z_grid (1D arra): redshift grid on which the model was trained.

    Returns
    -------
        - Pytorch Lightning model.
    '''
    
    if config['model']['type'] == 'MLP':
        mlp = calpit.nn.models.MLP(
            config['data']['n_features'] + 1, # photometric features + 1 alpha
            config['model']['hidden_layers']
        )
    elif config['model']['type'] == 'MonotonicMLP':
        lipschitz_mlp = basic_models.LipschitzMLP(
            config['data']['n_features'] + 1, # photometric features + 1 alpha
            config['model']['hidden_layers']
        )
        mlp = basic_models.MonotonicMLP(
            lipschitz_mlp,
            monotonic_constraints=[1,0,0,0,0],
            lipschitz_const=config['model']['lipschitz_const']
        )
    elif config['model']['type'] == 'UMNN':
        mlp = calpit.nn.umnn.MonotonicNN(
            config['data']['n_features'] + 1,
            config['model']['hidden_layers'],
            sigmoid=True
        )

    model = fully_supervised_model.CalpitPhotometryLightning.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=mlp,
        lr=config["training"]["learning_rate"],
        lr_scheduler=None,
        alpha_grid=np.linspace(0, 1, 201, dtype="float32"),
        y_grid=z_grid.astype("float32"),
        cde_init_type="uniform"
    )

    return model

def load_calpita_model(config, checkpoint_path, z_grid, transforms):
    '''
    Loads a CalPITALightning model given a config file, checkpoint_path, and z_grid.

    Arguments
    ---------
        - config (dict): dictionary of hyperparameters used during training.
        - checkpoint_path (str): path to pytorch lightning checkpoint.
        - z_grid (1D arra): redshift grid on which the model was trained.
        - transforms: transforms applied on the images.

    Returns
    -------
        - Pytorch Lightning model.
    '''

    latent_d = config['model']['latent_d']
    projection_d = config['model']['projection_d']
    encoder = models.convnext_base(weights=None)
    encoder._modules["features"][0][0] = nn.Conv2d(config['data']['n_filters'], 128, kernel_size=(4,4), stride=(4,4))
    encoder_mlp = basic_models.MLP(input_dim=1000, hidden_layers=[512], output_dim=latent_d)
    projection_head = basic_models.MLP(input_dim=latent_d, hidden_layers=[128], output_dim=projection_d)
    color_mlp = basic_models.MLP(input_dim=latent_d, hidden_layers=config['model']['color_mlp_hidden_layers'], output_dim=config['data']['n_filters'])    
    if config['model']['type'] == 'MLP':
        redshift_mlp = calpit.nn.models.MLP(
                latent_d+1, # 4 photometric fluxes + 1 alpha
                config['model']['redshift_mlp_hidden_layers'],
                sigmoid=False
            )
    elif config['model']['type'] == 'UMNN':
        redshift_mlp = calpit.nn.umnn.MonotonicNN(
            latent_d+1,
            config['model']['redshift_mlp_hidden_layers'],
            sigmoid=False
        )

    model = pita_model.CalPITALightning.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        encoder=encoder,
        encoder_mlp=encoder_mlp,
        projection_head=projection_head,
        redshift_mlp=redshift_mlp,
        color_mlp=color_mlp,
        loss_type=config['training']['loss_type'],
        alpha_grid=np.linspace(0.001, 0.999, config['training']['n_alphas'], dtype='float32'),
        y_grid=z_grid.astype('float32'),
        cde_init_type=config['data']['cde_init_type'],
        transforms=transforms,
        momentum=config['training']['momentum'],
        queue_size=config['model']['queue_size'],
        temperature=config['model']['temperature'],
        cl_loss_weight=config['training']['cl_loss_weight'],
        redshift_loss_weight=config['training']['redshift_loss_weight'],
        color_loss_weight=config['training']['color_loss_weight'],
        lr=config['training']['learning_rate'],
        lr_scheduler=config['training']['lr_scheduler']['type'],
        cosine_T_max=config['training']['lr_scheduler']['cosine']['T_max'],
        cosine_eta_min=config['training']['lr_scheduler']['cosine']['eta_min']
    )  
        
    return model