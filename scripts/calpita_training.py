import numpy as np
from torch import nn
from torchvision.transforms import v2
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch import loggers as pl_loggers
import yaml
import argparse
import calpit
import h5py

# import custom modules
from pita_z.data_modules import data_modules
from pita_z.models import pita_model
from pita_z.models import basic_models
from pita_z.utils import reddening
from pita_z.utils import augmentations 

parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str)
parser.add_argument('run', type=int)
args = parser.parse_args()

config_file = args.config_file
config_dir = '/global/homes/a/ashodkh/calpit/rubin_dp1/configs/'
with open(config_dir + f"{config_file}.yaml", "r") as f:
    config = yaml.safe_load(f)
run = args.run

if __name__ == '__main__':
    ## prepping data
    path_train = config['data']['path_train']
    path_val = config['data']['path_val']

    z_min = config['initial_cde']['z_min']
    z_max = config['initial_cde']['z_max']

    with h5py.File(path_train, 'r') as f_train:
        y_train = f_train['redshifts'][:].astype('float32').clip(z_min,z_max)
    with h5py.File(path_val, 'r') as f_val:
        y_val = f_val['redshifts'][:].astype('float32').clip(z_min,z_max)
        
    initial_cde_type = config['initial_cde']['type']
    if initial_cde_type == 'uniform':
        n_bin_edges = config['initial_cde']['n_bin_edges']
        z_bin_edges = np.linspace(z_min, z_max, n_bin_edges, dtype='float32')
        z_grid = (z_bin_edges[1:] + z_bin_edges[:-1]) / 2
        n_grid = len(z_grid)
        d_z = z_grid[1] - z_grid[0]

        cde_train = np.zeros((y_train.shape[0],n_grid), dtype='float32')
        cde_train[:,:] = 1 / (z_max - z_min)

        cde_val = np.zeros((y_val.shape[0],n_grid), dtype='float32')
        cde_val[:,:] = 1 / (z_max - z_min)

    pit_train = calpit.metrics.probability_integral_transform(
        cde_train,
        z_grid,
        y_train
    )

    pit_val = calpit.metrics.probability_integral_transform(
        cde_val,
        z_grid,
        y_val
    )

    reddening_transform = reddening.ReddeningTransform(R=config['augmentations']['reddening_R'], redden_aug=False)
    if config['augmentations']['gaussian_transform']:
        band_mads = np.load(config['data']['path_band_mads']).astype(np.float32)
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(180, interpolation=v2.InterpolationMode.BILINEAR),
            augmentations.JitterCrop(output_dim=config['augmentations']['crop_dim'], jitter_lim=config['augmentations']['jitter_lim']),
            augmentations.AddGaussianNoise(mean=0, std=band_mads)
        ])
    else:
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(180, interpolation=v2.InterpolationMode.BILINEAR),
            augmentations.JitterCrop(output_dim=config['augmentations']['crop_dim'], jitter_lim=config['augmentations']['jitter_lim']),
        ])
        
    data_module = data_modules.CalpitImagesDataModule(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        path_train=path_train,
        pit_train=pit_train,
        path_val=path_val,
        pit_val=pit_val,
        with_redshift=True,
        with_features=True,
        with_weights=True,
        reddening_transform=reddening_transform,
        load_ebv=True,
        label_f=config['data']['label_f']
    )
     
    ## prepping model

    latent_d = config['model']['latent_d']
    projection_d = config['model']['projection_d']
    encoder = models.convnext_tiny(weights=None)
    encoder._modules["features"][0][0] = nn.Conv2d(config['data']['n_filters'], 96, kernel_size=(4,4), stride=(4,4))
    encoder_mlp = basic_models.MLP(input_dim=1000, hidden_layers=[512], output_dim=latent_d)
    projection_head = basic_models.MLP(input_dim=latent_d, hidden_layers=[128], output_dim=projection_d)
    color_mlp = basic_models.MLP(input_dim=latent_d, hidden_layers=config['model']['color_mlp_hidden_layers'], output_dim=config['data']['n_filters'])    
    redshift_mlp = calpit.nn.models.MLP(
            latent_d+1, # latent vectors + 1 alpha
            config['model']['redshift_mlp_hidden_layers']
        )

    pl_model = pita_model.CalPITALightning(
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
        transforms_z_metric=augmentations.JitterCrop(output_dim=config['augmentations']['crop_dim'], jitter_lim=0),
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
    
    checkpoint_filename = f'candels_{config_file}_run{run}_'+'{epoch}'
    
    checkpoint_callback = ModelCheckpoint(
        #monitor='epoch',
        #mode='max',
        dirpath=config['logging_and_checkpoint']['dir_checkpoint'],
        filename=checkpoint_filename,
        every_n_epochs=config['logging_and_checkpoint']['every_n_epochs'],
        save_top_k=-1,
        enable_version_counter=False
    )
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=config['logging_and_checkpoint']['dir_log'],
        name=f'candels_{config_file}_run{run}'
    )

    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=4,
        min_epochs=1,
        max_epochs=config['training']['epochs'],
        precision='32',
        log_every_n_steps=1,
        default_root_dir="/global/homes/a/ashodkh/calpit/rubin_dp1/scripts",
        strategy='ddp',
        logger=tb_logger,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, lr_monitor_callback],
    )
    trainer.fit(pl_model, data_module)