import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch import loggers as pl_loggers
import yaml
import argparse
import calpit
import h5py

# import custom modules
from pita_z.data_modules import data_modules
from pita_z.models import fully_supervised_model

parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str)
parser.add_argument('run', type=int)
args = parser.parse_args()

config_file = args.config_file
config_dir = '/global/homes/a/ashodkh/calpit/configs/'
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
    
    data_module = data_modules.CalpitPhotometryDataModule(
        path_train=path_train,
        feature_name=config['data']['feature_name'],
        pit_train=pit_train,
        scaler_path=config['data']['scaler_path'],
        path_val=path_val,
        pit_val=pit_val,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
     
    ## prepping model
    if config['model']['type'] == 'MLP':
        model = calpit.nn.models.MLP(
            5, # 4 photometric fluxes + 1 alpha
            config['model']['hidden_layers']
        )

    pl_model = fully_supervised_model.CalpitPhotometryLightning(
        model=model,
        lr=config['training']['learning_rate'],
        lr_scheduler=None, #config['training']['lr_scheduler']
        alpha_grid=np.linspace(0.001, 0.999, config['training']['n_alphas'], dtype='float32'),
        y_grid=z_grid.astype('float32'),
        cde_init_type='uniform'
    )
    
    checkpoint_filename = f'candels_{config_file}_run{run}_'+'{epoch}'
    
    checkpoint_callback = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=config['logging_and_checkpoint']['dir_checkpoint'],
        filename=checkpoint_filename,
        every_n_epochs=config['logging_and_checkpoint']['every_n_epochs'],
        save_top_k=20,
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
        default_root_dir="/global/homes/a/ashodkh/calpit/scripts",
        strategy='ddp',
        logger=tb_logger,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, lr_monitor_callback],
    )
    trainer.fit(pl_model, data_module)