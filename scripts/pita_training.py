import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch import loggers as pl_loggers
import yaml
import argparse

# import custom modules
from pita_z.models import pita_model
from pita_z.models import basic_models
from pita_z.data_modules import data_modules
from pita_z.utils import reddening
from pita_z.utils import augmentations 

parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str)
parser.add_argument('run', type=int)
args = parser.parse_args()

config_file = args.config_file
config_dir = '/global/homes/a/ashodkh/git_repos/PITA/configs/'
with open(config_dir + f"{config_file}.yaml", "r") as f:
    config = yaml.safe_load(f)

run = args.run

if __name__ == '__main__':
    ## prepping data
    reddening_transform = reddening.ReddeningTransform(R=config['augmentations']['reddening_R'], redden_aug=False)
    if config['augmentations']['gaussian_transform']:
        band_mads = np.load(config['data']['path_band_mads'])
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
            transform_utils.JitterCrop(output_dim=64, jitter_lim=4),
        ])
    data_module = data_modules.ImagesDataModule(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_size=0.8,
        path_train=config['data']['path_train'],
        path_val=config['data']['path_val'],
        with_redshift=True,
        with_features=True,
        with_weights=True,
        reddening_transform=reddening_transform,
        load_ebv=True,
    )

    ## prepping model
    
    latent_d = config['model']['latent_d']
    projection_d = config['model']['projection_d']

    # encoder_type = 'my_encoder'
    # joint_blocks = basic_models.JointBlocks(
    #                                 input_channels=32,
    #                                 block_channels=[32,64],
    #                                 avg_pooling_layers=[4,4]
    # )
    # encoder = basic_models.Encoder(
    #                             input_channels=n_filters,
    #                             first_layer_output_channels=32,
    #                             joint_blocks=joint_blocks
    # )
    
    # encoder_type = 'convnext'    
    # encoder = basic_models.CustomConvNeXt(n_filters)

    encoder = models.convnext_tiny(weights=None)
    encoder._modules["features"][0][0] = nn.Conv2d(config['data']['n_filters'], 96, kernel_size=(4,4), stride=(4,4))

    encoder_mlp = basic_models.MLP(input_dim=1000, hidden_layers=[512,latent_d])
    projection_head = basic_models.MLP(input_dim=latent_d, hidden_layers=[128,projection_d])
    redshift_mlp = basic_models.MLP(input_dim=latent_d, hidden_layers=config['model']['redshift_mlp_hidden_layers'])
    color_mlp = basic_models.MLP(input_dim=latent_d, hidden_layers=config['model']['color_mlp_hidden_layers'])    

    simclrmoco_model = pita_model.PITALightning(
        encoder=encoder,
        encoder_mlp=encoder_mlp,
        projection_head=projection_head,
        redshift_mlp=redshift_mlp,
        color_mlp=color_mlp,
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

    ## prepping checkpoints and logging
    
    checkpoint_filename = f'candels_{config_file}_run{run}_'+'{epoch}'

    checkpoint_callback = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=config['logging_and_checkpoint']['dir_checkpoint'],
        filename=checkpoint_filename,
        every_n_epochs=config['logging_and_checkpoint']['every_n_epochs'],
        save_top_k=200,
        enable_version_counter=False
    )
        
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=config['logging_and_checkpoint']['dir_log'],
        name=f'candels_{config_file}_run{run}'
    )

    ## training
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        min_epochs=1,
        max_epochs=config['training']['epochs'],
        precision='16-mixed',
        log_every_n_steps=1,
        default_root_dir="/global/homes/a/ashodkh/image_photo_z/scripts",
        strategy='ddp',
        logger=tb_logger,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, lr_monitor_callback],
    )
    trainer.fit(simclrmoco_model, data_module)