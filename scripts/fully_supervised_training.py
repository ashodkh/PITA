import numpy as np
from torch import nn
from torchvision.transforms import v2
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch import loggers as pl_loggers
import yaml
import argparse

# import custom modules
import fully_supervised_model as fully_supervised_model
import basic_models as basic_models
import candels_data_modules as dm
import reddening as reddening
import augmentations as augs

parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str)
parser.add_argument('run', type=int)
args = parser.parse_args()

config_file = args.config_file
with open(f"/global/homes/a/ashodkh/image_photo_z/scripts/{config_file}.yaml", "r") as f:
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
            augs.JitterCrop(output_dim=config['augmentations']['crop_dim'], jitter_lim=config['augmentations']['jitter_lim']),
            augs.AddGaussianNoise(mean=0, std=band_mads)
        ])
    else:
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(180, interpolation=v2.InterpolationMode.BILINEAR),
            augs.JitterCrop(output_dim=64, jitter_lim=4),
        ])
    data_module = dm.ImagesDataModule(
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
        label_f=config['data']['label_f']
    )

    ## prepping model

    latent_d = config['model']['latent_d']
    
    encoder = models.convnext_tiny(weights=None)
    encoder._modules["features"][0][0] = nn.Conv2d(config['data']['n_filters'], 96, kernel_size=(4,4), stride=(4,4))
    encoder_mlp = basic_models.MLP(input_dim=1000, hidden_layers=[512,latent_d])
    redshift_mlp = basic_models.MLP(input_dim=latent_d, hidden_layers=config['model']['redshift_mlp_hidden_layers']) 

    # encoder_type = 'my_encoder'
    # joint_blocks = basic_models.JointBlocks(input_channels=32, block_channels=[32,64], avg_pooling_layers=[4,4])
    # encoder = basic_models.Encoder(input_channels=config['data']['n_filters'], first_layer_output_channels=32, joint_blocks=joint_blocks)
    # encoder_mlp = None
    # redshift_mlp = basic_models.MLP(input_dim=1024, hidden_layers=[512,256,64,1])  
    
    photoz_model = fully_supervised_model.CNNPhotoz(
        encoder=encoder,
        encoder_mlp=encoder_mlp,
        redshift_mlp=redshift_mlp,
        transforms=transforms,
        lr=config['training']['learning_rate'],
        lr_scheduler=config['training']['lr_scheduler']['type'],
        warmupcosine_warmup_epochs=config['training']['lr_scheduler']['wc_ann']['warmup_epochs'],
        warmupcosine_half_period=config['training']['lr_scheduler']['wc_ann']['half_period'],
        warmupcosine_min_lr=config['training']['lr_scheduler']['wc_ann']['min_lr']
    )

    checkpoint_filename = f'candels_{config_file}_run{run}_'+'{epoch}'
    
    checkpoint_callback = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=config['logging_and_checkpoint']['dir_checkpoint'],
        filename=checkpoint_filename,
        every_n_epochs=config['logging_and_checkpoint']['every_n_epochs'],
        save_top_k=100,
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
        precision='16-mixed',
        log_every_n_steps=1,
        default_root_dir="/global/homes/a/ashodkh/image_photo_z/scripts",
        strategy='ddp',
        logger=tb_logger,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, lr_monitor_callback],
    )
    trainer.fit(photoz_model, data_module)