import pytorch_lightning as pl
import torch
from lr_schedulers import WarmupCosineAnnealingScheduler, WarmupCosine

class CNNPhotoz(pl.LightningModule):
    """
    A PyTorch Lightning module for CNN photo-z.

    Args:
        encoder (nn.Module): A CNN encoder.
        encoder_mlp (nn.Module, optional): An optional MLP that projects encoder outputs to a lower dimension.
        redshift_mlp (nn.Module): The final MLP for redshift prediction.
        lr (float): Learning rate for the optimizer.
        transforms (callable): Optional image augmentations. 
        lr_scheduler: Type of lr scheduler. Options are: multistep, cosine, warmupcosine, and wc_ann. 
    """
    
    def __init__(
        self,
        encoder: torch.nn.Module=None,
        encoder_mlp: torch.nn.Module=None,
        redshift_mlp: torch.nn.Module=None,
        transforms=None,
        lr=None,
        lr_scheduler=None,

        # cosine lr params
        cosine_T_max=500,
        cosine_eta_min=1e-6,

        # multistep lr params
        multistep_milestones=[1500],
        multistep_gamma=0.1,

        # warmupcosine lr params
        warmupcosine_warmup_epochs=200,
        warmupcosine_half_period=900,
        warmupcosine_min_lr=1e-6,

        #wc_ann lr params
        wc_ann_warmup_epochs=200,
        wc_ann_half_period=900,
        wc_ann_min_lr=1e-6
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_mlp = encoder_mlp
        self.redshift_mlp = redshift_mlp
        self.transforms = transforms
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.cosine_T_max = cosine_T_max
        self.cosine_eta_min = cosine_eta_min
        self.multistep_milestones = multistep_milestones
        self.multistep_gamma = multistep_gamma
        self.warmupcosine_warmup_epochs = warmupcosine_warmup_epochs
        self.warmupcosine_half_period = warmupcosine_half_period
        self.warmupcosine_min_lr = warmupcosine_min_lr
        self.wc_ann_warmup_epochs = wc_ann_warmup_epochs
        self.wc_ann_half_period = wc_ann_half_period
        self.wc_ann_min_lr = wc_ann_min_lr
        
    def forward(self, x):
        """
        Forward pass through the encoder, optional MLP, and the final MLP.
        """
        x = self.encoder(x)
        if self.encoder_mlp is not None:
            x = self.encoder_mlp(x)
        x = self.redshift_mlp(x)
        return x
    
    def huber_loss(self, predictions, truths, delta=0.15):
        """
        Huber loss is quadratic (l2) for x < delta and linear (l1) for x > delta.
        """
        loss = torch.nn.HuberLoss(delta=delta)
        return loss(predictions, truths)
    
    def training_step(self, batch_data, batch_idx):
        """
        Training step: processes the batch, computes the loss, and logs metrics.
        """
        batch_images, batch_redshifts, batch_redshift_weights, _ = batch_data

        batch_redshifts = batch_redshifts.to(torch.float32)
        batch_redshift_weights = batch_redshift_weights.to(torch.float32)
        
        if self.transforms:
            batch_redshift_predictions = self.forward(self.transforms(batch_images)).squeeze()
        else:
            batch_redshift_predictions = self.forward(batch_images).squeeze()
        
        # assert Pytorch output and true redshifts/weights have same shape
        assert batch_redshifts.shape == batch_redshift_predictions.shape
        assert batch_redshift_predictions.shape == batch_redshift_weights.shape
        
        loss = self.huber_loss(batch_redshift_predictions, batch_redshifts)
        self.log("training_loss", loss, on_epoch=True, sync_dist=True)
        
        # Compute metrics (bias, NMAD, and outlier fraction) and log them
        
        delta = (batch_redshift_predictions - batch_redshifts) / (1+batch_redshifts)
        bias = torch.mean(delta)
        nmad = 1.4826*torch.median(torch.abs(delta-torch.median(delta)))
        outlier_fraction = torch.sum(torch.abs(delta)>0.15)/len(batch_redshifts)
        
        self.log('training_bias', bias, on_epoch=True, sync_dist=True)
        self.log('training_nmad', nmad, on_epoch=True, sync_dist=True)
        self.log('training_outlier_f', outlier_fraction, on_epoch=True, sync_dist=True)
        
        return loss
        
    def validation_step(self, batch_data, batch_idx):
        """
        Same as training step but for validation data.
        """
        batch_images, batch_redshifts, batch_redshift_weights, _ = batch_data
        
        batch_redshifts = batch_redshifts.to(torch.float32)
        batch_redshift_weights = batch_redshift_weights.to(torch.float32)
        
        if self.transforms:
            batch_redshift_predictions = self.forward(self.transforms(batch_images)).squeeze()
        else:
            batch_redshift_predictions = self.forward(batch_images).squeeze()

        # assert Pytorch output and true redshifts/weights have same shape
        assert batch_redshifts.shape == batch_redshift_predictions.shape
        assert batch_redshift_predictions.shape == batch_redshift_weights.shape
        
        loss = self.huber_loss(batch_redshift_predictions, batch_redshifts)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        # Compute metrics (bias, NMAD, and outlier fraction) and log them

        delta = (batch_redshift_predictions - batch_redshifts) / (1+batch_redshifts)
        bias = torch.mean(delta)
        nmad = 1.4826*torch.median(torch.abs(delta-torch.median(delta)))
        outlier_fraction = torch.sum(torch.abs(delta)>0.15)/len(batch_redshifts)
        
        self.log('val_bias', bias, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_nmad', nmad, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_outlier_f', outlier_fraction, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-05)

        if self.lr_scheduler == 'multistep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optim,
                milestones=self.multistep_milestones,
                gamma=self.multistep_gamma
            )
            
        if self.lr_scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optim,
                T_max=self.cosine_T_max,
                eta_min=self.cosine_eta_min
            )

        if self.lr_scheduler == 'warmupcosine':
            lr_scheduler = WarmupCosine(
                optimizer=optim,
                warmup_epochs=self.warmupcosine_warmup_epochs,
                cos_half_period=self.warmupcosine_half_period,
                min_lr=self.warmupcosine_min_lr
            )

        if self.lr_scheduler == 'wc_ann':
            lr_scheduler = WarmupCosineAnnealingScheduler(
                optimizer=optim,
                warmup_epochs=self.wc_ann_warmup_epochs,
                cos_half_period=self.wc_ann_half_period,
                min_lr=self.wc_ann_min_lr
            )

        if self.lr_scheduler is None:
            return optim
        else:
            return [optim], [lr_scheduler]