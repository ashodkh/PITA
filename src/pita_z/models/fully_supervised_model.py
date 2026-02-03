import pytorch_lightning as pl
import torch
from pita_z.utils.lr_schedulers import WarmupCosineAnnealingScheduler, WarmupCosine
from calpit.utils import trapz_grid_torch
from scipy.interpolate import PchipInterpolator
import numpy as np

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

class CalpitPhotometryLightning(pl.LightningModule):
    def __init__(
        self,
        model=None,
        loss_type='bce',
        lr=None,
        lr_scheduler=None,
        alpha_grid=None,
        y_grid=None,
        cde_init_type='uniform'
    ):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.register_buffer("alpha_grid", torch.as_tensor(alpha_grid, dtype=torch.float32))
        self.register_buffer("y_grid", torch.as_tensor(y_grid, dtype=torch.float32))
        if cde_init_type == 'uniform':
            self.register_buffer("cde_init", torch.full((len(y_grid),), 1/(y_grid[-1]-y_grid[0]), dtype=torch.float32))

    def forward(self, x):
        return self.model(x)

    def transform(self, x):
        """
        Transforms the input CDEs to the calibrated CDEs.
        """
        cde_init = torch.tile(self.cde_init, (x.shape[0],1))
        cdf_init = trapz_grid_torch(cde_init, self.y_grid)
        features = torch.cat(
            [
                torch.ravel(cdf_init)[:,None],
                torch.repeat_interleave(x, cdf_init.shape[1], dim=0)
            ],
            dim = -1
        )
        cdf_new = self.forward(features.float()).reshape((x.shape[0], len(self.y_grid)))
        cdf_new_funct = PchipInterpolator(self.y_grid.detach().cpu(), cdf_new.detach().cpu(), extrapolate=True, axis=1)
        pdf_func = cdf_new_funct.derivative(1)
        cde_new = pdf_func(self.y_grid.detach().cpu())
        return torch.tensor(cde_new, device=x.device)
        
    def loss_fn(self, predictions, truths):
        if self.loss_type == 'bce':
            loss = torch.nn.BCELoss(reduction='mean')
            return loss(predictions, truths)
        
    def training_step(self, batch, batch_idx):
        x, y, true_redshifts = batch
        n_batch = x.shape[0]
        alphas = torch.rand(n_batch, device=x.device)
        x = torch.hstack([alphas[:,None], x])
        y = (y <= alphas).float()

        outputs = self.model(x)
        
        loss = self.loss_fn(torch.squeeze(outputs), torch.squeeze(y))
        self.log("training_loss", loss, on_epoch=True, sync_dist=True)

        batch_cdes = self.transform(x[:,1:])
        max_idxs = torch.argmax(batch_cdes, axis=1)
        max_ys = self.y_grid[max_idxs]
        
        delta = (max_ys - true_redshifts) / (1 + true_redshifts)
        bias = torch.mean(delta)
        nmad = 1.4826*torch.median(torch.abs(delta-torch.median(delta)))
        outlier_fraction = torch.sum(torch.abs(delta)>0.15)/len(true_redshifts)
        
        self.log('training_bias', bias, on_epoch=True, sync_dist=True)
        self.log('training_nmad', nmad, on_epoch=True, sync_dist=True)
        self.log('training_outlier_f', outlier_fraction, on_epoch=True, sync_dist=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y, true_redshifts = batch
        x_device = x.device
        n_batch = x.shape[0]
        n_alphas = len(self.alpha_grid)
        features = torch.cat(
            [
                torch.repeat_interleave(self.alpha_grid, n_batch)[:,None],
                torch.tile(x, (n_alphas,1))
            ],
            dim = -1
        )
        y = (torch.tile(y, (n_alphas,)) <= torch.repeat_interleave(self.alpha_grid, n_batch)).float()

        outputs = self.model(features)

        loss = self.loss_fn(torch.squeeze(outputs), torch.squeeze(y))
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        batch_cdes = self.transform(x)
        max_idxs = torch.argmax(batch_cdes, axis=1)
        max_ys = self.y_grid[max_idxs]
        
        delta = (max_ys - true_redshifts) / (1 + true_redshifts)
        bias = torch.mean(delta)
        nmad = 1.4826*torch.median(torch.abs(delta-torch.median(delta)))
        outlier_fraction = torch.sum(torch.abs(delta)>0.15)/len(true_redshifts)
        
        self.log('val_bias', bias, on_epoch=True, sync_dist=True)
        self.log('val_nmad', nmad, on_epoch=True, sync_dist=True)
        self.log('val_outlier_f', outlier_fraction, on_epoch=True, sync_dist=True)
        
        return loss
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_scheduler is None:
            return optim
                

class CalpitCNNPhotoz(pl.LightningModule):
    """
    A PyTorch Lightning module for Calpit CNN photo-z.

    Args:
        encoder (nn.Module): A CNN encoder.
        encoder_mlp (nn.Module, optional): An optional MLP that projects encoder outputs to a lower dimension.
        redshift_mlp (nn.Module): The final MLP for redshift prediction.
        loss_type (string): Calpit loss type. Default is binary cross-entropy.
        alpha_grid (np.ndarray or torch.tensor): Fixed grid of alpha's to calculate validation loss.
        y_grid (np.ndarray or torch.tensor): Grid of y-values on which pdfs are calculated.
        cde_init_type (string): Initial cde guess. Defaults to uniform.
        transforms (callable): Optional image augmentations.
        transforms_val (callable): Optional image augmentations for the validation set.
        lr (float): Learning rate for the optimizer.
        lr_scheduler: Type of lr scheduler. Options are: multistep, cosine, warmupcosine, and wc_ann. 
    """
    
    def __init__(
        self,
        encoder: torch.nn.Module=None,
        encoder_mlp: torch.nn.Module=None,
        redshift_mlp: torch.nn.Module=None,
        loss_type='bce',
        alpha_grid=None,
        y_grid=None,
        cde_init_type='uniform',
        transforms=None,
        transforms_val=None,
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
        self.loss_type = loss_type
        self.register_buffer("alpha_grid", torch.as_tensor(alpha_grid, dtype=torch.float32))
        self.register_buffer("y_grid", torch.as_tensor(y_grid, dtype=torch.float32))
        if cde_init_type == 'uniform':
            self.register_buffer("cde_init", torch.full((len(y_grid),), 1/(y_grid[-1]-y_grid[0]), dtype=torch.float32))
        self.transforms = transforms
        self.transforms_val = transforms_val
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
        
    def forward(self, x, goal='train'):
        """
        Forward pass through the encoder, optional MLP, and the final MLP. Adds alphas before final MLP
        """
        x = self.encoder(x)
        x = self.encoder_mlp(x)
        n_batch = x.shape[0]
        if goal == 'train':
            alphas = torch.rand(n_batch, device=x.device)
            x = torch.hstack([alphas[:,None], x])
            x = self.redshift_mlp(x)
            return x.squeeze(), alphas
        elif goal == 'validate':
            n_alphas = len(self.alpha_grid)
            x = torch.cat(
                [
                    torch.repeat_interleave(self.alpha_grid, n_batch)[:,None],
                    torch.tile(x, (n_alphas,1))
                ],
                dim=-1
            )
            x = self.redshift_mlp(x)
            return x.squeeze(), None

    @torch.no_grad()
    def transform_cde(self, x):
        """
        Transforms the input CDEs to the calibrated CDEs.
        """
        cde_init = torch.tile(self.cde_init, (x.shape[0],1))
        cdf_init = trapz_grid_torch(cde_init, self.y_grid)
        features = self.encoder_mlp(self.encoder(x))
        features = torch.cat(
            [
                torch.ravel(cdf_init)[:,None],
                torch.repeat_interleave(features, cdf_init.shape[1], dim=0)
            ],
            dim = -1
        )
        cdf_new = self.redshift_mlp(features.float()).reshape((x.shape[0], len(self.y_grid)))
        cdf_new_funct = PchipInterpolator(self.y_grid.detach().cpu(), cdf_new.detach().cpu(), extrapolate=True, axis=1)
        pdf_func = cdf_new_funct.derivative(1)
        cde_new = pdf_func(self.y_grid.detach().cpu())
        return torch.tensor(cde_new, device=x.device)

    def loss_fn(self, predictions, truths):
        if self.loss_type == 'bce':
            loss = torch.nn.BCELoss(reduction='mean')
            return loss(predictions, truths)
    
    def training_step(self, batch_data, batch_idx):
        """
        Training step: processes the batch, computes the loss, and logs metrics.
        """
        batch_images, batch_pits, batch_redshifts, batch_redshift_weights, _ = batch_data

        if self.transforms:
            batch_images = self.transforms(batch_images)
            
        batch_predictions, alphas = self.forward(batch_images, goal='train')
        y = (batch_pits <= alphas).float()
        loss = self.loss_fn(batch_predictions, torch.squeeze(y))
        
        self.log("training_loss", loss, on_epoch=True, sync_dist=True)
        
        # Compute metrics (bias, NMAD, and outlier fraction) and log them
        batch_cdes = self.transform_cde(batch_images)
        max_idxs = torch.argmax(batch_cdes, axis=1)
        max_ys = self.y_grid[max_idxs]
        
        delta = (max_ys - batch_redshifts) / (1 + batch_redshifts)
        bias = torch.mean(delta)
        nmad = 1.4826 * torch.median(torch.abs(delta-torch.median(delta)))
        outlier_fraction = torch.sum(torch.abs(delta)>0.15) / len(batch_redshifts)
        
        self.log('training_bias', bias, on_epoch=True, sync_dist=True)
        self.log('training_nmad', nmad, on_epoch=True, sync_dist=True)
        self.log('training_outlier_f', outlier_fraction, on_epoch=True, sync_dist=True)
        
        return loss
        
    def validation_step(self, batch_data, batch_idx):
        """
        Same as training step but for validation data.
        """
        batch_images, batch_pits, batch_redshifts, batch_redshift_weights, _ = batch_data
        
        if self.transforms_val:
            batch_images = self.transforms_val(batch_images)
        
        batch_predictions, _ = self.forward(batch_images, goal='validate')
        n_batch = batch_images.shape[0]
        n_alphas = len(self.alpha_grid)
        y = (torch.tile(batch_pits, (n_alphas,)) <= torch.repeat_interleave(self.alpha_grid, n_batch)).float()
        
        loss = self.loss_fn(batch_predictions, torch.squeeze(y))
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        # Compute metrics (bias, NMAD, and outlier fraction) and log them
        batch_cdes = self.transform_cde(batch_images)
        max_idxs = torch.argmax(batch_cdes, axis=1)
        max_ys = self.y_grid[max_idxs]
        
        delta = (max_ys - batch_redshifts) / (1 + batch_redshifts)
        bias = torch.mean(delta)
        nmad = 1.4826 * torch.median(torch.abs(delta-torch.median(delta)))
        outlier_fraction = torch.sum(torch.abs(delta)>0.15) / len(batch_redshifts)
        
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