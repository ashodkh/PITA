import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pita_z.utils.lr_schedulers import WarmupCosineAnnealingScheduler, WarmupCosine
import copy
import random
from calpit.utils import trapz_grid_torch
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
import numpy as np

class PITALightning(pl.LightningModule):
    """
    PITA Pytorch Lightning Module.

    Attributes:
        encoder (nn.Module): A CNN encoder.
        encoder_mlp (nn.Module, optional): An optional MLP that projects encoder outputs to a lower dimension.
        projection_head (nn.Module): An MLP that projects encoder outputs (or encoder_mlp outputs)
                                     to a lower-dimensional space where the contrastive loss is calculated.
        redshift_mlp (nn.Module): An MLP that estimates redshift from encoder outputs (or encoder_mlp outputs).
        color_mlp (nn.Module): An MLP that estimates photometric colors from encoder outputs (or encoder_mlp outputs).
        transforms (callable): Image augmentations used to generate two views for contrastive learning. 
        momentum: Momentum parameter for updating dictionary encoder.
        queue_size: Queue size of the dictionary.
        temperature: Contrastive loss function hyperparameter.
        cl_loss_weight: The weight of the contrastive loss. Default is 0.0025.
        redshift_loss_weight: The weight of the redshift prediction loss. Default is 1.
        color_loss_weight: The weight of the color prediction loss. Default is 1.
        lr (float): Learning rate for the optimizer.
        lr_scheduler: Type of lr scheduler. Options are: multistep, cosine, and warmupcosine. 
    """
    
    def __init__(
        self,
        encoder=None,
        encoder_mlp=None,
        projection_head=None,
        redshift_mlp=None,
        color_mlp=None,
        transforms=None,
        momentum=0.999,
        queue_size=50000,
        temperature=0.1,
        cl_loss_weight=0.0025,
        redshift_loss_weight=1,
        color_loss_weight=1,
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
        wc_ann_warmup_epochs=100,
        wc_ann_half_period=900,
        wc_ann_min_lr=1e-6
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_mlp = encoder_mlp
        self.projection_head = projection_head
        self.redshift_mlp = redshift_mlp
        self.color_mlp = color_mlp
        self.transforms = transforms
        self.momentum = momentum
        self.temperature = temperature
        self.cl_loss_weight = cl_loss_weight
        self.redshift_loss_weight = redshift_loss_weight
        self.color_loss_weight = color_loss_weight
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
        
        # Initialize the momentum (key) encoder and its heads as copies of the original
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_encoder_mlp = copy.deepcopy(self.encoder_mlp) if encoder_mlp else None
        self.momentum_projection_head = copy.deepcopy(self.projection_head)
        
        # Freeze all parameters in the momentum encoder and its heads
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        if self.momentum_encoder_mlp:
            for param in self.momentum_encoder_mlp.parameters():
                param.requires_grad = False
        for param in self.momentum_projection_head.parameters():
            param.requires_grad = False
        
        # Initialize the queue for negative samples
        self.register_buffer("queue", torch.randn(queue_size, projection_head.output_dim))
        self.queue = F.normalize(self.queue, dim=1) 
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def forward(self, x, use_momentum_encoder=False):
        """Forward pass through the encoder and MLPs."""
        if use_momentum_encoder:
            x = self.momentum_encoder(x)
            if self.momentum_encoder_mlp:
                x = self.momentum_encoder_mlp(x)
            x_proj = self.momentum_projection_head(x)
        else:
            x = self.encoder(x)
            if self.encoder_mlp:
                x = self.encoder_mlp(x)
            x_proj = self.projection_head(x)

        x_redshift, x_color = None, None
        if self.redshift_mlp:
            x_redshift = self.redshift_mlp(x).squeeze()
        if self.color_mlp:
            x_color = self.color_mlp(x)
        
        return F.normalize(x_proj, dim=1), x_redshift, x_color
    
    @torch.no_grad()
    def update_momentum_encoder(self):
        """Update momentum encoder and its heads using exponential moving average."""
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data
        if self.encoder_mlp and self.momentum_encoder_mlp:
            for param_q, param_k in zip(self.encoder_mlp.parameters(), self.momentum_encoder_mlp.parameters()):
                param_k.data = self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data
        for param_q, param_k in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
            param_k.data = self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """
        Enqueue the current batch of keys and dequeue the oldest to maintain a fixed-size queue.
        Each GPU contributes its keys to ensure the queue is synchronized across processes.
        """
        # Gather keys from all GPUs
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            keys_all = [torch.zeros_like(keys) for _ in range(world_size)]
            torch.distributed.all_gather(keys_all, keys)
            keys = torch.cat(keys_all, dim=0)  # (world_size * batch_size, dim)
    
        batch_size = keys.shape[0]
        queue_size = self.queue.shape[0]
        ptr = int(self.queue_ptr.item())  # Convert from 1-element tensor to int
    
        # If not enough space to enqueue the entire batch, wrap around
        if ptr + batch_size > queue_size:
            overflow = (ptr + batch_size) - queue_size
            self.queue[ptr:] = keys[:queue_size - ptr]
            self.queue[:overflow] = keys[queue_size - ptr:]
        else:
            self.queue[ptr:ptr + batch_size] = keys
    
        # Update pointer
        ptr = (ptr + batch_size) % queue_size
        self.queue_ptr[0] = ptr
            
    def contrastive_loss(self, queries, keys):
        """Compute contrastive loss for MoCo using a memory queue of negative samples."""
        # Positive logits: Nx1 (dot product of each query with its corresponding key)
        pos_logits = torch.einsum('nc,nc->n', [queries, keys]).unsqueeze(-1) / self.temperature
        
        # Negative logits: NxK (dot product of each query with all keys in the queue)
        neg_logits = torch.einsum('nc,kc->nk', [queries, self.queue.clone().detach()]) / self.temperature
        
        # Combine positive and negative logits
        logits = torch.cat([pos_logits, neg_logits], dim=1) # results in Nx(1+k)
        # the zero labels indicate that the 0 index is the target class (or positive pair)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Return average positive similarity and cross entropy loss
        return torch.mean(pos_logits)*self.temperature, F.cross_entropy(logits, labels)
    
    def weighted_mse_loss(self, predictions, truths, weights=1):
        mse_loss = torch.mean((predictions - truths) ** 2 * weights)
        return mse_loss
    
    def huber_loss(self, predictions, truths, delta=0.15):
        """
        Huber loss is quadratic (l2) for x < delta and linear (l1) for x > delta.
        """
        loss = torch.nn.HuberLoss(delta=delta)
        return loss(predictions, truths)

    def redshift_loss_and_metrics(self, predicted_redshifts, true_redshifts, redshift_weights):
        """
        Calculates huber loss and photo-z performance metrics
        """

        # only use available redshifts
        good_redshifts_mask = redshift_weights == 1
        if good_redshifts_mask.sum() == 0:
            redshift_loss = 0 * self.redshift_loss_fn(w_alphas[:1], torch.squeeze(y)[:1])
            redshift_loss, bias, nmad, outlier_fraction = 0, 0, 0, 0
        else:
            predicted_redshifts = predicted_redshifts[good_redshifts_mask]
            true_redshifts = true_redshifts[good_redshifts_mask]

            redshift_loss = self.huber_loss(predicted_redshifts, true_redshifts)
            redshift_loss = redshift_loss * self.redshift_loss_weight

            delta = (predicted_redshifts - true_redshifts) / (1+true_redshifts)
            bias = torch.mean(delta)
            nmad = 1.4826*torch.median(torch.abs(delta-torch.median(delta)))
            outlier_fraction = torch.sum(torch.abs(delta)>0.15)/len(true_redshifts)

        return redshift_loss, bias, nmad, outlier_fraction
            
    def training_step(self, batch_data, batch_idx):
        batch_images, batch_redshifts, batch_redshift_weights, batch_colors = batch_data
        batch_images = batch_images.to(torch.float32)
        batch_redshifts = batch_redshifts.to(torch.float32)
        batch_redshift_weights = batch_redshift_weights.to(torch.float32)
        batch_colors = batch_colors.to(torch.float32)
        # Apply transformations to create two augmented views
        view_1 = self.transforms(batch_images)
        view_2 = self.transforms(batch_images)
        
        # Forward pass for query (main encoder) and key (momentum encoder)
        queries, redshift_predictions, color_predictions = self.forward(view_1) # Queries, redshifts, and colors from main encoder
        with torch.no_grad():  # No gradients for momentum encoder
            keys, _, _ = self.forward(view_2, use_momentum_encoder=True)  # Keys from momentum encoder

        # Update the momentum encoder and enqueue the keys
        self.update_momentum_encoder()
        self.dequeue_and_enqueue(keys)
        
        # Compute and log the contrastive loss
        pos_sim, cl_loss = self.contrastive_loss(queries, keys)
        cl_loss = cl_loss * self.cl_loss_weight
        
        self.log("cl_training_loss", cl_loss, on_epoch=True, sync_dist=True)
        self.log("training_pos_sim", pos_sim, on_epoch=True, sync_dist=True)

        total_loss = cl_loss
        if self.redshift_mlp:
            redshift_loss, bias, nmad, outlier_fraction\
            = self.redshift_loss_and_metrics(redshift_predictions, batch_redshifts, batch_redshift_weights)
            
            self.log('training_bias', bias, on_step=True, on_epoch=True, sync_dist=True)
            self.log('training_nmad', nmad, on_step=True, on_epoch=True, sync_dist=True)
            self.log('training_outlier_f', outlier_fraction, on_step=True, on_epoch=True, sync_dist=True)
            self.log('redshift_training_loss', redshift_loss, on_step=True, on_epoch=True, sync_dist=True)

            total_loss += redshift_loss
            
        if self.color_mlp:
            color_loss = self.weighted_mse_loss(color_predictions, batch_colors)
            color_loss = color_loss * self.color_loss_weight
            self.log("color_training_loss", color_loss, on_epoch=True, sync_dist=True)
            total_loss += color_loss

        self.log("total_training_loss", total_loss, on_epoch=True, sync_dist=True)

        return total_loss
    
    def validation_step(self, batch_data, batch_idx):
        batch_images, batch_redshifts, batch_redshift_weights, batch_colors = batch_data
        batch_images = batch_images.to(torch.float32)
        batch_redshifts = batch_redshifts.to(torch.float32)
        batch_redshift_weights = batch_redshift_weights.to(torch.float32)
        batch_colors = batch_colors.to(torch.float32)
        
        # Apply transformations to create two augmented views
        view_1 = self.transforms(batch_images)
        view_2 = self.transforms(batch_images)
        
        # Forward pass for query (main encoder) and key (momentum encoder)
        queries, redshift_predictions, color_predictions = self.forward(view_1) # Queries, redshifts, and colors from main encoder
        with torch.no_grad():  # No gradients for momentum encoder
            keys, _, _ = self.forward(view_2, use_momentum_encoder=True)  # Keys from momentum encoder
        
        # Compute and log the contrastive loss
        pos_sim, cl_loss = self.contrastive_loss(queries, keys)
        cl_loss = cl_loss * self.cl_loss_weight
        self.log("cl_validation_loss", cl_loss, on_epoch=True, sync_dist=True)
        self.log("validation_pos_sim", pos_sim, on_epoch=True, sync_dist=True)

        total_loss = cl_loss

        if self.redshift_mlp is not None:
            redshift_loss, bias, nmad, outlier_fraction\
            = self.redshift_loss_and_metrics(redshift_predictions, batch_redshifts, batch_redshift_weights)
            
            self.log('val_bias', bias, on_step=True, on_epoch=True, sync_dist=True)
            self.log('val_nmad', nmad, on_step=True, on_epoch=True, sync_dist=True)
            self.log('val_outlier_f', outlier_fraction, on_step=True, on_epoch=True, sync_dist=True)
            self.log('redshift_validation_loss', redshift_loss, on_step=True, on_epoch=True, sync_dist=True)
            
            total_loss += redshift_loss

        if self.color_mlp is not None:
            color_loss = self.weighted_mse_loss(color_predictions, batch_colors)
            color_loss = color_loss * self.color_loss_weight
            self.log("color_validation_loss", color_loss, on_epoch=True, sync_dist=True)

            total_loss += color_loss
        
        self.log("total_validation_loss", total_loss, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-05)
        #optim = ADOPT(self.parameters(), lr=self.lr, weight_decay=1e-05)

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

class CalPITALightning(pl.LightningModule):
    """
    CalPITA Pytorch Lightning Module.

    Attributes:
        encoder (nn.Module): A CNN encoder.
        encoder_mlp (nn.Module, optional): An optional MLP that projects encoder outputs to a lower dimension.
        projection_head (nn.Module): An MLP that projects encoder outputs (or encoder_mlp outputs)
                                     to a lower-dimensional space where the contrastive loss is calculated.
        redshift_mlp (nn.Module): An MLP that estimates redshift from encoder outputs (or encoder_mlp outputs).
        color_mlp (nn.Module): An MLP that estimates photometric colors from encoder outputs (or encoder_mlp outputs).
        loss_type (string): Calpit loss type. Default is binary cross-entropy.
        alpha_grid (np.ndarray or torch.tensor): Fixed grid of alpha's to calculate validation loss.
        y_grid (np.ndarray or torch.tensor): Grid of y-values on which pdfs are calculated.
        cde_init_type (string): Initial cde guess. Defaults to uniform.
        transforms (callable): Image augmentations used to generate two views for contrastive learning. 
        momentum: Momentum parameter for updating dictionary encoder.
        queue_size: Queue size of the dictionary.
        temperature: Contrastive loss function hyperparameter.
        cl_loss_weight: The weight of the contrastive loss. Default is 0.0025.
        redshift_loss_weight: The weight of the redshift prediction loss. Default is 1.
        color_loss_weight: The weight of the color prediction loss. Default is 1.
        lr (float): Learning rate for the optimizer.
        lr_scheduler: Type of lr scheduler. Options are: multistep, cosine, and warmupcosine. 
    """
    
    def __init__(
        self,
        encoder=None,
        encoder_mlp=None,
        projection_head=None,
        redshift_mlp=None,
        color_mlp=None,
        loss_type='bce',
        alpha_grid=None,
        y_grid=None,
        cde_init_type='uniform',
        transforms=None,
        transforms_z_metric=None,
        momentum=0.999,
        queue_size=50000,
        temperature=0.1,
        cl_loss_weight=0.0025,
        redshift_loss_weight=1,
        color_loss_weight=1,
        lr=None,
        lamda=1,
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
        wc_ann_warmup_epochs=100,
        wc_ann_half_period=900,
        wc_ann_min_lr=1e-6
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_mlp = encoder_mlp
        self.projection_head = projection_head
        self.redshift_mlp = redshift_mlp
        self.color_mlp = color_mlp
        self.loss_type = loss_type
        self.register_buffer("alpha_grid", torch.as_tensor(alpha_grid, dtype=torch.float32))
        self.register_buffer("y_grid", torch.as_tensor(y_grid, dtype=torch.float32))
        if cde_init_type == 'uniform':
            self.register_buffer("cde_init", torch.full((len(y_grid),), 1/(y_grid[-1]-y_grid[0]), dtype=torch.float32))
        self.transforms = transforms
        self.transforms_z_metric = transforms_z_metric
        self.momentum = momentum
        self.temperature = temperature
        self.cl_loss_weight = cl_loss_weight
        self.redshift_loss_weight = redshift_loss_weight
        self.color_loss_weight = color_loss_weight
        self.lr = lr
        self.lamda = lamda
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
        
        # Initialize the momentum (key) encoder and its heads as copies of the original
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_encoder_mlp = copy.deepcopy(self.encoder_mlp) if encoder_mlp else None
        self.momentum_projection_head = copy.deepcopy(self.projection_head)
        
        # Freeze all parameters in the momentum encoder and its heads
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        if self.momentum_encoder_mlp:
            for param in self.momentum_encoder_mlp.parameters():
                param.requires_grad = False
        for param in self.momentum_projection_head.parameters():
            param.requires_grad = False
        
        # Initialize the queue for negative samples
        self.register_buffer("queue", torch.randn(queue_size, projection_head.output_dim))
        self.queue = F.normalize(self.queue, dim=1) 
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.automatic_optimization = False
    
    def forward(self, x, use_momentum_encoder=False, goal='train'):
        """
        Forward pass through the encoder and MLPs.
        If goal is 'train', then a random alpha is generated and concatenated to the latent vector.
        If goal is to 'validate', then alpha_grid is concatenated to the latent vector.
        """
        if use_momentum_encoder:
            x = self.momentum_encoder(x)
            if self.momentum_encoder_mlp:
                x = self.momentum_encoder_mlp(x)
            x_proj = self.momentum_projection_head(x)
        else:
            x = self.encoder(x)
            if self.encoder_mlp:
                x = self.encoder_mlp(x)
            x_proj = self.projection_head(x)

        n_batch = x.shape[0]
        x_redshift, x_color = None, None            
        if self.redshift_mlp:
            if goal == 'train':
                alphas = torch.rand(n_batch, 1, device=x.device).requires_grad_(True)
                x_redshift = torch.hstack([alphas, x])
                x_redshift = self.redshift_mlp(x_redshift)
            elif goal == 'validate':
                n_alphas = len(self.alpha_grid)
                x_redshift = torch.cat(
                    [
                        torch.repeat_interleave(self.alpha_grid, n_batch)[:,None],
                        torch.tile(x, (n_alphas,1))
                    ],
                    dim=-1
                )
                x_redshift = self.redshift_mlp(x_redshift)
        if self.color_mlp:
            x_color = self.color_mlp(x).squeeze()

        if goal == 'train':
            return F.normalize(x_proj, dim=1), x_redshift, x_color, alphas
        elif goal == 'validate':
            return F.normalize(x_proj, dim=1), x_redshift, x_color, None
            
    @torch.no_grad()
    def update_momentum_encoder(self):
        """Update momentum encoder and its heads using exponential moving average."""
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data
        if self.encoder_mlp and self.momentum_encoder_mlp:
            for param_q, param_k in zip(self.encoder_mlp.parameters(), self.momentum_encoder_mlp.parameters()):
                param_k.data = self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data
        for param_q, param_k in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
            param_k.data = self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """
        Enqueue the current batch of keys and dequeue the oldest to maintain a fixed-size queue.
        Each GPU contributes its keys to ensure the queue is synchronized across processes.
        """
        # Gather keys from all GPUs
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            keys_all = [torch.zeros_like(keys) for _ in range(world_size)]
            torch.distributed.all_gather(keys_all, keys)
            keys = torch.cat(keys_all, dim=0)  # (world_size * batch_size, dim)
    
        batch_size = keys.shape[0]
        queue_size = self.queue.shape[0]
        ptr = int(self.queue_ptr.item())  # Convert from 1-element tensor to int
    
        # If not enough space to enqueue the entire batch, wrap around
        if ptr + batch_size > queue_size:
            overflow = (ptr + batch_size) - queue_size
            self.queue[ptr:] = keys[:queue_size - ptr]
            self.queue[:overflow] = keys[queue_size - ptr:]
        else:
            self.queue[ptr:ptr + batch_size] = keys
    
        # Update pointer
        ptr = (ptr + batch_size) % queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def transform_cde(self, x):
        """
        Transforms the initial CDEs to the calibrated CDEs using the regression.
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
        cdf_new = self.redshift_mlp(features.float()).reshape((x.shape[0], len(self.y_grid))).detach().cpu()
        
        cdf_new_funct = PchipInterpolator(self.y_grid.detach().cpu(), cdf_new.detach().cpu(), extrapolate=True, axis=1)
        pdf_func = cdf_new_funct.derivative(1)
        cde_new = pdf_func(self.y_grid.detach().cpu())

        ## Isotonic regression ensures that interpolation is monotonic
        ## not needed if using a monotonic NN
        # cdf_regressor = IsotonicRegression()
        # cdf_new_mono = np.stack([cdf_regressor.fit_transform(self.y_grid.detach().cpu(), cdf) for cdf in cdf_new])
        # cde_new = np.gradient(cdf_new_mono, self.y_grid.detach().cpu(), axis=1)
        
        return torch.tensor(cde_new, device=x.device)
    
    def redshift_loss_fn(self, predictions, truths):
        if self.loss_type in ('bce', 'monotonic_bce'):
            loss = torch.nn.BCELoss(reduction='mean')
            return loss(predictions, truths)
            
    def contrastive_loss(self, queries, keys):
        """Compute contrastive loss for MoCo using a memory queue of negative samples."""
        # Positive logits: Nx1 (dot product of each query with its corresponding key)
        pos_logits = torch.einsum('nc,nc->n', [queries, keys]).unsqueeze(-1) / self.temperature
        
        # Negative logits: NxK (dot product of each query with all keys in the queue)
        neg_logits = torch.einsum('nc,kc->nk', [queries, self.queue.clone().detach()]) / self.temperature
        
        # Combine positive and negative logits
        logits = torch.cat([pos_logits, neg_logits], dim=1) # results in Nx(1+k)
        # the zero labels indicate that the 0 index is the target class (or positive pair)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Return average positive similarity and cross entropy loss
        return torch.mean(pos_logits)*self.temperature, F.cross_entropy(logits, labels)
    
    def weighted_mse_loss(self, predictions, truths, weights=1):
        mse_loss = torch.mean((predictions - truths) ** 2 * weights)
        return mse_loss
    
    def huber_loss(self, predictions, truths, delta=0.15):
        """
        Huber loss is quadratic (l2) for x < delta and linear (l1) for x > delta.
        """
        loss = torch.nn.HuberLoss(delta=delta)
        return loss(predictions, truths)

    def redshift_metrics(self, batch_images, true_redshifts):
        """
        Calculates photo-z performance metrics using the redshift mode.
        """

        batch_cdes = self.transform_cde(batch_images)
        max_idxs = torch.argmax(batch_cdes, axis=1)
        max_ys = self.y_grid[max_idxs]

        delta = (max_ys - true_redshifts) / (1 + true_redshifts)
        bias = torch.mean(delta)
        nmad = 1.4826*torch.median(torch.abs(delta-torch.median(delta)))
        outlier_fraction = torch.sum(torch.abs(delta)>0.15)/len(true_redshifts)

        return bias, nmad, outlier_fraction

    # def _get_shared_layer_params(self):
    #     """Returns the parameters shared by the three tasks."""
    #     params = []
    #     for module in self.encoder.modules():
    #         params.extend(module.parameters(recurse=False))
    #     for module in self.encoder_mlp.modules():
    #         params.extend(module.parameters(recurse=False))
    
    #     return params

    @staticmethod
    def pcgrad_update(grads: list[torch.Tensor]) -> list[torch.Tensor]:
        num_tasks = len(grads)
        modified_grads = [g.clone() for g in grads]
    
        for i in range(num_tasks):
            task_indices = list(range(num_tasks))
            task_indices.remove(i)
            random.shuffle(task_indices)  # ← shuffle the order of other tasks
    
            for j in task_indices:
                gi, gj = modified_grads[i], grads[j]
                dot = torch.dot(gi, gj)
                if dot < 0:
                    modified_grads[i] = gi - (dot / (gj.norm() ** 2 + 1e-8)) * gj
    
        return modified_grads
        
    @staticmethod
    def monotonicity_loss(output, x_mono):
        grad = torch.autograd.grad(
            outputs=output,
            inputs=x_mono,
            grad_outputs=torch.ones_like(output),
            create_graph=True
        )[0]

        return torch.relu(-grad).mean()
        
    def training_step(self, batch_data, batch_idx):
        batch_images, batch_pits, batch_redshifts, batch_redshift_weights, batch_colors = batch_data
        # Apply transformations to create two augmented views
        view_1 = self.transforms(batch_images)
        view_2 = self.transforms(batch_images)
        
        # Forward pass for query (main encoder) and key (momentum encoder)
        # alphas (PITs) are concatenated to the latent vectors.
        # To do so, they are generated randomly in the forward function (if goal='Train').
        # forward function also returns these random alphas to calculate true W (binary PIT variable) and loss.
        queries, w_alphas, color_predictions, alphas = self.forward(view_1, goal='train') 
        with torch.no_grad():  # No gradients for momentum encoder
            keys, _, _, _ = self.forward(view_2, use_momentum_encoder=True, goal='train')  # Keys from momentum encoder

        # Update the momentum encoder and enqueue the keys
        self.update_momentum_encoder()
        self.dequeue_and_enqueue(keys)
        
        # Compute and log the contrastive loss
        pos_sim, cl_loss = self.contrastive_loss(queries, keys)
        cl_loss = cl_loss * self.cl_loss_weight
        
        self.log("losses/cl_training_loss", cl_loss, on_epoch=True, sync_dist=True)
        self.log("losses/training_pos_sim", pos_sim, on_epoch=True, sync_dist=True)

        total_loss = cl_loss
        if self.redshift_mlp:
            y = (batch_pits <= alphas.squeeze()).float()
            good_redshifts_mask = batch_redshift_weights == 1
            if good_redshifts_mask.sum() == 0:
                # dummy loss makes sure backprop works properly if there are no redshifts
                dummy_loss = 0 * self.redshift_loss_fn(w_alphas.squeeze()[:1], torch.squeeze(y)[:1])
                if self.loss_type == 'monotonic_bce':
                    dummy_loss += 0 * self.monotonicity_loss(w_alphas, alphas)
                total_loss += dummy_loss
                redshift_loss_for_backward = dummy_loss
                redshift_loss, bias, nmad, outlier_fraction = 0, 0, 0, 0
            else:
                redshift_loss = self.redshift_loss_fn(w_alphas.squeeze()[good_redshifts_mask], torch.squeeze(y)[good_redshifts_mask])
                redshift_loss = redshift_loss * self.redshift_loss_weight
                total_loss += redshift_loss
                bias, nmad, outlier_fraction\
                = self.redshift_metrics(
                    self.transforms_z_metric(batch_images[good_redshifts_mask]),
                    batch_redshifts[good_redshifts_mask]
                )
                if self.loss_type == 'monotonic_bce':
                    mono_loss = self.lamda * self.monotonicity_loss(w_alphas, alphas)
                    total_loss += mono_loss
                    redshift_loss_for_backward = redshift_loss + mono_loss
                else:
                    redshift_loss_for_backward = redshift_loss
                
            self.log("losses/redshift_training_loss", redshift_loss, on_epoch=True, sync_dist=True)  
            self.log('metrics/training_bias', bias, on_step=True, on_epoch=True, sync_dist=True)
            self.log('metrics/training_nmad', nmad, on_step=True, on_epoch=True, sync_dist=True)
            self.log('metrics/training_outlier_f', outlier_fraction, on_step=True, on_epoch=True, sync_dist=True)
            
        if self.color_mlp:
            color_loss = self.weighted_mse_loss(color_predictions, batch_colors)
            color_loss = color_loss * self.color_loss_weight
            self.log("losses/color_training_loss", color_loss, on_epoch=True, sync_dist=True)
            total_loss += color_loss

        self.log("losses/total_training_loss", total_loss, on_epoch=True, sync_dist=True)

        opt = self.optimizers()

        shared_params = list(self.encoder.parameters()) + list(self.encoder_mlp.parameters())
        shared_params = [p for p in shared_params if p.requires_grad]

        task_specific_params = [
            list(self.projection_head.parameters()),
            list(self.color_mlp.parameters()),
            list(self.redshift_mlp.parameters()),
        ]

        task_grads_shared = []
        saved_task_specific_grads = []
        losses = [cl_loss, color_loss, redshift_loss_for_backward]
        grads_norms_pre = []
        for i, (loss, ts_params) in enumerate(zip(losses, task_specific_params)):
            opt.zero_grad()
            self.manual_backward(loss, retain_graph=(i < len(losses) - 1))
            grad = torch.cat([
                p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=p.device)
                for p in shared_params
            ])
            task_grads_shared.append(grad)
            grads_norms_pre.append(torch.norm(grad))
            saved_task_specific_grads.append([
                p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                for p in ts_params
            ])

        modified_grads = self.pcgrad_update(task_grads_shared)
        for i,loss_name in enumerate(['cl', 'color', 'redshift']):
            self.log(f'grads/{loss_name}_grad_pre_projection', grads_norms_pre[i], on_epoch=True, sync_dist=True)
            self.log(f'grads/{loss_name}_grad_post', torch.norm(modified_grads[i]), on_epoch=True, sync_dist=True)
        final_shared_grad = torch.stack(modified_grads).sum(dim=0)

        opt.zero_grad()

        # Set shared param grads to PCGrad result
        offset = 0
        for p in shared_params:
            numel = p.numel()
            p.grad = final_shared_grad[offset: offset + numel].view_as(p).clone()
            offset += numel

        # Restore task-specific param grads saved from individual backward passes
        for ts_params, grads in zip(task_specific_params, saved_task_specific_grads):
            for p, g in zip(ts_params, grads):
                p.grad = g

        opt.step()
        
        return total_loss.detach()
    
    def validation_step(self, batch_data, batch_idx):
        batch_images, batch_pits, batch_redshifts, batch_redshift_weights, batch_colors = batch_data
        
        # Apply transformations to create two augmented views
        view_1 = self.transforms(batch_images)
        view_2 = self.transforms(batch_images)
        
        # Forward pass for query (main encoder) and key (momentum encoder)
        # Validation is done on the whole range [0,1] of PIT.
        # So they don't need to be generated in the forward function (goal='validate').
        queries, w_alphas, color_predictions, _ = self.forward(view_1, goal='validate')
        with torch.no_grad():  # No gradients for momentum encoder
            keys, _, _, _ = self.forward(view_2, use_momentum_encoder=True, goal='validate')  # Keys from momentum encoder
        
        # Compute and log the contrastive loss
        pos_sim, cl_loss = self.contrastive_loss(queries, keys)
        cl_loss = cl_loss * self.cl_loss_weight
        self.log("losses/cl_validation_loss", cl_loss, on_epoch=True, sync_dist=True)
        self.log("losses/validation_pos_sim", pos_sim, on_epoch=True, sync_dist=True)

        total_loss = cl_loss

        if self.redshift_mlp is not None:
            n_batch = batch_images.shape[0]
            n_alphas = len(self.alpha_grid)
            y = (torch.tile(batch_pits, (n_alphas,)) <= torch.repeat_interleave(self.alpha_grid, n_batch)).float()
            good_redshifts_mask = batch_redshift_weights == 1
            good_redshifts_mask_tiled = torch.tile(good_redshifts_mask, (n_alphas,))
            if good_redshifts_mask.sum() == 0:
                dummy_loss = 0 * self.redshift_loss_fn(w_alphas.squeeze()[:1], torch.squeeze(y)[:1])
                total_loss += dummy_loss
                redshift_loss, bias, nmad, outlier_fraction = 0, 0, 0, 0
            else:
                redshift_loss = self.redshift_loss_fn(w_alphas.squeeze()[good_redshifts_mask_tiled], torch.squeeze(y)[good_redshifts_mask_tiled])
                redshift_loss = redshift_loss * self.redshift_loss_weight
                total_loss += redshift_loss
                bias, nmad, outlier_fraction\
                = self.redshift_metrics(
                    self.transforms_z_metric(batch_images[good_redshifts_mask]),
                    batch_redshifts[good_redshifts_mask]
                )
                
            self.log("losses/redshift_val_loss", redshift_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log('metrics/val_bias', bias, on_step=True, on_epoch=True, sync_dist=True)
            self.log('metrics/val_nmad', nmad, on_step=True, on_epoch=True, sync_dist=True)
            self.log('metrics/val_outlier_f', outlier_fraction, on_step=True, on_epoch=True, sync_dist=True)
            
        if self.color_mlp is not None:
            color_loss = self.weighted_mse_loss(color_predictions, batch_colors)
            color_loss = color_loss * self.color_loss_weight
            self.log("losses/color_validation_loss", color_loss, on_epoch=True, sync_dist=True)

            total_loss += color_loss
        
        self.log("losses/total_validation_loss", total_loss, on_epoch=True, sync_dist=True)
        
        return total_loss

    def on_train_epoch_end(self):
        """Manually step the lr scheduler (required with automatic_optimization=False)."""
        sch = self.lr_schedulers()
        if sch is not None:
            if isinstance(sch, list):
                sch[0].step()
            else:
                sch.step()
                
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-05)
        #optim = ADOPT(self.parameters(), lr=self.lr, weight_decay=1e-05)

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
            
        