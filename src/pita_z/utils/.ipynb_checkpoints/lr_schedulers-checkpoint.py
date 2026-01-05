import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, cos_half_period: int,
                 min_lr: float = 5e-7, last_epoch: int = -1, verbose: bool = False):
        self.warmup_epochs = warmup_epochs
        self.cos_half_period = cos_half_period
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linearly increase the learning rate
            return [(base_lr * (self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]
        elif self.last_epoch < self.cos_half_period + self.warmup_epochs:
            # Cosine annealing phase
            cos_anneal_epoch = self.last_epoch - self.warmup_epochs
            cos_inner = (1 + np.cos(np.pi * cos_anneal_epoch / self.cos_half_period)) / 2
            return [self.min_lr + (base_lr - self.min_lr) * cos_inner for base_lr in self.base_lrs]
        else:
            return [self.min_lr]

class WarmupCosine(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, cos_half_period: int,
                 min_lr: float = 5e-7, last_epoch: int = -1, verbose: bool = False):
        self.warmup_epochs = warmup_epochs
        self.cos_half_period = cos_half_period
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linearly increase the learning rate
            return [(base_lr * (self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            cos_anneal_epoch = self.last_epoch - self.warmup_epochs
            cos_inner = (1 + np.cos(np.pi * cos_anneal_epoch / self.cos_half_period)) / 2
            return [self.min_lr + (base_lr - self.min_lr) * cos_inner for base_lr in self.base_lrs]
