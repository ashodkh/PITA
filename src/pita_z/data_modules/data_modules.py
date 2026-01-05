import pytorch_lightning as pl
import torch
import h5py
import numpy as np

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_file: str = None,
        with_redshift: bool = False,
        with_features: bool = False,
        with_weights: bool = False,
        reddening_transform = None,
        load_ebv: bool = False,
        label_f: int = 1
    ):
        super().__init__()
        self.h5_file = h5py.File(path_file, 'r')
        self.with_redshift = with_redshift
        self.with_features = with_features
        self.with_weights = with_weights
        self.reddening_transform = reddening_transform
        self.load_ebv = load_ebv
        self.label_f = label_f
        
    def __len__(self) -> int:
        return len(self.h5_file['images'])
    
    def __getitem__(self, idx: int):
        image = self.h5_file['images'][idx]
        ebv = self.h5_file['ebvs'][idx] if self.load_ebv else None
        redshift = self.h5_file['redshifts'][idx] if self.with_redshift else None
        color_features = self.h5_file['dered_color_features'][idx] if self.with_features else 1
        redshift_weight = self.h5_file[f'use_redshift_{self.label_f}'][idx] if self.with_weights else 1

        # Apply reddening transformation if provided
        if self.reddening_transform:
            image = self.reddening_transform([image, ebv])

        if self.with_redshift:
            return image, redshift, redshift_weight, color_features
        else:
            return image

        
class ImagesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 1,
        train_size: float = 0.8,
        path_train: str = None,
        path_val: str = None,
        with_redshift: bool = False,
        with_features: bool = False,
        with_weights: bool = False,
        reddening_transform = None,
        load_ebv: bool = False,
        label_f: int = 1
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.path_train = path_train
        self.path_val = path_val
        self.with_redshift = with_redshift
        self.with_features = with_features
        self.with_weights = with_weights
        self.reddening_transform = reddening_transform
        self.load_ebv = load_ebv
        self.label_f = label_f
    
    def setup(self, stage):
        if self.path_val:
            self.images_train = self._create_dataset(self.path_train, self.label_f)
            self.images_val = self._create_dataset(self.path_val, 1)
        else:
            full_dataset = self._create_dataset(self.path_train)
            generator = torch.Generator().manual_seed(42)
            self.images_train, self.images_val = torch.utils.data.random_split(
                full_dataset,
                [self.train_size, 1 - self.train_size],
                generator=generator
            )
        
    def _create_dataset(self, path: str, label_f: int) -> ImagesDataset:
        """Helper to create a dataset with consistent parameters."""
        return ImagesDataset(
            path_file=path,
            with_redshift=self.with_redshift,
            with_features=self.with_features,
            with_weights=self.with_weights,
            reddening_transform=self.reddening_transform,
            load_ebv=self.load_ebv,
            label_f=label_f
        )
     
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the training dataloader."""
        return self._create_dataloader(self.images_train, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the validation dataloader."""
        return self._create_dataloader(self.images_val, shuffle=False)

    def _create_dataloader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> torch.utils.data.DataLoader:
        """Helper to create a DataLoader with consistent parameters."""
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            persistent_workers=True,
            pin_memory=torch.cuda.is_available(),
        )