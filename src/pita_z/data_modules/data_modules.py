import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import h5py
import numpy as np
from joblib import load
from pathlib import Path

class ImagesDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that uses h5py and memory mapping to read images, redshift, and other metadata.
    MW extinction can be removed by using a reddening transform.

    Attributes:
        path_file (str): Path to h5py file.
        with_redshift (bool): If True, reads and returns redshift.
        with_features (bool): If True, reads and returns photometric features (dereddened colors and magnitudes).
        with_weights (bool): If True, reads and returns weights associated with redshifts. 
                             The weights are 1 if objects have a redshift label, and 0 otherwise.
        reddening_transform (class): Removes MW extinction using E(B-V).
        load_ebv (bool): If True, loads and returns E(B-V) values.
        label_f (int): Fraction of redshift labels to use when training (1/label_f is the fraction).
                       Loads different redshift weights based on the fraction.
    """
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
    """
    PyTorch Lightning data module that creates ImagesDatasets and uses PyTorch dataloaders.

    Attributes:
        batch_size (int): Batch size.
        num_workers (int): number of workers used in DataLoader.
        train_size (float): Fraction of data used for training. Only used if path_val is None.
        path_train (str): Path to h5py file that contains the training set.
        path_val (str): Path to h5py file that contains the validation set.
        with_redshift (bool): If True, reads and returns redshift.
        with_features (bool): If True, reads and returns photometric features (dereddened colors and magnitudes).
        with_weights (bool): If True, reads and returns weights associated with redshifts.
                             The weights are 1 if objects have a redshift label, and 0 otherwise.
        reddening_transform (class): Removes MW extinction using E(B-V).
        load_ebv (bool): If True, loads and returns E(B-V) values.
        label_f (int): Fraction of redshift labels to use when training (1/label_f is the fraction).
                       Loads different redshift weights based on the fraction.
        
    """
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
     
    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.images_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.images_val, shuffle=False)

    def _create_dataloader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> DataLoader:
        """Helper to create a DataLoader with consistent parameters."""
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            persistent_workers=True,
            pin_memory=torch.cuda.is_available(),
        )

class CalpitPhotometryDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that uses h5py and memory mapping to read photometry, redshift, and other metadata.
    Takes in PITs and returns them per object. PITs are the outputs or the 'true y' values that are used to calculate the Cal-PIT loss.
    Assumes that photometry are corrected for MW extinction. Their column name in the dataset is a variable that can be set with 'feature_name'.

    Attributes:
        path_file (str): Path to h5py file.
        feature_name (str): Column name of features in the dataset.
        pit (np.array): PIT values of the outputs.
        scaler_path (str): Path to scikit-learn StandardScaler for rescaling the features.
    """
    def __init__(self, path_file=None, feature_name=None, pit=None, scaler_path=None):
        if Path(path_file).suffix == '.hdf5':
            self.file = h5py.File(path_file, 'r')
        self.feature_name = feature_name
        self.pit = pit
        self.scaler = load(scaler_path) if scaler_path else None

    def __len__(self):
        key = list(self.file.keys())[0]
        return len(self.file[key])

    def __getitem__(self, idx):
        x = self.file[self.feature_name][idx].astype(np.float32)
        if self.scaler:
            x = self.scaler.transform(x.reshape(1,-1))
        #x = torch.tensor(x.squeeze())
        #y = torch.tensor(self.pit[idx])
        
        return x.squeeze(), self.pit[idx], self.file['redshifts'][idx].astype(np.float32)
    
class CalpitPhotometryDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module that creates CalpitPhotometryDataset and uses PyTorch dataloaders.

    Attributes:
        path_train (str): Path to h5py file that contains the training set.
        feature_name (str): Column name of features in the dataset.
        pit_train (np.array): PIT values for training set.
        scaler_path (str): Path to scikit-learn StandardScaler for rescaling the features.
        path_val (str): Path to h5py file that contains the validation set.
        pit_val (np.array): PIT values for training set.
        batch_size (int): Batch size.
        num_workers (int): number of workers used in DataLoader.
    """
    def __init__(
        self,
        path_train: str=None,
        feature_name: str=None,
        pit_train=None,
        scaler_path: str=None,
        path_val: str=None,
        pit_val=None,
        batch_size: int=None,
        num_workers: int=None
    ):
        super().__init__()
        self.path_train = path_train
        self.feature_name = feature_name
        self.pit_train = pit_train
        self.scaler_path = scaler_path
        self.path_val = path_val
        self.pit_val = pit_val
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self.ds_train = CalpitPhotometryDataset(
            path_file=self.path_train,
            feature_name=self.feature_name,
            pit=self.pit_train,
            scaler_path=self.scaler_path
        )

        self.ds_val = CalpitPhotometryDataset(
            path_file=self.path_val,
            feature_name=self.feature_name,
            pit=self.pit_val,
            scaler_path=self.scaler_path
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, pin_memory=torch.cuda.is_available())

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=torch.cuda.is_available())

class CalpitImagesDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that uses h5py and memory mapping to read images, redshift, and other metadata.
    Takes in PITs and returns them per object. PITs are the outputs or the 'true y' values that are used to calculate the Cal-PIT loss.
    MW extinction can be removed by using a reddening transform.

    Attributes:
        path_file (str): Path to h5py file.
        pit (np.array): PIT values of the outputs.
        with_redshift (bool): If True, reads and returns redshift.
        with_features (bool): If True, reads and returns photometric features (dereddened colors and magnitudes).
        with_weights (bool): If True, reads and returns weights associated with redshifts.
                             The weights are 1 if objects have a redshift label, and 0 otherwise.
        reddening_transform (class): Removes MW extinction using E(B-V).
        load_ebv (bool): If True, loads and returns E(B-V) values.
        label_f (int): Fraction of redshift labels to use when training (1/label_f is the fraction).
                       Loads different redshift weights based on the fraction.
    """
    def __init__(
        self,
        path_file: str = None,
        pit: np.ndarray = None,
        with_redshift: bool = False,
        with_features: bool = False,
        with_weights: bool = False,
        reddening_transform = None,
        load_ebv: bool = False,
        label_f: int = 1
    ):
        super().__init__()
        if Path(path_file).suffix == '.hdf5':
            self.file = h5py.File(path_file, 'r')
        self.pit = pit
        self.with_redshift = with_redshift
        self.with_features = with_features
        self.with_weights = with_weights
        self.reddening_transform = reddening_transform
        self.load_ebv = load_ebv
        self.label_f = label_f

    def __len__(self) -> int:
        return len(self.file['images'])
    
    def __getitem__(self, idx: int):
        image = self.file['images'][idx]
        ebv = self.file['ebvs'][idx] if self.load_ebv else None
        redshift = self.file['redshifts'][idx] if self.with_redshift else None
        color_features = self.file['dered_color_features'][idx] if self.with_features else 1
        redshift_weight = self.file[f'use_redshift_{self.label_f}'][idx] if self.with_weights else 1

        # Apply reddening transformation if provided
        if self.reddening_transform:
            image = self.reddening_transform([image, ebv])

        if self.with_redshift:
            return image.astype(np.float32), self.pit[idx], redshift.astype(np.float32), redshift_weight.astype(np.float32), color_features.astype(np.float32)
        else:
            return image.astype(np.float32), self.pit[idx]

class CalpitImagesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module that creates CalpitImagesDataset and uses PyTorch dataloaders.

    Attributes:
        batch_size (int): Batch size.
        num_workers (int): number of workers used in DataLoader.
        path_train (str): Path to h5py file that contains the training set.
        pit_train (np.array): PIT values for training set.
        path_val (str): Path to h5py file that contains the validation set.
        pit_val (np.array): PIT values for validation set.
        with_redshift (bool): If True, reads and returns redshift.
        with_features (bool): If True, reads and returns photometric features (dereddened colors and magnitudes).
        with_weights (bool): If True, reads and returns weights associated with redshifts.
                             The weights are 1 if objects have a redshift label, and 0 otherwise.
        reddening_transform (class): Removes MW extinction using E(B-V).
        load_ebv (bool): If True, loads and returns E(B-V) values.
        label_f (int): Fraction of redshift labels to use when training (1/label_f is the fraction).
                       Loads different redshift weights based on the fraction.
    """
    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 1,
        path_train: str = None,
        pit_train: np.ndarray = None,
        path_val: str = None,
        pit_val: np.ndarray = None,
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
        self.path_train = path_train
        self.pit_train = pit_train
        self.path_val = path_val
        self.pit_val = pit_val
        self.with_redshift = with_redshift
        self.with_features = with_features
        self.with_weights = with_weights
        self.reddening_transform = reddening_transform
        self.load_ebv = load_ebv
        self.label_f = label_f


    def setup(self, stage):
        self.images_train = self._create_dataset(
            self.path_train,
            self.pit_train,
            self.label_f
        )
        self.images_val = self._create_dataset(
            self.path_val,
            self.pit_val,
            1
        )

    def _create_dataset(self, path: str, pit: np.ndarray, label_f: int):
        return CalpitImagesDataset(
            path_file=path,
            pit=pit,
            with_redshift=self.with_redshift,
            with_features=self.with_features,
            with_weights=self.with_weights,
            reddening_transform=self.reddening_transform,
            load_ebv=self.load_ebv,
            label_f=label_f
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.images_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.images_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=torch.cuda.is_available()
        )







    