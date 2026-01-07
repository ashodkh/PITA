This repo contains the Pytorch Lightning implementation of PITA (Photo-z Inference with a Triple-loss Algorithm).

PITA is a semi-supervised, image-based photometric redshift (photo-z) algorithm. It combines contrastive learning with color and redshift prediction losses, enabling simultaneous training with both unlabeled and labeled data.

<img width="1466" height="818" alt="semi_supervised_arch_new" src="https://github.com/user-attachments/assets/0aa5d73d-58e1-469b-83c5-9961d9320fcc" />

In our paper, we show that PITA outperforms traditional ML and fully-supervised CNN methods on HST [CANDELS survey](https://www.ipac.caltech.edu/project/candels) galaxies out to z ~ 3. 

# Running PITA
The `scripts` folder contains an example script for training PITA.

## Installation
PITA can be installed with pip:
```
pip install pita-z
```
Alternatively, you can clone the repo and run `pip install .` in the repo directory.

## Config files
All the hyperparameters and relevant file paths are contained in the `/configs/pita_default.yaml` file. To train PITA on your own dataset, the directories within the config file should be changed accordingly.

## Dataset
To use the default Pytorch Lightning data module in `src/pita_z/data_modules/data_modules.py`, your data should be stored in an hdf5 file. Along with images, photometric colors, and redshifts, the code relies on an additional redshift weight array `use_redshift` that should be 1 when the image has a redshift label, and 0 otherwise.

If your data is stored in another format, you can define your own dataset class and lightning data module. Additional information on lightning data modules can be found at [this link](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).

<!--
```
class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, path_file: str = None):
        super().__init__()
        self.h5_file = h5py.File(path_file, 'r')
        
    def __len__(self) -> int:
        return len(self.h5_file['images'])
    
    def __getitem__(self, idx: int):
        return self.h5_file['images'][idx]
    
class ImagesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 128, path_train: str = None, path_val: str = None):
        super().__init__()
        self.batch_size = batch_size
        self.path_train = path_train
        self.path_val = path_val

    def setup(self, stage):
          self.images_train = self._create_dataset(self.path_train, self.label_f)
          self.images_val = self._create_dataset(self.path_val, 1)
        
    def _create_dataset(self, path: str, label_f: int) -> ImagesDataset:
        return ImagesDataset(
            path_file=path
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
            num_workers=1,
            shuffle=shuffle,
            persistent_workers=True,
            pin_memory=torch.cuda.is_available(),
        )
```
-->
