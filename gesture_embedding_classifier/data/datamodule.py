import os

from lightning import LightningDataModule
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ..transforms import AddGaussianNoise, Gain, Pad, Reverse, TrimZeros
from .dataset import EmbeddingsDataset


class EmbeddingsDataModule(LightningDataModule):
    """
    LightningDataModule for loading embeddings datasets.
    """

    def __init__(
        self,
        dataset_path: str = "./data",
        batch_size: int = 32,
        num_workers: int = 3,
    ) -> None:
        """
        Constructor for EmbeddingsDataModule class.

        Parameters
        ----------
        dataset_path : str, optional
            Path to the dataset directory (default is "./data").
        batch_size : int, optional
            Batch size for data loaders (default is 32).
        num_workers : int, optional
            Number of workers for data loading (default is 3).
        """
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train_dataset = EmbeddingsDataset(
            features_dir=os.path.join(dataset_path, "train"),
            labels_file=os.path.join(dataset_path, "train", "labels.pth"),
            transform=transforms.Compose(
                [
                    TrimZeros(),
                    transforms.RandomApply(nn.ModuleList([AddGaussianNoise(scale=0.05)]), p=0.5),
                    transforms.RandomApply(nn.ModuleList([Gain(gain=1.05)]), p=0.3),
                    transforms.RandomApply(nn.ModuleList([Reverse()]), p=0.2),
                    Pad(size=107, value=0),
                ],
            ),
            target_transform=None,
        )
        self._val_dataset = EmbeddingsDataset(
            features_dir=os.path.join(dataset_path, "val"),
            labels_file=os.path.join(dataset_path, "val", "labels.pth"),
            transform=None,
            target_transform=None,
        )
        self._test_dataset = EmbeddingsDataset(
            features_dir=os.path.join(dataset_path, "test"),
            labels_file=os.path.join(dataset_path, "test", "labels.pth"),
            transform=None,
            target_transform=None,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training data loader.

        Returns
        -------
        DataLoader
            DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation data loader.

        Returns
        -------
        DataLoader
            DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset=self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test data loader.

        Returns
        -------
        DataLoader
            DataLoader for the test dataset.
        """
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            pin_memory=True,
        )
