import os
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset


class EmbeddingsDataset(Dataset):
    """
    Custom Dataset class for loading graph neural network (GNN) embeddings and
    their corresponding labels.
    """

    def __init__(
        self,
        features_dir: str,
        labels_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Constructor method for EmbeddingsDataset class.

        Parameters
        ----------
        features_dir : str
            Directory where the feature files are stored.
        labels_file : str
            Path to the file containing the labels.
        transform : Optional[Callable]
            Optional transform to be applied on a sample (default is None).
        target_transform : Optional[Callable]
            Optional transform to be applied on the label (default is None).
        """
        self._transform = transform
        self._label_transform = target_transform
        self._features_dir = features_dir
        self._labels = torch.load(
            labels_file,
            map_location=None if torch.cuda.is_available() else torch.device("cpu"),
        )

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        length : int
            Number of samples in the dataset.
        """
        return self._labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns the sample and label at the specified index.

        Parameters
        ----------
        idx : int
            Index of the sample to be fetched.

        Returns
        -------
        feature : torch.Tensor
            The feature tensor corresponding to the specified index.
        label : int
            The label corresponding to the specified index.
        """
        feature_name = os.path.join(self._features_dir, f"{idx}.pth")
        feature = torch.load(
            feature_name,
            map_location=None if torch.cuda.is_available() else torch.device("cpu"),
        )
        label = self._labels[idx]
        if self._transform:
            feature = self._transform(feature)
        if self._label_transform:
            label = self._label_transform(label)
        return feature, label
