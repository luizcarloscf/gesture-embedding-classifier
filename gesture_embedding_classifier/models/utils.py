import torch
from torch import nn


class Flatten(nn.Module):
    """
    A module to flatten the input tensor.
    """

    def __init__(self, keep_batch_dim: bool = True) -> None:
        """
        Constructor method for Flatten class.

        Parameters
        ----------
        keep_batch_dim : bool, optional
            If True, the batch dimension is kept (default is True).
        """
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to flatten the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Flattened tensor.
        """
        if self.keep_batch_dim:
            return x.reshape(x.size(0), -1)
        return x.reshape(-1)
