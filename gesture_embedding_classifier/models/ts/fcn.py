import torch
from torch import nn

from ..utils import Flatten


class ConvBlock(nn.Module):
    """
    A convolutional block consisting of a convolutional layer, batch
    normalization, and ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        """
        Constructor method for ConvBlock class.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input signal.
        out_channels : int
            Number of channels produced by the convolution.
        kernel_size : int
            Size of the convolving kernel.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the model.
        """
        return self.layers(x)


class FCN1D(nn.Module):
    """
    Fully Convolutional Network (FCN) considering embeddings as multidimensional
    time series. Reference: https://arxiv.org/abs/1611.06455
    """

    def __init__(
        self,
        num_classes: int = 20,
        in_channels: int = 256,
    ) -> None:
        """
        Constructor method for FCN1D class.

        Parameters
        ----------
        num_classes : int, optional
            Number of output classes (default is 20).
        in_channels : int, optional
            The number of input features (default is 256).
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=128, kernel_size=8),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=5),
            ConvBlock(in_channels=256, out_channels=128, kernel_size=3),
        )
        self.flatten = Flatten(keep_batch_dim=True)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=num_classes),
        )
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Dimentions should be [B, C, L], where B is the number of batches,
            C is the number of features and L is the sequence lenght.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the model.
        """
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        x = self.soft(x)
        return x
