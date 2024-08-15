import torch
from torch import nn

from .fcn import ConvBlock
from ..utils import Flatten


class ResNetBlock(nn.Module):
    """
    A residual block consisting of a 3 convolutional blocks with kernel with
    size 8, 5, 3.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """
        Constructor method for ResNetBlock class.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input signal.
        out_channels : int
            Number of channels produced by the convolution.
        """
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=8,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=5,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
            ),
        )

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding="same",
                ),
                nn.BatchNorm1d(num_features=out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the resnet block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the model.
        """
        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


class ResNet1D(nn.Module):
    """
    Resnet Convolutional Network considering embeddings as multidimensional
    time series. Reference: https://arxiv.org/abs/1611.06455
    """

    def __init__(
        self,
        in_channels: int = 256,
        mid_channels: int = 64,
        num_classes: int = 1,
    ) -> None:
        """
        Constructor method for Resnet1D class.

        Parameters
        ----------
        num_classes : int, optional
            Number of output classes (default is 20).
        num_features : int, optional
            The number of input features (default is 256).
        dropout : float, optional
            Dropout probability (default is 0.5).
        """
        super().__init__()
        self.layers = nn.Sequential(
            ResNetBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
            ),
            ResNetBlock(
                in_channels=mid_channels,
                out_channels=mid_channels * 2,
            ),
            ResNetBlock(
                in_channels=mid_channels * 2,
                out_channels=mid_channels * 2,
            ),
        )
        self.flatten = Flatten(keep_batch_dim=True)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Linear(mid_channels * 2, num_classes)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Dimentions should be [B, C, L], where B is the number of batches,
            C is the number of features and L is the sequence lenght.
            Where, channels = 1.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the model.
        """
        x = self.layers(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        x = self.soft(x)
        return x
