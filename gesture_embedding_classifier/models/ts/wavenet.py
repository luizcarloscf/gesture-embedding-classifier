from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..utils import Flatten


class CausalConv1d(nn.Conv1d):
    """
    Applies a 1D causal convolution over an input signal composed of several
    input planes. Reference: https://github.com/pytorch/pytorch/issues/1333
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """
        Constructor method for CausalConv1d class.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input signal.
        out_channels : int
            Number of channels produced by the convolution.
        kernel_size : int
            Size of the convolving kernel.
        stride : int, optional
            Stride of the convolution (default is 1).
        dilation : int, optional
            Spacing between kernel elements (default is 1).
        groups : int, optional
            Number of blocked connections from input channels to output channels (default is 1).
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the causal convolutional block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the model.
        """
        x = F.pad(x, (self.__padding, 0))
        x = super().forward(x)
        return x


class CausalDilatedResidual(nn.Module):
    """
    A module implementing a causal dilated residual block with separate
    skip and residual connections.
    """

    def __init__(
        self,
        num_residual_channels: int,
        num_skip_channels: int,
        dilation: int,
    ) -> None:
        """
        Constructor method for CausalDilatedResidual class.

        Parameters
        ----------
        num_residual_channels : int
            Number of channels in the residual connections.
        num_skip_channels : int
            Number of channels in the skip connections.
        dilation : int
            Dilation factor for the causal convolutions.
        """
        super().__init__()
        self.num_residual_channels = num_residual_channels
        self.num_skip_channels = num_skip_channels
        self.dilation = dilation
        self.kernel_size = 2

        self.conv_sigmoid = nn.Sequential(
            CausalConv1d(
                in_channels=num_residual_channels,
                out_channels=num_residual_channels,
                kernel_size=self.kernel_size,
                dilation=dilation,
            ),
            nn.Sigmoid(),
        )
        self.conv_tanh = nn.Sequential(
            CausalConv1d(
                in_channels=num_residual_channels,
                out_channels=num_residual_channels,
                kernel_size=self.kernel_size,
                dilation=dilation,
            ),
            nn.Tanh(),
        )
        self.conv_skip = nn.Conv1d(
            in_channels=num_residual_channels,
            out_channels=num_skip_channels,
            kernel_size=1,
        )
        self.conv_residual = nn.Conv1d(
            in_channels=num_residual_channels,
            out_channels=num_residual_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the causal dilated residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the skip connection tensor and the residual
            output tensor.
        """
        u = self.conv_sigmoid(x) * self.conv_tanh(x)
        s = self.conv_skip(u)
        out = self.conv_residual(u)
        out = out + x
        return s, out


class ResidualStack(torch.nn.Module):
    """
    A stack of causal dilated residual blocks.
    """

    def __init__(
        self,
        num_residual_channels: int,
        num_skip_channels: int,
        layer_size: int = 5,
        stack_size: int = 5,
    ) -> None:
        """
        Constructor method for ResidualStack class.

        Parameters
        ----------
        num_residual_channels : int
            Number of channels in the residual connections.
        num_skip_channels : int
            Number of channels in the skip connections.
        layer_size : int, optional
            Number of layers in each stack (default is 5).
        stack_size : int, optional
            Number of stacks (default is 5).
        """
        super().__init__()
        self.layer_size = layer_size
        self.stack_size = stack_size
        self.res_blocks = self._stack_residual_blocks(
            num_residual_channels=num_residual_channels,
            num_skip_channels=num_skip_channels,
        )

    @staticmethod
    def _residual_block(
        num_residual_channels: int,
        num_skip_channels: int,
        dilation: int,
    ) -> CausalDilatedResidual:
        """
        Creates a causal dilated residual block.

        Parameters
        ----------
        num_residual_channels : int
            Number of channels in the residual connections.
        num_skip_channels : int
            Number of channels in the skip connections.
        dilation : int
            Dilation factor for the causal convolutions.

        Returns
        -------
        CausalDilatedResidual
            A causal dilated residual block.
        """
        block = CausalDilatedResidual(
            num_residual_channels=num_residual_channels,
            num_skip_channels=num_skip_channels,
            dilation=dilation,
        )
        if torch.cuda.device_count() > 1:
            block = torch.nn.DataParallel(block)
        if torch.cuda.is_available():
            block.cuda()
        return block

    def _build_dilations(self) -> List[float]:
        """
        Builds a list of dilation factors.

        Returns
        -------
        List[int]
            List of dilation factors.
        """
        dilations = []
        for _ in range(0, self.stack_size):
            for l in range(0, self.layer_size):
                dilations.append(2**l)
        return dilations

    def _stack_residual_blocks(
        self,
        num_residual_channels: int,
        num_skip_channels: int,
    ) -> List[CausalDilatedResidual]:
        """
        Stacks the residual blocks.

        Parameters
        ----------
        num_residual_channels : int
            Number of channels in the residual connections.
        num_skip_channels : int
            Number of channels in the skip connections.

        Returns
        -------
        List[CausalDilatedResidual]
            List of stacked residual blocks.
        """
        res_blocks = []
        dilations = self._build_dilations()
        for dilation in dilations:
            block = self._residual_block(
                num_residual_channels=num_residual_channels,
                num_skip_channels=num_skip_channels,
                dilation=dilation,
            )
            res_blocks.append(block)
        return res_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the stack of residual blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with stacked skip connections.
        """
        output = x
        skip_connections = []
        for res_block in self.res_blocks:
            output, skip = res_block(output)
            skip_connections.append(skip)
        return torch.stack(skip_connections)


class DenseLayer(torch.nn.Module):
    """
    A dense layer block with Conv1d, ReLU, AdaptiveAvgPool1d, and
    LogSoftmax activations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """
        Constructor method for DenseLayer class.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Number of channels in the output tensor.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(keep_batch_dim=True),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dense layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the model.
        """
        x = self.layers(x)
        return x


class WaveNet(torch.nn.Module):
    """
    A WaveNet model for sequence modeling with causal convolutions and
    residual blocks.
    """

    def __init__(
        self,
        layer_size: int,
        stack_size: int,
        in_channels: int,
        res_channels: int,
        num_classes: int = 20,
    ):
        """
        Constructor method for WaveNet class.

        Parameters
        ----------
        layer_size : int
            Number of layers in each residual stack.
        stack_size : int
            Number of stacks of residual layers.
        in_channels : int
            Number of input channels.
        res_channels : int
            Number of residual channels.
        num_classes : int, optional, default=20
            Number of output classes.
        """
        super().__init__()
        self.causal = CausalConv1d(
            in_channels=in_channels,
            out_channels=res_channels,
            kernel_size=2,
        )
        self.res_stack = ResidualStack(
            num_residual_channels=res_channels,
            num_skip_channels=in_channels,
            layer_size=layer_size,
            stack_size=stack_size,
        )
        self.dense = DenseLayer(
            in_channels=in_channels,
            out_channels=num_classes,
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape should be (batch_size, channels, sequence_length, num_features).
            Where, channels = 1.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the model.
        """
        x = self.causal(x)
        skip_connections = self.res_stack(x)
        x = torch.sum(skip_connections, dim=0)
        x = self.dense(x)
        return x
