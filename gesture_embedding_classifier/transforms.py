import torch
from torch import nn
from torch import Tensor


class TrimZeros(nn.Module):
    """
    Trims trailing columns of zeros from sequences.
    """

    def __init__(self) -> None:
        """
        Constructor method for TrimZeros class.
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the TrimZeros block.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [number_of_features, sequence_length].

        Returns
        -------
        Tensor
            Tensor with trailing zero columns removed. The shape will be
            [number_of_features, trimmed_sequence_length].

        Notes
        -----
        The method checks each column of the input tensor in reverse order
        (from the last column to the first) and stops when it finds a column
        that is not entirely zeros.

        If all columns are zeros, the tensor is returned unchanged.
        """
        sequence_length = x.shape[1]
        for col_idx in reversed(range(sequence_length)):
            if not torch.all(x[:, col_idx] == 0):
                return x[:, : col_idx + 1]
        return x[:, :]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Pad(nn.Module):
    """
    Pads sequences to a specified length.
    """

    def __init__(self, size: int, value: int = 0) -> None:
        """
        Constructor method for Pad class.

        Parameters
        ----------
        size : int
            The target size (length) of the sequences after padding.
        value : int, optional
            The value to use for padding.
        """
        super().__init__()
        self.size = size
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the pad block.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [number_of_features, sequence_length].

        Returns
        -------
        Tensor
            Padded tensor with shape [number_of_features, target_size].

        Raises
        ------
        ValueError
            If the computed padding size is smaller than the current sequence length.
        """
        sequence_lenght = x.shape[1]
        pad_size = self.size - sequence_lenght
        if pad_size < 0:
            raise ValueError("Computed padding size smaller than sequence lenght")
        return nn.functional.pad(
            x,
            pad=(0, pad_size),
            mode="constant",
            value=self.value,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AddGaussianNoise(nn.Module):
    """
    Add gaussian noise to multidimentional sequences.
    """

    def __init__(self, scale: float = 0.1):
        """
        Constructor method for AddGaussianNoise class.

        Parameters
        ----------
        scale : float, optional, default=0.1
            The scale factor for the Gaussian noise. It determines the standard deviation
            of the noise relative to the input. Must be between 0 and 1.
        """
        super().__init__()
        assert 0 <= scale <= 1
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the AddGaussianNoise block.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [number_of_features, sequence_length].

        Returns
        -------
        Tensor
            Output tensor with noise and shape [number_of_features, sequence_length].
        """
        std = torch.std(x, dim=1, keepdim=True) * self.scale
        noise = (std**0.5) * torch.randn_like(x)
        x = x + noise
        return x


class Gain(nn.Module):
    """
    Apply gain multidimentional sequences.
    """

    def __init__(self, gain: float = 1.1):
        """
        Constructor method for Gain class.

        Parameters
        ----------
        gain : float, optional, default=0.1
            The scale factor
        """
        super().__init__()
        self.gain = gain

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Gain block.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [number_of_features, sequence_length].

        Returns
        -------
        Tensor
            Output amplified tensor and shape [number_of_features, sequence_length].
        """
        x = x * self.gain
        return x


class Reverse(nn.Module):
    """
    Reverse multidimentional sequences.
    """

    def __init__(self):
        """
        Constructor method for Reverse class.
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Reverse block.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [number_of_features, sequence_length].

        Returns
        -------
        Tensor
            Output amplified tensor and shape [number_of_features, sequence_length].
        """
        x = torch.flip(x, dims=(-1,))
        return x
