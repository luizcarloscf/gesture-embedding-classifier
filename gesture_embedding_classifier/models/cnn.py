from typing import Literal, cast, Optional

import torch
from torch import nn
from torchvision.models.vgg import vgg19


class VGG19(nn.Module):
    """
    Custom VGG19 model for usually used for image classification. Here,
    it will be used to classify all stacked embeddings.
    """

    def __init__(
        self,
        num_class: int = 20,
        weights: Optional[Literal["IMAGENET1K_V1", "DEFAULT"]] = None,
    ) -> None:
        """
        Constructor method for VGG19 class.

        Parameters
        ----------
        num_class : int, optional
            Number of output classes (default is 20).
        in_channels : int, optional
            Number of input channels (default is 1).
        weights : Optional[Literal["IMAGENET1K_V1", "DEFAULT"]], optional
            Pre-trained weights to initialize the model (default is None).
        """
        super().__init__()
        self.conv_model = vgg19(weights=weights)
        conv_layers = cast(list, self.conv_model.features)
        conv_layers[0] = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        mlp_layers = cast(list, self.conv_model.classifier)
        mlp_layers[-1] = nn.Linear(
            in_features=4096,
            out_features=num_class,
        )
        self.soft = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the model.
        """
        x = self.conv_model.features(x)
        x = self.conv_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.conv_model.classifier(x)
        return self.soft(x)
