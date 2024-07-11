from typing import Any, Dict, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Accuracy
from lightning.pytorch import LightningModule

from gesture_embedding_classifier.utils import (
    LRSchedulerType,
    OptimizerType,
    ModuleType,
)


class BaseModule(LightningModule):
    """
    A base LightningModule for training models with custom loss, optimizer and learning
    rate scheduler.
    """

    def __init__(
        self,
        loss_class: ModuleType,
        model_class: ModuleType,
        optimizer_class: OptimizerType = AdamW,
        lr_scheduler_class: LRSchedulerType = MultiStepLR,
        loss_kwargs: Dict[str, Any] = {},
        model_kwargs: Dict[str, Any] = {},
        optimizer_kwargs: Dict[str, Any] = {},
        lr_scheduler_kwargs: Dict[str, Any] = {},
    ) -> None:
        """
        Constructor method for BaseModule class.

        Parameters
        ----------
        loss_class : ModuleType
            Loss function class.
        model_class : ModuleType
            Model class.
        optimizer_class : OptimizerType, optional
            Optimizer class (default is AdamW).
        lr_scheduler_class : LRSchedulerType, optional
            Learning rate scheduler class (default is MultiStepLR).
        loss_kwargs : Dict[str, Any], optional
            Additional keyword arguments for the loss function (default is {}).
        model_kwargs : Dict[str, Any], optional
            Additional keyword arguments for the model (default is {}).
        optimizer_kwargs : Dict[str, Any], optional
            Additional keyword arguments for the optimizer (default is {}).
        lr_scheduler_kwargs : Dict[str, Any], optional
            Additional keyword arguments for the learning rate scheduler (default is {}).
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model_class(**model_kwargs)
        self.loss_module = loss_class(**loss_kwargs)
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.example_input_array = torch.zeros((1, 1, 107, 255), dtype=torch.float32)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=20)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=20)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=20)

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
        return self.model(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Batch of data (inputs and labels).
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Computed loss for the batch.
        """
        inputs, labels = batch
        outputs = self.model(inputs)

        loss = self.loss_module(outputs, labels)
        # accuracy = (outputs.argmax(dim=-1) == labels).float().mean()
        preds = outputs.argmax(dim=-1)
        self.train_accuracy(preds, labels)

        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """
        Validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Batch of data (inputs and labels).
        batch_idx : int
            Index of the batch.
        """
        inputs, labels = batch
        outputs = self.model(inputs)

        loss = self.loss_module(outputs, labels)
        # accuracy = (outputs.argmax(dim=-1) == labels).sum().item()
        preds = outputs.argmax(dim=-1)
        self.val_accuracy(preds, labels)

        self.log(
            "val_acc",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """
        Test step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Batch of data (inputs and labels).
        batch_idx : int
            Index of the batch.
        """
        inputs, labels = batch
        outputs = self.model(inputs)

        # acc = (labels == outputs.argmax(dim=-1)).sum().item()
        preds = outputs.argmax(dim=-1)
        self.test_accuracy(preds, labels)

        self.log(
            "test_acc",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = self.optimizer_class(
            self.model.parameters(),
            **self.optimizer_kwargs,
        )
        lr_scheduler = self.lr_scheduler_class(
            optimizer,
            **self.lr_scheduler_kwargs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }
