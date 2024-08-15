import logging

from torchinfo import summary
from lightning.pytorch import cli_lightning_logo

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from gesture_embedding_classifier.data.datamodule import EmbeddingsDataModule
from gesture_embedding_classifier.module import BaseModule


def cli_main():
    """
    Main function that sets up and runs the LightningCLI with the specified
    configurations for model, data module, and trainer.
    """
    log = logging.getLogger(__name__)
    cli = LightningCLI(
        seed_everything_default=True,
        model_class=BaseModule,
        datamodule_class=EmbeddingsDataModule,
        save_config_kwargs={
            "overwrite": False,
        },
        trainer_defaults={
            "devices": 1,
            "max_epochs": 200,
            "accelerator": "gpu",
            "callbacks": [
                ModelCheckpoint(
                    mode="max",
                    monitor="val_acc",
                ),
                LearningRateMonitor("epoch"),
            ],
            "enable_progress_bar": True,
        },
        run=False,
    )
    log.info("Model Summary: \n %s", summary(model=cli.model, input_size=(1, 256, 107)))
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
