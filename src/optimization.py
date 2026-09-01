from typing import Callable, Any

import optuna
import lightning as L
from optuna.integration import PyTorchLightningPruningCallback


def objective(
        trial: optuna.trial.Trial,
        model_class: L.LightningModule,
        datamodule: L.LightningDataModule,
        suggest_hparams: Callable[[], dict[str, Any]],
        **trainer_kwargs
) -> float:
    hparams = suggest_hparams()
    model = model_class(**hparams)

    trainer = L.Trainer(
        logger=True,
        limit_val_batches=trainer_kwargs.get('limit_val_batches', 1),
        enable_checkpointing=False,
        max_epochs=trainer_kwargs.get('max_epochs', 20),
        accelerator='auto',
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='overall/val_loss')],
    )
    trainer.logger.log_hyperparams(hparams)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics['overall/val_loss'].item()
