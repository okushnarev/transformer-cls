import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import optuna
import lightning as L
from optuna.integration import PyTorchLightningPruningCallback


@dataclass
class OptimizationReport:
    study: optuna.Study
    best_trial: optuna.Trial
    best_value: float
    n_finished_trials: int


def optimize(
        model_class: L.LightningModule,
        datamodule_class: L.LightningDataModule,
        suggest_hparams: Callable[[optuna.trial.Trial], dict[str, Any]],
        log_dir: Path,

        n_trials=100,
        timeout=600,

        max_epochs: int = 30,
        limit_val_batches: float = 1,
        monitor_metric: str = 'overall/val_loss',

) -> OptimizationReport:
    def objective(
            trial: optuna.trial.Trial,
    ) -> float:
        hparams = suggest_hparams(trial)
        model = model_class(**hparams)

        dm_sig = inspect.signature(datamodule_class)
        dm_params = {}
        for param, value in hparams.items():
            if param in dm_sig.parameters:
                dm_params[param] = value
        dm = datamodule_class(**dm_params)

        trainer = L.Trainer(
            default_root_dir=log_dir,
            logger=True,
            limit_val_batches=limit_val_batches,
            enable_checkpointing=False,
            max_epochs=max_epochs,
            accelerator='auto',
            devices=1,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor=monitor_metric)],
        )
        trainer.logger.log_hyperparams(hparams)
        trainer.fit(model, datamodule=dm)

        return trainer.callback_metrics[monitor_metric].item()

    pruner = optuna.pruners.HyperbandPruner(
        max_resource=max_epochs,
    )

    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
    )

    return OptimizationReport(
        study=study,
        best_trial=study.best_trial,
        best_value=study.best_trial.value,
        n_finished_trials=len(study.trials),
    )
