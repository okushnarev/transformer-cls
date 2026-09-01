import inspect
from typing import Callable, Any

import optuna
import lightning as L
from optuna.integration import PyTorchLightningPruningCallback


def optimize(
        model_class: L.LightningModule,
        datamodule_class: L.LightningDataModule,
        suggest_hparams: Callable[[optuna.trial.Trial], dict[str, Any]],

        n_trials=100,
        timeout=600,

        max_epochs: int = 30,
        limit_val_batches: float = 1,
        monitor_metric: str = 'overall/val_loss',
) -> None:
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

    print(f'Number of finished trials: {len(study.trials)}')

    print('Best trial:')
    trial = study.best_trial

    print(f'  Value: {trial.value}')

    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')


if __name__ == '__main__':
    from functools import partial
    from src.models.transformer import LitTransformer
    from src.datamodules.belyaev_kushnarev import BelyaevKushnarevDataModule

    model_class = LitTransformer
    features = [
        'm1vel',
        'm2vel',
        'm3vel',
        'm1cur',
        'm2cur',
        'm3cur',
        'Ke',
    ]

    datamodule_class = partial(
        BelyaevKushnarevDataModule,
        features=features,
        mode='cls+reg',
        segment_size=100,
        batch_size=1000,
        num_workers=2,
    )


    def suggest_params(trial: optuna.trial.Trial) -> dict[str, Any]:
        sequence_length = trial.suggest_int('sequence_length', low=5, high=90, log=True)

        n_in_layers = trial.suggest_int('n_in_layers', low=1, high=5, log=True)
        in_layers = [trial.suggest_int(f'in_layer_{i}', low=8, high=256, log=True) for i in range(n_in_layers)]

        n_out_reg_layers = trial.suggest_int('n_out_reg_layers', low=1, high=5, log=True)
        out_reg_layers = [trial.suggest_int(f'out_reg_layer_{i}', low=8, high=256, log=True) for i in
                          range(n_out_reg_layers)]

        d_model = trial.suggest_int('d_model', low=16, high=256, log=True)
        dim_feedforward = trial.suggest_int('dim_feedforward', low=16, high=512, log=True)

        hparams = {
            'input_dim':               len(features),
            'sequence_length':         sequence_length,
            'in_mlp_hidden_dims':      in_layers,
            'out_reg_mlp_hidden_dims': out_reg_layers,
            'd_model':                 d_model,
            'dim_feedforward':         dim_feedforward,
        }

        return hparams


    optimize(
        model_class=model_class,
        datamodule_class=datamodule_class,
        suggest_hparams=suggest_params
    )