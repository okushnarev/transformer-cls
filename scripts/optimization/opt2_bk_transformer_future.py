import json
import shutil
from argparse import ArgumentParser
from functools import partial
from typing import Any
import sys
from pathlib import Path
import logging

import joblib
import optuna

# Add project root to PATH
project_root = str(Path.cwd())
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.transformer import LitTransformerRegression
from src.datamodules.belyaev_kushnarev import BelyaevKushnarevFutureDataModule
from src.optimization import optimize


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--study-dir', type=Path, default=Path('experiments/optimization/default'))
    parser.add_argument('--overwrite', action='store_true', help='Overwrite study folder')
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--segment-size', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--features', type=str, nargs='+',
                        default=['m1vel',
                                 'm2vel',
                                 'm3vel',
                                 'm1cur',
                                 'm2cur',
                                 'm3cur',
                                 'Ke', ]
                        )
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=1800, help='Timeout to stop trials in seconds')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.study_dir.exists():
        logging.warning('Directory for this study exists: {}\n'.format(args.study_dir))
        if args.overwrite:
            shutil.rmtree(args.study_dir)
        else:
            logging.warning('Cannot proceed\n'
                            'Use different name for the study\n'
                            'or\n'
                            'Use --overwrite flag to overwrite the directory')
            return

    args.study_dir.mkdir(parents=True)

    model_class = LitTransformerRegression

    datamodule_class = partial(
        BelyaevKushnarevFutureDataModule,
        features=args.features,
        mode='reg',
        segment_size=args.segment_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    def suggest_params(trial: optuna.trial.Trial) -> dict[str, Any]:
        sequence_length = trial.suggest_int('sequence_length', low=5, high=90, log=True)

        n_in_layers = trial.suggest_int('n_in_layers', low=1, high=5, log=True)
        in_layers = [2 ** trial.suggest_int(f'in_layer_{i}_pow', low=3, high=8, log=True) for i in range(n_in_layers)]

        n_out_reg_layers = trial.suggest_int('n_out_reg_layers', low=1, high=5, log=True)
        out_reg_layers = [2 ** trial.suggest_int(f'out_reg_layer_{i}_pow', low=3, high=8, log=True) for i in
                          range(n_out_reg_layers)]

        d_model = 2 ** trial.suggest_int('d_model_pow', low=4, high=8, log=True)
        dim_feedforward = 2 ** trial.suggest_int('dim_feedforward_pow', low=4, high=9, log=True)

        hparams = {
            'input_dim':               len(args.features),
            'out_dim_reg':             len(args.features),
            'sequence_length':         sequence_length,
            'in_mlp_hidden_dims':      in_layers,
            'out_reg_mlp_hidden_dims': out_reg_layers,
            'd_model':                 d_model,
            'dim_feedforward':         dim_feedforward,
        }

        return hparams

    report = optimize(
        log_dir=args.study_dir,
        model_class=model_class,
        datamodule_class=datamodule_class,
        suggest_hparams=suggest_params,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    best_params = {}
    for param, value in report.best_trial.params.items():
        if (sfx := '_pow') in param:
            best_params[param.replace(sfx, '')] = 2 ** value
        else:
            best_params[param] = value
    print('Best params:')
    for k, v in best_params.items():
        print(f'\t{k}: {v}')

    study_path = args.study_dir / 'study.joblib'
    joblib.dump(report.study, study_path)
    logging.info('Study saved to {}'.format(study_path))

    params_path = args.srudy_dir / 'best_params.json'
    with open(params_path, 'w') as file:
        json.dump(best_params, file)
    logging.info('Best params saved to {}'.format(params_path))


if __name__ == '__main__':
    main()
