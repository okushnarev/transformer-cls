import pydoc
import shutil
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import classification_report, mean_absolute_error as mae, r2_score as r2, \
    root_mean_squared_error as rmse
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Add project root to PATH
project_root = str(Path.cwd())
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datamodules.base import BaseDataModule
from src.paths import ProjectPaths


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--exp-group', type=str, required=True)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--stage', type=str, choices=['predict', 'full'], default='predict')

    model_group = parser.add_argument_group('Model args')
    model_group.add_argument('--update-prototypes', action='store_true')

    dm_group = parser.add_argument_group('Datamodule args')
    dm_group.add_argument('--consecutive-steps', type=int, default=1)
    dm_group.add_argument('--info-cols', type=list[str], default=['surf', 'movedir', 'exp_idx', 't'])
    dm_group.add_argument('--batch-size', type=int, default=1)
    dm_group.add_argument('--pin-memory', type=bool, default=False)
    dm_group.add_argument('--num-workers', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    dm_args = retrieve_arguments(
        args,
        ['batch_size', 'pin_memory', 'num_workers', 'consecutive_steps', 'info_cols']
    )

    exp_dir = ProjectPaths.get_exp_dir(args.exp_group, args.exp_name)
    if not exp_dir.exists():
        raise FileNotFoundError('Experiment dir does not exist')
    eval_dir = ProjectPaths.get_exp_eval_dir(args.exp_group, args.exp_name)
    upd_proto_subfolder = 'proto_update' if args.update_prototypes else 'no_proto_updates'
    eval_dir /= upd_proto_subfolder

    if eval_dir.exists():
        print(f'Evaluation dir already exists at {eval_dir}')
        print(f'Removing...')
        shutil.rmtree(eval_dir)

    fig_dir = eval_dir / 'figures'
    fig_dir.mkdir(exist_ok=True, parents=True)

    with open(exp_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_class_name = config['model']['class_path']
    model_class = pydoc.locate(model_class_name)

    dm_class_name = config['data']['class_path']
    dm_class = pydoc.locate(dm_class_name)

    dm_config = config['data']['init_args']
    dm_config |= dm_args
    mode: Literal['cls', 'reg', 'cls+reg'] = config['data']['init_args']['mode']

    dm: BaseDataModule = dm_class(**dm_config)
    dm.setup(args.stage)
    dm.info_cols += [dm._segment_col]

    surf_le = LabelEncoder()
    dm._transform_cols(surf_le, dm.cls_target)
    match args.stage:
        case 'full':
            dataloader = dm.full_dataloader()
        case 'predict':
            dataloader = dm.predict_dataloader()
        case _:
            raise ValueError(f'Unknown stage: {args.stage}')

    model = model_class.load_from_checkpoint(
        f'experiments/{args.exp_group}/{args.exp_name}/checkpoints/best.ckpt',
    )
    model.eval()
    model.model.update_prototypes = args.update_prototypes

    # Evaluation
    print(f'Running evaluation for {args.exp_group} – {args.exp_name}')

    in_data = []
    in_info = []
    out_reg_pred = []
    out_reg_true = []
    out_cls_pred = []
    out_cls_pred_logits = []
    out_cls_true = []
    surf_embeddings = []
    prototypes = []

    for data in tqdm(dataloader):
        match mode:
            case 'cls+reg':
                X, X_actions, X_info, y_cls, y_reg = data
            case 'cls':
                X, X_actions, X_info, y_cls = data
                y_reg = None
            case 'reg':
                X, X_actions, X_info, y_reg = data
                y_cls = None
            case _:
                raise ValueError(f'Unknown mode: {mode}')

        in_data.append(X)
        in_info.append(X_info)
        prototypes.append(model.model.prototypes.clone())

        with torch.no_grad():
            _out = model(X, X_actions)
        out_reg_pred.append(_out.reg_out)
        out_reg_true.append(y_reg)

        out_cls_pred_logits.append(_out.cls_out)
        out_cls_pred.append(_out.cls_out.squeeze(1).argmax(-1, keepdims=True))
        out_cls_true.append(y_cls)

        surf_embeddings.append(_out.surf_embedding)

    # Prepare outputs
    with torch.no_grad():
        proto_history = torch.stack(prototypes)

    all_data = [torch.cat(item).squeeze().cpu().numpy() for item in (
        in_data,
        in_info,
        out_cls_true,
        out_reg_true,
        out_cls_pred,
        out_reg_pred,
    )]

    # Convert sequences to their last values
    for idx, item in enumerate(all_data):
        if item.ndim == 3:
            all_data[idx] = item[:, -1, :].squeeze()
        elif item.ndim == 1:
            all_data[idx] = item[:, np.newaxis]
        elif item.ndim > 3:
            raise ValueError('Unexpected amount of dimensions for item with idx {}'.format(idx))

    all_data = np.concat(all_data, axis=1)

    # Create all in one dataframe
    columns = (
            dm.features
            + dm.info_cols
            + ([f'{dm.cls_target}_true'] if 'cls' in mode else [])
            + ([f'{n}_true' for n in dm.reg_targets] if 'reg' in mode else [])
            + ([f'{dm.cls_target}_pred'] if 'cls' in mode else [])
            + ([f'{n}_pred' for n in dm.reg_targets] if 'reg' in mode else [])
    )
    df = pd.DataFrame(all_data, columns=columns)

    # Convert int info cols from float
    for col in ('movedir', 'exp_idx', dm._segment_col):
        if col in dm.info_cols:
            df[col] = df[col].astype(int)

    if 'cls' in mode:
        df[[f'{dm.cls_target}_true', f'{dm.cls_target}_pred']] = df[
            [f'{dm.cls_target}_true', f'{dm.cls_target}_pred']].astype(int)
    df[dm.features] = dm.feature_scaler.inverse_transform(df[dm.features])
    if (sc := 'surf') in df.columns:
        df[sc] = surf_le.inverse_transform(df[sc].astype(int))

    _reg_cols = {}
    for col_type in ('true', 'pred'):
        if 'cls' in mode:
            df[f'{dm.cls_target}_{col_type}'] = dm.label_encoder.inverse_transform(df[f'{dm.cls_target}_{col_type}'])
        if 'reg' in mode:
            _reg_cols[col_type] = [f'{n}_{col_type}' for n in dm.reg_targets]
            df[_reg_cols[col_type]] = dm.reg_scaler.inverse_transform(df[_reg_cols[col_type]])

    if (sc := 'surf') in df.columns:
        df[sc] = surf_le.inverse_transform(df[sc].astype(int))

    # Process results
    # Classification
    if 'cls' in mode:
        print('Classification report\n')
        print(classification_report(df['surf_true'], df['surf_pred'], digits=4, zero_division=0))

        cls_rep_df = pd.DataFrame(
            classification_report(
                df['surf_true'],
                df['surf_pred'],
                output_dict=True,
                zero_division=0
            )
        ).T

        fig = plot_df(cls_rep_df)
        fig_path = fig_dir / f'classification_report.png'
        fig.write_image(fig_path, width=500, scale=3)
        print(f'Saved fig for classification report to {fig_path}')
    else:
        print('No classification')
    print('\n')

    # Regression
    if 'reg' in mode:
        r_metrics = {
            'MAE':  mae(df[_reg_cols['true']], df[_reg_cols['pred']], multioutput='raw_values'),
            'RMSE': rmse(df[_reg_cols['true']], df[_reg_cols['pred']], multioutput='raw_values'),
            'R2':   r2(df[_reg_cols['true']], df[_reg_cols['pred']], multioutput='raw_values'),
        }

        reg_stats_df = pd.DataFrame(
            data=dict(zip(dm.reg_targets, zip(*r_metrics.values()))),
            index=r_metrics.keys(),
        ).T

        print('Regression metrics')
        print(reg_stats_df)

        fig = plot_df(reg_stats_df)
        fig_path = fig_dir / f'regression_metrics.png'
        fig.write_image(fig_path, width=500, scale=3)
        print(f'Saved fig for regression metrics to {fig_path}')

    else:
        print('No regression')
    print('\n')

    # Single exp regression exmaples
    reg_exp_idx = df.exp_idx.unique()[0]
    fig = px.line(
        df.query('exp_idx == @reg_exp_idx'),
        y=[_reg_cols['true'][0], _reg_cols['pred'][0]],
        template='plotly_white',
    )
    fig_path = fig_dir / f'regression_example_{reg_exp_idx}.png'
    fig.write_image(fig_path, width=1200, height=600, scale=2)
    print(f'Saved fig for regression example (step {reg_exp_idx}) to {fig_path}')

    # Final prototypes similarity
    p_norm = F.normalize(model.model.prototypes.detach(), dim=-1)
    final_proto_sim = p_norm @ p_norm.T

    fig = plot_similarities(final_proto_sim)
    fig_path = fig_dir / 'proto_similarities_final.png'
    fig.write_image(fig_path, width=500, height=500, scale=3)
    print(f'Saved fig for final proto similarities to {fig_path}')

    # History of proto similarity
    proto_hist_norm = F.normalize(proto_history, dim=-1)
    proto_hist_sim = proto_hist_norm @ proto_hist_norm.transpose(-1, -2)

    n_full_frames = 100
    n_selective_frames = 4
    indices_map = {
        'beginning_idx': (0, n_full_frames),
        'sparse_idx':    (n_full_frames, proto_hist_sim.size(0) - 1),
    }
    for name, borders in indices_map.items():
        # Interactive figs
        full_indices = np.linspace(*borders, n_full_frames, dtype=int)
        fig = plot_similarities(
            proto_hist_sim[full_indices],
            animation_frame=0,
            labels=dict(animation_frame='Frame'),
            title='Prototypes similarity',
            width=500,
            height=500,
        )
        fig_path = fig_dir / f'proto_similarities_{name}_full.html'
        fig.write_html(fig_path, auto_play=False, include_plotlyjs='cdn')
        print(f'Saved interactive fig for {name} to {fig_path}')

        # Individual figs
        selective_indices = np.linspace(*borders, n_selective_frames, endpoint=False, dtype=int)
        for idx in selective_indices:
            fig = plot_similarities(
                proto_hist_sim[idx],
                title=f'Prototypes similarity at step: {idx}',
            )
            fig_path = fig_dir / f'proto_similarities_{idx}.png'
            fig.write_image(fig_path, width=500, height=500, scale=3)
            print(f'Saved fig for step {idx} to {fig_path}')

    print(f'Evaluation for {args.exp_group} – {args.exp_name} is done')
    print(f'See results at {eval_dir}')


def plot_similarities(sim, **imshow_kwargs) -> go.Figure:
    imshow_kwargs = dict(
        color_continuous_scale='Blues',
        text_auto='.3f',
        x=[str(x) for x in range(4)],
        y=[str(x) for x in range(4)],
        template='plotly_white',
        title='Final Prototype similarity',
        range_color=(-1, 1),
        width=500,
        height=500,
    ) | imshow_kwargs

    fig = px.imshow(sim, **imshow_kwargs)
    fig.update_layout(
        xaxis=dict(side='top'),
        coloraxis_showscale=False,
        title=dict(x=0.5),
    )
    return fig


def plot_df(
        df: pd.DataFrame,
        n_decimals: int = 4,
        width: int = 500,
) -> go.Figure:
    df_formatted = df.map(lambda x: f'{x:.{n_decimals}f}' if isinstance(x, float) else x)
    fig = ff.create_table(
        df_formatted,
        index=True,
        colorscale=[[0, '#1d2432'], [0.5, '#e0e0e0'], [1, '#ffffff']],
    )
    fig.update_layout(
        width=width,
        autosize=True,
    )

    return fig


def retrieve_arguments(cfg: Namespace, args: list[str]) -> dict[str, Any]:
    return {k: v for k in args if (v := getattr(cfg, k, None)) is not None}


if __name__ == '__main__':
    main()
