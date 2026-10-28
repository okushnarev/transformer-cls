import pydoc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import torch
import yaml
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.datamodules.base import BaseDataModule
from src.helpers import get_model_hash
from src.paths import ProjectPaths

memory = joblib.Memory(ProjectPaths.get_root() / '.math_cache', verbose=0)


@dataclass
class InferenceOut:
    in_data: torch.Tensor
    in_info: torch.Tensor

    out_reg_pred: torch.Tensor
    out_reg_true: torch.Tensor

    out_cls_pred: torch.Tensor
    out_cls_true: torch.Tensor

    surf_embeddings: torch.Tensor
    prototypes: torch.Tensor


def model_inference(
        exp_group: str,
        exp_name: str,
        stage: str,
        dm_args: dict[str, Any],
        update_prototypes: bool,
        ckpt_type: Literal['best', 'last'] = 'best',
) -> InferenceOut:
    exp_dir = ProjectPaths.get_exp_dir(exp_group, exp_name)
    if not exp_dir.exists():
        raise FileNotFoundError('Experiment dir does not exist')
    ckpt = torch.load(f'experiments/{exp_group}/{exp_name}/checkpoints/{ckpt_type}.ckpt', map_location='cpu')
    model_hash = get_model_hash(ckpt['state_dict'])

    return run_model(
        exp_group=exp_group,
        exp_name=exp_name,
        exp_dir=exp_dir,
        ckpt_type=ckpt_type,
        stage=stage,
        dm_args=dm_args,
        update_prototypes=update_prototypes,
        model_hash=model_hash,
    )


@memory.cache
def run_model(
        exp_group: str,
        exp_name: str,
        exp_dir: Path,
        ckpt_type: Literal['best', 'last'],
        stage: str,
        dm_args: dict[str, Any],
        update_prototypes: bool,
        model_hash: str,
) -> InferenceOut:
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
    dm.setup(stage)
    dm.info_cols += [dm._segment_col]

    surf_le = LabelEncoder()
    dm._transform_cols(surf_le, dm.cls_target)
    match stage:
        case 'full':
            dataloader = dm.full_dataloader()
        case 'predict':
            dataloader = dm.predict_dataloader()
        case _:
            raise ValueError(f'Unknown stage: {stage}')

    model = model_class.load_from_checkpoint(
        f'experiments/{exp_group}/{exp_name}/checkpoints/{ckpt_type}.ckpt',
    )
    model.eval()
    model.model.update_prototypes = update_prototypes
    # Evaluation
    print(f'Running evaluation for {exp_group} – {exp_name}')

    in_data = []
    in_info = []
    out_reg_pred = []
    out_reg_true = []
    out_cls_pred = []
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

        out_cls_pred.append(_out.cls_out.squeeze(1).argmax(-1, keepdims=True))
        out_cls_true.append(y_cls)

        surf_embeddings.append(_out.surf_embedding)

    with torch.no_grad():
        all_data = [torch.cat(item) for item in (
            in_data,
            in_info,
            out_cls_true,
            out_reg_true,
            out_cls_pred,
            out_reg_pred,
            surf_embeddings
        )]
        proto_history = torch.stack(prototypes)

    return InferenceOut(*all_data, proto_history)
