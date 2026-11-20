import pydoc
from pathlib import Path
from typing import Any, Literal

import lightning as L
import yaml
from torch import nn


def load_model(exp_dir: Path, ckpt_type: Literal['best', 'last'] = 'best') -> nn.Module:
    """Loads model from experiment dir with auto class discovery by config file parsing"""
    with open(exp_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_class_name = config['model']['class_path']
    model_class = pydoc.locate(model_class_name)
    model = model_class.load_from_checkpoint(
        exp_dir / f'checkpoints/{ckpt_type}.ckpt',
    )
    return model


def load_datamodule(exp_dir: Path, dm_args: dict[str, Any]) -> L.LightningDataModule:
    """Loads datamodule from experiment dir with auto class discovery by config file parsing"""
    with open(exp_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dm_class_name = config['data']['class_path']
    dm_class = pydoc.locate(dm_class_name)
    dm_config = config['data']['init_args']
    dm_config |= dm_args
    dm = dm_class(**dm_config)

    return dm
