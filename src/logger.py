import shutil

from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

class OverwritableTensorBoardLogger(TensorBoardLogger):
    def __init__(self, overwrite=False, *args, **kwargs):
        save_dir = kwargs.get('save_dir', 'logs')
        name = kwargs.get('name', 'default')
        version = kwargs.get('version', None)
        if version is not None:
            if isinstance(version, int):
                exp_dir = Path(f'{save_dir}/{name}/version_{version}')
            elif isinstance(version, str):
                exp_dir = Path(f'{save_dir}/{name}/{version}')
            else:
                raise ValueError('Version must be an int or str')
            if overwrite and exp_dir.exists():
                shutil.rmtree(str(exp_dir))
        super().__init__(*args, **kwargs)
