import shutil

from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

class OverwritableTensorBoardLogger(TensorBoardLogger):
    def __init__(self, overwrite=False, *args, **kwargs):
        save_dir = kwargs.get('save_dir', args[0] if args else 'logs')
        name = kwargs.get('name', args[1] if len(args) > 1 else 'default')
        version = kwargs.get('version', args[2] if len(args) > 2 else None)
        exp_dir = Path(f'{save_dir}/{name}/version_{version}')
        if overwrite and exp_dir.exists():
            shutil.rmtree(str(exp_dir))
        super().__init__(*args, **kwargs)
