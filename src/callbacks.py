import logging
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import BasePredictionWriter, Callback

logger = logging.getLogger(__name__)


class CustomPredictionWriter(BasePredictionWriter):
    def setup(self, trainer, pl_module, stage: str) -> None:
        self.output_dir = Path(trainer.log_dir) / 'predictions'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        out = dict(predictions=predictions, batch_indices=batch_indices)
        torch.save(out, self.output_dir / 'predictions.pt')


class ComponentLoadCallback(Callback):
    def __init__(
            self,
            checkpoint_path: str,
            component_map: dict[str, str],
            freeze: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.component_map = component_map
        self.freeze = freeze

    def setup(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            stage: Literal['fit', 'validate', 'test', 'predict'],
    ) -> None:
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict']

        for comp_name, raw_prefix in self.component_map.items():
            prefix = raw_prefix.rstrip('.')

            # Get the real submodule iteratively from string since we cannot access module by string
            component = pl_module
            for attr in comp_name.split('.'):
                component = getattr(component, attr)

            if isinstance(component, nn.Parameter):
                n_loaded = self._load_parameter(comp_name, prefix, component, state_dict)
            elif isinstance(component, nn.Module):
                n_loaded = self._load_module(comp_name, prefix, component, state_dict)
            else:
                logger.warning(
                    '[%s] unsupported component type %r — skipping',
                    comp_name, type(component).__name__,
                )
                continue

            if n_loaded == 0:
                continue

            if self.freeze:
                self._freeze(comp_name, component)

            logger.info(
                'Loaded %d tensors into %r from %s',
                n_loaded, comp_name, self.checkpoint_path,
            )

    def _load_parameter(
            self,
            comp_name: str,
            prefix: str,
            param: nn.Parameter,
            state_dict: dict[str, torch.Tensor],
    ) -> int:
        if prefix not in state_dict:
            logger.warning(
                'No weights found for parameter %r (key %r)',
                comp_name, prefix,
            )
            return 0
        src = state_dict[prefix]
        if src.shape != param.shape:
            logger.warning(
                '[%s] shape mismatch: ckpt %s vs model %s — skipping',
                comp_name, tuple(src.shape), tuple(param.shape),
            )
            return 0
        with torch.no_grad():
            param.copy_(src.to(param.device, param.dtype))
        return 1

    def _load_module(
            self,
            comp_name: str,
            prefix: str,
            module: nn.Module,
            state_dict: dict[str, torch.Tensor],
    ) -> int:
        head = prefix + '.'
        # Extract only the keys that belong to this component
        comp_state = {
            k[len(head):]: v
            for k, v in state_dict.items()
            if k.startswith(head)
        }
        if not comp_state:
            logger.warning(
                'No weights found for component %r with prefix %r',
                comp_name, head,
            )
            return 0
        # Load non-strictly to allow partial / mismatched dicts
        missing, unexpected = module.load_state_dict(comp_state, strict=False)
        if missing:
            logger.info('[%s] missing keys: %d', comp_name, len(missing))
        if unexpected:
            logger.info('[%s] unexpected keys: %d', comp_name, len(unexpected))
        return len(comp_state)

    def _freeze(
            self,
            comp_name: str,
            component: nn.Module | nn.Parameter,
    ) -> None:
        component.requires_grad_(False)
        if isinstance(component, nn.Module):
            component.eval()
            n_params = sum(p.numel() for p in component.parameters())
        else:
            n_params = component.numel()
        logger.info('[%s] frozen (%d params)', comp_name, n_params)
