from pathlib import Path

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class CustomPredictionWriter(BasePredictionWriter):
    def setup(self, trainer, pl_module, stage: str) -> None:
        self.output_dir = Path(trainer.log_dir) / 'predictions'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        out = dict(predictions=predictions, batch_indices=batch_indices)
        torch.save(out, self.output_dir / 'predictions.pt')

