from collections import defaultdict

import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self._epoch_metrics = defaultdict(float)


    def process_epoch_metrics(self, metrics: dict, batch_idx: int, max_batches: int):
        for k, v in metrics.items():
            self._epoch_metrics[k] += v

        if batch_idx >= max_batches - 1:
            metric_dict = {k: self._epoch_metrics[k] / max_batches for k in metrics}
            self.logger.log_metrics(metric_dict, step=self.current_epoch)
            for k in metrics:
                del self._epoch_metrics[k]

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            min_lr=1e-6,
        )
        return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor='train_loss')