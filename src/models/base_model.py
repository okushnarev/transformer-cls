from collections import defaultdict
from typing import Literal

import lightning as L
from sympy.printing.pytorch import torch
from torch.nn.functional import cross_entropy, mse_loss, softmax
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LitBaseModel(L.LightningModule):
    def __init__(self, start_lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.start_lr = start_lr
        self._epoch_metrics = defaultdict(float)

    def process_epoch_metrics(self, metrics: dict, batch_idx: int, max_batches: int):
        for k, v in metrics.items():
            self._epoch_metrics[k] += v

        if batch_idx >= max_batches - 1:
            metric_dict = {k: self._epoch_metrics[k] / max_batches for k in metrics}
            self.logger.log_metrics(metric_dict, step=self.current_epoch)
            for k in metrics:
                del self._epoch_metrics[k]

    def log_step_and_epoch_metric(self, name, value, batch_idx, stage: Literal['train', 'val'] = 'train'):
        match stage:
            case 'train':
                max_batches = self.trainer.num_training_batches
            case 'val':
                max_batches = self.trainer.num_val_batches[0]
            case _:
                raise ValueError(f'Cannot determine max number of batches for stage: {stage}')

        self.log(name, value, prog_bar=True, on_epoch=True, on_step=False)
        self.process_epoch_metrics({f'epoch/{name}': value}, batch_idx, max_batches)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.start_lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=2,
            min_lr=1e-6,
        )
        return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor='overall/val_loss')

    def predict_step(self, batch, batch_idx):
        return self(batch[0])


class LitClassificationModel(LitBaseModel):
    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = cross_entropy(outputs, y.squeeze())
        self.log_step_and_epoch_metric('cls/train_loss', loss, batch_idx)
        self.log_step_and_epoch_metric('overall/train_loss', loss, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        # loss
        loss = cross_entropy(outputs, y.squeeze())
        self.log_step_and_epoch_metric('cls/val_loss', loss, batch_idx, stage='val')
        self.log_step_and_epoch_metric('overall/val_loss', loss, batch_idx, stage='val')

        # accuracy
        predicted = torch.argmax(outputs, 1)
        correct = (predicted.view(-1, 1) == y).sum().item()
        accuracy = correct / len(y)
        self.log_step_and_epoch_metric('cls/val_acc', accuracy, batch_idx, stage='val')

    def predict_step(self, batch, batch_idx):
        cls_out = self(batch[0])
        return softmax(cls_out, dim=-1)


class LitRegressionModel(LitBaseModel):
    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = mse_loss(outputs, y.squeeze())
        self.log_step_and_epoch_metric('reg/train_loss', loss, batch_idx)
        self.log_step_and_epoch_metric('overall/train_loss', loss, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        # loss
        loss = mse_loss(outputs, y.squeeze())
        self.log_step_and_epoch_metric('reg/val_loss', loss, batch_idx, stage='val')
        self.log_step_and_epoch_metric('overall/val_loss', loss, batch_idx, stage='val')


class LitMixedModel(LitBaseModel):
    def training_step(self, batch, batch_idx):
        X, y_cls, y_reg = batch
        out_cls, out_reg = self(X)

        # Classification
        cls_loss = cross_entropy(out_cls, y_cls.squeeze())
        self.log_step_and_epoch_metric('cls/train_loss', cls_loss, batch_idx)

        # Regression
        reg_loss = mse_loss(out_reg, y_reg.squeeze())
        self.log_step_and_epoch_metric('reg/train_loss', reg_loss, batch_idx)

        overall_loss = cls_loss + reg_loss
        self.log_step_and_epoch_metric('overall/train_loss', overall_loss, batch_idx)
        return overall_loss

    def validation_step(self, batch, batch_idx):
        X, y_cls, y_reg = batch
        out_cls, out_reg = self(X)

        # Classification
        cls_loss = cross_entropy(out_cls, y_cls.squeeze())
        self.log_step_and_epoch_metric('cls/val_loss', cls_loss, batch_idx, stage='val')
        # accuracy
        predicted = torch.argmax(out_cls, 1)
        correct = (predicted.view(-1, 1) == y_cls).sum().item()
        accuracy = correct / len(y_cls)
        self.log_step_and_epoch_metric('cls/val_acc', accuracy, batch_idx, stage='val')

        # Regression
        reg_loss = mse_loss(out_reg, y_reg.squeeze())
        self.log_step_and_epoch_metric('reg/val_loss', reg_loss, batch_idx, stage='val')

        overall_loss = cls_loss + reg_loss
        self.log_step_and_epoch_metric('overall/val_loss', overall_loss, batch_idx, stage='val')

    def predict_step(self, batch, batch_idx):
        cls_out, reg_out = self(batch[0])
        return softmax(cls_out, dim=-1), reg_out
