from collections import defaultdict

import lightning as L
from sympy.printing.pytorch import torch
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LitBaseModel(L.LightningModule):
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

    def log_step_and_epoch_metric(self, name, value, batch_idx):
        self.log(name, value, prog_bar=True, on_epoch=True, on_step=False)
        self.process_epoch_metrics({f'epoch/{name}': value}, batch_idx, self.trainer.num_training_batches)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            min_lr=1e-6,
        )
        return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor='overall/train_loss')


class LitClassificationModel(LitBaseModel):
    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = cross_entropy(outputs, y.squeeze())
        self.log_step_and_epoch_metric('cls/train_loss', loss, batch_idx)
        self.log('overall/train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        # loss
        loss = cross_entropy(outputs, y.squeeze())
        self.log_step_and_epoch_metric('cls/val_loss', loss, batch_idx)
        self.log('overall/val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

        # accuracy
        predicted = torch.argmax(outputs, 1)
        correct = (predicted.view(-1, 1) == y).sum().item()
        accuracy = correct / len(y)
        self.log_step_and_epoch_metric('cls/val_acc', accuracy, batch_idx)


class LitRegressionModel(LitBaseModel):
    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = mse_loss(outputs, y.squeeze())
        self.log_step_and_epoch_metric('reg/train_loss', loss, batch_idx)
        self.log('overall/train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        # loss
        loss = mse_loss(outputs, y.squeeze())
        self.log_step_and_epoch_metric('reg/val_loss', loss, batch_idx)
        self.log('overall/val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)


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
        self.log_step_and_epoch_metric('cls/val_loss', cls_loss, batch_idx)
        # accuracy
        predicted = torch.argmax(out_cls, 1)
        correct = (predicted.view(-1, 1) == y_cls).sum().item()
        accuracy = correct / len(y_cls)
        self.log_step_and_epoch_metric('cls/val_acc', accuracy, batch_idx)

        # Regression
        reg_loss = mse_loss(out_reg, y_reg.squeeze())
        self.log_step_and_epoch_metric('reg/val_loss', reg_loss, batch_idx)

        overall_loss = cls_loss + reg_loss
        self.log_step_and_epoch_metric('overall/val_loss', overall_loss, batch_idx)
