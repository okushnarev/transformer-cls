from collections import defaultdict
from typing import Callable, Literal

import lightning as L
from sympy.printing.pytorch import torch
from torch.nn.functional import cross_entropy, mse_loss, softmax
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.helpers import get_callable_name
from src.loss import cosine_loss, div_attn_loss, sharp_attn_loss
from src.models.utils import LossTerm, MultiTaskLoss, VerboseModelOutput


class LitBaseModel(L.LightningModule):
    def __init__(
            self,
            start_lr: float = 1e-3,
            min_lr: float = 1e-6,
            lr_patience: int = 2,  # in eval epochs
            lr_factor: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self._epoch_metrics = defaultdict(float)

    def process_epoch_metrics(self, metrics: dict, batch_idx: int, max_batches: int):
        for k, v in metrics.items():
            self._epoch_metrics[k] += v

        if batch_idx >= max_batches - 1:
            metric_dict = {k: self._epoch_metrics[k] / max_batches for k in metrics}
            self.logger.log_metrics(metric_dict, step=self.current_epoch)
            for k in metrics:
                del self._epoch_metrics[k]

    def log_step_and_epoch_metric(
            self,
            name: str,
            value: torch.Tensor,
            batch_idx: int,
            stage: Literal['train', 'val'] = 'train',
            **log_kwargs,
    ):
        log_kwargs = dict(
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        ) | log_kwargs

        match stage:
            case 'train':
                max_batches = self.trainer.num_training_batches
            case 'val':
                max_batches = self.trainer.num_val_batches[0]
            case _:
                raise ValueError(f'Cannot determine max number of batches for stage: {stage}')

        self.log(name, value, **log_kwargs)
        self.process_epoch_metrics({f'epoch/{name}': value}, batch_idx, max_batches)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.start_lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr,
        )

        val_frequency = self.trainer.check_val_every_n_epoch

        return {
            'optimizer':    optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor':   'overall/val_loss',
                'interval':  'epoch',
                'frequency': val_frequency,
                'strict':    False
            }
        }

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


class LitRegressionModelSurfLoss(LitRegressionModel):
    def __init__(
            self,
            reg_loss: Callable = mse_loss,
            model_loss: Callable = cosine_loss,
            lambda_div: float = 0.2,
            start_lr: float = 1e-3,
            min_lr: float = 1e-6,
            lr_patience: int = 2,  # in eval epochs
            lr_factor: float = 0.1,
    ):
        super().__init__(
            start_lr=start_lr,
            min_lr=min_lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
        )

        self.reg_loss = reg_loss
        self.model_loss = model_loss
        self.lambda_div = lambda_div

    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs, models = self(X)
        reg_loss = self.reg_loss(outputs, y.squeeze())
        self.log_step_and_epoch_metric('reg/train_loss', reg_loss, batch_idx)

        model_loss = self.model_loss(models)
        self.log_step_and_epoch_metric('model/train_loss', model_loss, batch_idx)

        overall_loss = reg_loss + self.lambda_div * model_loss
        self.log_step_and_epoch_metric('overall/train_loss', overall_loss, batch_idx)
        return reg_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs, models = self(X)
        reg_loss = self.reg_loss(outputs, y.squeeze())
        self.log_step_and_epoch_metric('reg/val_loss', reg_loss, batch_idx, stage='val')

        model_loss = self.model_loss(models)
        self.log_step_and_epoch_metric('model/val_loss', model_loss, batch_idx, stage='val')

        overall_loss = reg_loss + self.lambda_div * model_loss
        self.log_step_and_epoch_metric('overall/val_loss', overall_loss, batch_idx, stage='val')


class LitRegressionSelfAttnLoss(LitRegressionModel):
    def __init__(
            self,
            reg_loss: Callable = mse_loss,
            sa_loss: Callable | list[Callable] = [sharp_attn_loss, div_attn_loss],
            sa_loss_weight: float | list[float] = 0.2,
            start_lr: float = 1e-3,
            min_lr: float = 1e-6,
            lr_patience: int = 2,  # in eval epochs
            lr_factor: float = 0.1,
    ):
        super().__init__(
            start_lr=start_lr,
            min_lr=min_lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
        )

        self.reg_loss = reg_loss

        if not isinstance(sa_loss, list):
            sa_loss = [sa_loss]
        self.sa_loss: list[Callable] = sa_loss

        if not isinstance(sa_loss_weight, list):
            if not isinstance(sa_loss_weight, float):
                raise TypeError('Coefficients should be of type float. Now: {}'.format(type(sa_loss_weight)))
            sa_loss_weight = [sa_loss_weight] * len(self.sa_loss)
        elif len(sa_loss_weight) != len(sa_loss):
            raise ValueError('Number of coefficients should be the same as number of losses. '
                             'Now sa_loss_coeff ({}) and sa_loss ({})'.format(len(sa_loss_weight), len(sa_loss)))
        self.sa_loss_weight: list[float] = sa_loss_weight

    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs, sa_weights = self(X)
        reg_loss = self.reg_loss(outputs, y.squeeze())
        self.log_step_and_epoch_metric('reg/train_loss', reg_loss, batch_idx)

        sa_losses = []
        for _loss_fn in self.sa_loss:
            _loss = _loss_fn(sa_weights)
            _loss_name = get_callable_name(_loss_fn).replace('_loss', '')
            self.log_step_and_epoch_metric(f'self_attn/{_loss_name}/train_loss', _loss, batch_idx)
            sa_losses.append(_loss)

        overall_loss = reg_loss + sum((l * w for l, w in zip(sa_losses, self.sa_loss_weight)))
        self.log_step_and_epoch_metric('overall/train_loss', overall_loss, batch_idx)
        return reg_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs, sa_weights = self(X)
        reg_loss = self.reg_loss(outputs, y.squeeze())
        self.log_step_and_epoch_metric('reg/val_loss', reg_loss, batch_idx, stage='val')

        sa_losses = []
        for _loss_fn in self.sa_loss:
            _loss = _loss_fn(sa_weights)
            _loss_name = get_callable_name(_loss_fn).replace('_loss', '')
            self.log_step_and_epoch_metric(f'self_attn/{_loss_name}/val_loss', _loss, batch_idx, stage='val')
            sa_losses.append(_loss)

        overall_loss = reg_loss + sum((l * w for l, w in zip(sa_losses, self.sa_loss_weight)))
        self.log_step_and_epoch_metric('overall/val_loss', overall_loss, batch_idx, stage='val')


class LitMixedLossModel(LitBaseModel):
    def __init__(
            self,
            reg_loss: LossTerm | None = LossTerm(mse_loss, 1),
            cls_loss: LossTerm | None = None,
            attention_losses: dict[str, list[LossTerm]] | None = None,
            start_lr: float = 1e-3,
            min_lr: float = 1e-6,
            lr_patience: int = 2,
            lr_factor: float = 0.1,
    ):
        super().__init__(
            start_lr=start_lr,
            min_lr=min_lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
        )

        if not (cls_loss or reg_loss):
            raise ValueError('Both cls and reg losses cannot be None')

        self.reg_loss = reg_loss
        self.cls_loss = cls_loss
        self.attention_losses = attention_losses or {}

    def _step(self, batch: object, batch_idx: object, stage: object) -> object:

        if self.cls_loss:
            if self.reg_loss:
                X, y_cls, y_reg = batch
            else:
                X, y_cls = batch
        elif self.reg_loss:
            X, y_reg = batch
        else:
            raise ValueError('Both cls and reg losses cannot be None')


        model_output: VerboseModelOutput = self(X)

        overall_loss = 0

        if self.cls_loss:
            cls_loss = self.cls_loss.weight * self.cls_loss.fn(
                model_output.cls_out,
                y_cls.squeeze(),
            )

            self.log_step_and_epoch_metric(
                f'cls/{stage}_loss',
                cls_loss,
                batch_idx,
                stage=stage,
            )

            overall_loss += cls_loss

        if self.reg_loss:
            reg_loss = self.reg_loss.weight * self.reg_loss.fn(
                model_output.reg_out,
                y_reg.squeeze(),
            )

            self.log_step_and_epoch_metric(
                f'reg/{stage}_loss',
                reg_loss,
                batch_idx,
                stage=stage,
            )

            overall_loss += reg_loss


        for attention_name, loss_terms in self.attention_losses.items():
            attention = getattr(model_output, attention_name)

            if attention is None:
                raise ValueError(
                    f'Attention \'{attention_name}\' is None, '
                    'but losses were configured for it.'
                )

            for term in loss_terms:
                loss = term.fn(attention)

                loss_name = get_callable_name(term.fn).replace('_loss', '')

                self.log_step_and_epoch_metric(
                    f'{attention_name}/{loss_name}/{stage}_loss',
                    loss,
                    batch_idx,
                    stage=stage,
                )

                overall_loss = overall_loss + term.weight * loss

        self.log_step_and_epoch_metric(
            f'overall/{stage}_loss',
            overall_loss,
            batch_idx,
            stage=stage,
        )

        return overall_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, 'val')


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


class LitMixedModelWeightedLoss(LitMixedModel):
    def __init__(
            self,
            model: torch.nn.Module,
            reg_loss_fn: Callable = mse_loss,
            cls_loss_fn: Callable = cross_entropy,
            model_start_lr: float = 1e-3,
            loss_start_lr: float = 1e-2,
            min_lr: float = 1e-6,
            lr_patience: int = 2,
            lr_factor: float = 0.1,
    ):
        super().__init__(
            start_lr=model_start_lr,
            min_lr=min_lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
        )

        self.loss = MultiTaskLoss(reg_loss_fn=reg_loss_fn, cls_loss_fn=cls_loss_fn)
        self.loss_start_lr = loss_start_lr
        self.model = model

    def configure_optimizers(self):
        config = super().configure_optimizers()
        optimizer = AdamW([
            {'params': self.model.parameters(), 'lr': self.start_lr},
            {'params': self.loss.parameters(), 'lr': self.loss_start_lr},
        ])

        # Update 'optimizer' dict field and scheduler to reference the new optimizer
        config['optimizer'] = optimizer
        config['lr_scheduler']['scheduler'].optimizer = optimizer

        return config

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y_cls, y_reg = batch
        out_cls, out_reg = self(X)

        overall_loss, info_losses = self.loss(
            out_cls, y_cls.squeeze(),
            out_reg, y_reg.squeeze(),
        )
        # Classification
        self.log_step_and_epoch_metric('cls/train_loss', info_losses['cls_loss'], batch_idx)
        self.log_step_and_epoch_metric('cls/train_loss_weighted', info_losses['cls_loss_weighted'], batch_idx,
                                       prog_bar=False)

        # Regression
        self.log_step_and_epoch_metric('reg/train_loss', info_losses['reg_loss'], batch_idx)
        self.log_step_and_epoch_metric('reg/train_loss_weighted', info_losses['reg_loss_weighted'], batch_idx,
                                       prog_bar=False)

        # Overall loss
        self.log_step_and_epoch_metric('overall/train_loss', overall_loss, batch_idx)

        # Weigths
        self.log_step_and_epoch_metric('uncertainty/s_cls', self.loss.s_cls, batch_idx, prog_bar=False)
        self.log_step_and_epoch_metric('uncertainty/s_reg', self.loss.s_reg, batch_idx, prog_bar=False)

        return overall_loss

    def validation_step(self, batch, batch_idx):
        X, y_cls, y_reg = batch
        out_cls, out_reg = self(X)

        overall_loss, info_losses = self.loss(
            out_cls, y_cls.squeeze(),
            out_reg, y_reg.squeeze(),
        )
        # Classification
        self.log_step_and_epoch_metric('cls/val_loss', info_losses['cls_loss'], batch_idx, stage='val')
        self.log_step_and_epoch_metric('cls/val_loss_weighted', info_losses['cls_loss_weighted'], batch_idx,
                                       stage='val', prog_bar=False)

        # accuracy
        predicted = torch.argmax(out_cls, 1)
        correct = (predicted.view(-1, 1) == y_cls).sum().item()
        accuracy = correct / len(y_cls)
        self.log_step_and_epoch_metric('cls/val_acc', accuracy, batch_idx, stage='val')

        # Regression
        self.log_step_and_epoch_metric('reg/val_loss', info_losses['reg_loss'], batch_idx, stage='val')
        self.log_step_and_epoch_metric('reg/val_loss_weighted', info_losses['reg_loss_weighted'], batch_idx,
                                       stage='val', prog_bar=False)

        self.log_step_and_epoch_metric('overall/val_loss', overall_loss, batch_idx, stage='val')
