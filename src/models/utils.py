from typing import Callable
from torch.nn.functional import cross_entropy, mse_loss
import torch
from torch import nn


def init_weights(component):
    for module in component.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
        else:
            continue
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def build_mlp(
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
        activation: nn.Module = nn.ReLU
) -> nn.Module:
    layers = []
    current_dim = input_dim
    for dim in hidden_dims:
        layers.extend([
            nn.Linear(current_dim, dim),
            activation(),
            nn.Dropout(dropout)
        ])
        current_dim = dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class MultiTaskLoss(nn.Module):
    """
        This class implements the homoscedastic uncertainty weighting approach
        proposed by Kendall, Gal, and Cipolla in "Multi-Task Learning Using Uncertainty
        to Weigh Losses for Scene Geometry and Semantics"
    """

    def __init__(
            self,
            reg_loss_fn: Callable = mse_loss,
            cls_loss_fn: Callable = cross_entropy,
    ):
        super().__init__()
        self.s_reg = nn.Parameter(torch.zeros(1))
        self.s_cls = nn.Parameter(torch.zeros(1))

        self.reg_loss_fn = reg_loss_fn
        self.cls_loss_fn = cls_loss_fn

    def forward(
            self,
            cls_pred: torch.Tensor,
            cls_tgt: torch.Tensor,
            reg_pred: torch.Tensor,
            reg_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the weighted combined multi-task loss

        :param cls_pred: Predicted logits for the classification task
        :param cls_tgt: Target labels for the classification task
        :param reg_pred: Predicted values for the regression task
        :param reg_tgt: Target values for the regression task
        :return: A tuple containing:
            **overall_loss**: The combined, weighted loss scalar to be used for backpropagation
            **info**: Dictionary containing both the raw and weighted losses for debugging and metrics tracking
        """
        cls_loss = self.cls_loss_fn(cls_pred, cls_tgt)
        reg_loss = self.reg_loss_fn(reg_pred, reg_tgt)

        cls_weighted = torch.exp(-self.s_cls) * cls_loss + 0.5 * self.s_cls
        reg_weighted = 0.5 * torch.exp(-self.s_reg) * reg_loss + 0.5 * self.s_reg

        overall_loss = cls_weighted + reg_weighted
        info = {
            'reg_loss':          reg_loss,
            'cls_loss':          cls_loss,
            'reg_loss_weighted': reg_weighted,
            'cls_loss_weighted': cls_weighted,
        }

        return overall_loss, info
