from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer

from src.models.base_model import LitClassificationModel
from src.models.utils import build_mlp, init_weights


class DetectionTransformer(Module):
    def __init__(
            self,
            input_dim: int | None,
            in_mlp_hidden_dims: list[int],
            d_model: int,
            dim_feedforward: int,
            n_head: int,
            num_layers: int,
            activation: str,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.in_mlp_hidden_dims = in_mlp_hidden_dims

        self.batch_first = True
        self.norm_first = True
        self.bias = False
        self.enable_nested_tensor = False

        self.dim_feedforward = dim_feedforward
        self.layer_norm_eps = 1e-5
        self.dropout = 0.1
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers

        self.activation = activation

        if self.input_dim:
            if self.in_mlp_hidden_dims:
                self.in_proj = build_mlp(self.input_dim, self.in_mlp_hidden_dims, d_model, self.dropout)
            else:
                self.in_proj = nn.Linear(self.input_dim, d_model)
        else:
            self.in_proj = nn.Identity()

        self.cls_token = nn.Parameter(torch.randn(d_model))

        in_encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            batch_first=self.batch_first,
            norm_first=self.norm_first,
            bias=self.bias,
            activation=self.activation,
        )

        self.in_encoder = TransformerEncoder(
            in_encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=self.enable_nested_tensor
        )
        init_weights(self.in_encoder)

        # Binary classification
        self.cls_ffn = nn.Linear(self.d_model, 2)

    def forward(
            self,
            x: Tensor,
            src_key_padding_mask: Tensor | None,
            surf_embedding: Tensor | None,
    ) -> Tensor:
        batch_size = x.size(0)
        x = self.in_proj(x)
        cls_token = self.cls_token[None, None, :].expand(batch_size, -1, -1)

        if x.size(-1) != cls_token.size(-1):
            raise ValueError(f'Input ({x.shape}) and cls token ({cls_token.shape}) should match at the last dimension')

        enc_in = torch.cat((cls_token, surf_embedding, x), dim=1)
        skp_mask_with_token = torch.cat(
            (
                torch.zeros(batch_size, cls_token.size(1) + surf_embedding.size(1), dtype=torch.bool, device=x.device),
                src_key_padding_mask
            ),
            dim=1
        )
        enc_out = self.in_encoder(enc_in, src_key_padding_mask=skp_mask_with_token)
        cls_out = self.cls_ffn(enc_out[:, 0, :])
        return cls_out


class LitDetectionTransformer(LitClassificationModel):
    def __init__(
            self,
            input_dim: int | None,
            in_mlp_hidden_dims: list[int],
            d_model: int,
            dim_feedforward: int,
            n_head: int,
            num_layers: int,
            activation: str,

            compile_model: bool,

            start_lr: float,
            min_lr: float,
            lr_patience: int,
            lr_factor: float,
    ):
        super().__init__(
            start_lr=start_lr,
            min_lr=min_lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
        )

        model = DetectionTransformer(
            input_dim=input_dim,
            in_mlp_hidden_dims=in_mlp_hidden_dims,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            n_head=n_head,
            num_layers=num_layers,
            activation=activation,
        )
        self.model = torch.compile(model, disable=not compile_model)

    def forward(self, *args) -> Any:
        return self.model(*args)


if __name__ == '__main__':
    from src.datamodules.detection.det_base import BaseDetectionDataModule
    import lightning as L
    from pathlib import Path

    dm = BaseDetectionDataModule(
        predictions_path=Path('datasets/bk_exp76_preds.pt'),
        test_size=0.2,
        val_size=0.1,
        batch_size=100,
        num_workers=2,
        pin_memory=True,
        seed=69,
    )
    dm.setup('train')

    model = LitDetectionTransformer(
        input_dim=None,
        in_mlp_hidden_dims=[],
        d_model=32,
        dim_feedforward=64,
        n_head=1,
        num_layers=1,
        activation='gelu',

        compile_model=False,

        start_lr=0.001,
        min_lr=0.00001,
        lr_factor=0.5,
        lr_patience=20,
    )

    trainer = L.Trainer()
    trainer.fit(model, dm)
