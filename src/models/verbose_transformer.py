from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base_model import LitRegressionModel
from src.models.modules import VerboseTransformerDecoder, VerboseTransformerDecoderLayer
from src.models.transformer import Transformer
from src.models.utils import VerboseModelOutput, init_weights


class VerboseTransformer(Transformer):
    def __init__(
            self,
            input_dim: int = 6,
            in_mlp_hidden_dims: list[int] = [],
            sequence_length: int = 10,
            out_dim_cls: int = 4,
            out_dim_reg: int = 3,
            out_reg_mlp_hidden_dims: list[int] = [],
            d_model: int = 128,
            n_head: int = 1,
            num_layers: int = 1,
            activation: str = 'gelu',
            surf_upd_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
            norm_surf_models: bool = False,
            dim_feedforward: int = 64
    ):
        super().__init__(
            input_dim=input_dim,
            in_mlp_hidden_dims=in_mlp_hidden_dims,
            sequence_length=sequence_length,
            out_dim_cls=out_dim_cls,
            out_dim_reg=out_dim_reg,
            out_reg_mlp_hidden_dims=out_reg_mlp_hidden_dims,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            activation=activation,
            surf_upd_function=surf_upd_function,
            norm_surf_models=norm_surf_models,
            dim_feedforward=dim_feedforward,
        )

        in_decoder_layer = VerboseTransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            batch_first=self.batch_first,
            norm_first=self.norm_first,
            bias=self.bias,
            activation=self.activation,
        )

        decoder_norm = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps,
                                    bias=self.bias) if self.norm_surf_models else None
        self.surf_models_decoder = VerboseTransformerDecoder(
            in_decoder_layer,
            num_layers=self.num_layers,
            norm=decoder_norm,
        )
        init_weights(self.surf_models_decoder)

    def forward(
            self,
            x: Tensor,
            out_models: bool = False,
            out_dec_weights: bool = False,
    ) -> VerboseModelOutput:
        batch_size = x.size(0)

        x = self.in_proj(x)
        x = self.pos_encoder(x)
        encoder_out = self.in_encoder(x)

        batched_surf_models = self.surf_models.unsqueeze(0).expand(batch_size, -1, -1)
        decoder_out, dec_sa_weights, dec_ca_weights = self.surf_models_decoder(
            batched_surf_models,
            encoder_out,
            need_weights=out_dec_weights,
        )

        new_surf_models = decoder_out.detach()
        self.surf_models = self.surf_upd_function(self.surf_models, new_surf_models)

        if self.norm_first:
            ca_out, ca_weights = self.cross_attn(
                encoder_out,
                self.norm1(decoder_out),
                self.norm1(decoder_out),
                need_weights=True,
            )
            ca_out = self.ca_ffn(self.norm2(ca_out))
        else:
            ca_out, ca_weights = self.cross_attn(
                encoder_out,
                decoder_out,
                decoder_out,
                need_weights=True,
            )
            ca_out = self.norm1(ca_out)
            ca_out = self.norm2(self.ca_ffn(ca_out))

        cls_out = self.cls_ffn(ca_weights.permute(0, 2, 1)).squeeze()
        reg_out = self.reg_ffn(ca_out.flatten(start_dim=1))

        return VerboseModelOutput(
            cls_out=cls_out,
            reg_out=reg_out,
            decoder_out=decoder_out if out_models else None,
            decoder_sa_weights=dec_sa_weights,
            decoder_ca_weights=dec_ca_weights,
        )


class LitVerboseTransformerRegression(LitRegressionModel):
    def __init__(
            self,
            input_dim: int = 6,
            in_mlp_hidden_dims: list[int] = [],
            sequence_length: int = 10,
            out_dim_cls: int = 4,
            out_dim_reg: int = 3,
            out_reg_mlp_hidden_dims: list[int] = [],
            d_model: int = 128,
            n_head: int = 1,
            num_layers: int = 1,
            activation: str = 'gelu',
            surf_upd_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
            norm_surf_models: bool = False,
            dim_feedforward: int = 64,
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
        model = VerboseTransformer(
            input_dim=input_dim,
            in_mlp_hidden_dims=in_mlp_hidden_dims,
            sequence_length=sequence_length,
            out_dim_cls=out_dim_cls,
            out_dim_reg=out_dim_reg,
            out_reg_mlp_hidden_dims=out_reg_mlp_hidden_dims,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            activation=activation,
            surf_upd_function=surf_upd_function,
            norm_surf_models=norm_surf_models,
            dim_feedforward=dim_feedforward,
        )
        self.model = torch.compile(model)

    def forward(self, x: Tensor) -> Tensor:
        output: VerboseModelOutput = self.model(x)
        return output.reg_out
