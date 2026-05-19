import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

from src.models.base_model import LitBaseModel, LitMixedModel
from src.models.modules import PositionalEncoding


class Transformer(Module):
    def __init__(
            self,
            in_dim: int = 6,
            sequence_length: int = 10,
            out_dim_cls: int = 4,
            out_dim_reg: int = 3,
            d_model: int = 128,
            n_head: int = 1,
            num_layers: int = 1,
            activation: str = 'gelu',
            dim_feedforward: int = 64
    ):
        super().__init__()

        self.in_dim = in_dim
        self.sequence_length = sequence_length

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

        self.in_proj = (nn.Linear(self.in_dim, d_model))
        surf_models = torch.rand(out_dim_cls, d_model)
        self.register_buffer('surf_models', surf_models)

        self.pos_encoder = (
            PositionalEncoding(
                d_model=self.d_model,
            )
        )

        in_encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            batch_first=self.batch_first,
            norm_first=self.norm_first,
            bias=self.bias,
            activation=self.activation,
        )

        self.in_encoder: TransformerEncoder = (
            TransformerEncoder(
                in_encoder_layer,
                num_layers=self.num_layers,
                enable_nested_tensor=self.enable_nested_tensor
            )
        )

        in_decoder_layer = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            batch_first=self.batch_first,
            norm_first=self.norm_first,
            bias=self.bias,
            activation=self.activation,
        )

        self.surf_models_decoder: TransformerDecoder = (
            TransformerDecoder(
                in_decoder_layer,
                num_layers=self.num_layers,
            )
        )

        self.cross_attn: nn.MultiheadAttention = (
            nn.MultiheadAttention(
                self.d_model,
                self.n_head,
                dropout=self.dropout,
                bias=self.bias,
                batch_first=self.batch_first,
            )
        )

        self.norm1 = (nn.LayerNorm(self.d_model, eps=self.layer_norm_eps, bias=self.bias))
        self.norm2 = (nn.LayerNorm(self.d_model, eps=self.layer_norm_eps, bias=self.bias))
        self.ca_ffn = (
            nn.Sequential(
                nn.Linear(self.d_model, self.dim_feedforward, bias=self.bias),
                nn.GELU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.dim_feedforward, self.d_model, bias=self.bias),
                nn.Dropout(p=self.dropout),
            )
        )
        self.reg_ffn = (nn.Linear(self.d_model * self.sequence_length, out_dim_reg))
        self.cls_ffn = (nn.Linear(self.sequence_length, 1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.in_proj(x)
        x = self.pos_encoder(x)
        encoder_out = self.in_encoder(x)
        decoder_out = []
        for batch_idx, batch in enumerate(x):
            _dec_out = self.surf_models_decoder(
                self.surf_models,
                encoder_out[batch_idx]
            )
            self.surf_models = _dec_out.detach()
            decoder_out.append(_dec_out)
        decoder_out = torch.stack(decoder_out)

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

        return cls_out, reg_out


class LitTransformer(LitMixedModel):
    def __init__(
            self,
            in_dim: int = 6,
            sequence_length: int = 10,
            out_dim_cls: int = 4,
            out_dim_reg: int = 3,
            d_model: int = 128,
            n_head: int = 1,
            num_layers: int = 1,
            activation: str = 'gelu',
            dim_feedforward: int = 64
    ):
        super().__init__()
        model = Transformer(
            in_dim=in_dim,
            sequence_length=sequence_length,
            out_dim_cls=out_dim_cls,
            out_dim_reg=out_dim_reg,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            activation=activation,
            dim_feedforward=dim_feedforward,
        )
        self.model = torch.compile(model)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(x)
