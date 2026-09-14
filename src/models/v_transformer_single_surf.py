from typing import Callable

import torch
from torch import Tensor
from torch.nn.functional import mse_loss

from src.models.base_model import LitMixedLossModel
from src.models.utils import LossTerm, VerboseModelOutputDecoder
from src.models.verbose_transformer import VerboseTransformer


class VTransformeSingleSurfUpd(VerboseTransformer):
    def forward(
            self,
            x: Tensor,
            out_models: bool = False,
            out_dec_weights: bool = False,
    ) -> VerboseModelOutputDecoder:
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


        # Pick surface to update
        surf_upd_idx = ca_weights.detach().mean(-2).argmax(-1)
        new_surf_models = batched_surf_models.clone()
        batch_idx = torch.arange(batch_size, device=x.device)
        new_surf_models[batch_idx, surf_upd_idx] = decoder_out.detach()[batch_idx, surf_upd_idx]
        self.surf_models = self.surf_upd_function(self.surf_models, new_surf_models)


        cls_out = self.cls_ffn(ca_weights.permute(0, 2, 1)).squeeze()
        reg_out = self.reg_ffn(ca_out.flatten(start_dim=1))

        return VerboseModelOutputDecoder(
            cls_out=cls_out,
            reg_out=reg_out,
            decoder_out=decoder_out if out_models else None,
            decoder_self_attn=dec_sa_weights,
            decoder_cross_attn=dec_ca_weights,
        )

class LitVTransformerSingleSurfUpd(LitMixedLossModel):
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
            surf_upd_function: Callable[[Tensor, Tensor], Tensor] | None = None,
            norm_surf_models: bool = False,
            dim_feedforward: int = 64,
            reg_loss: Callable | None = mse_loss,
            cls_loss: Callable | None = None,
            attention_losses: dict[str, list[LossTerm]] | None = None,
            start_lr: float = 1e-3,
            min_lr: float = 1e-6,
            lr_patience: int = 2,
            lr_factor: float = 0.1,
    ):
        super().__init__(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
            start_lr=start_lr,
            attention_losses=attention_losses,
            min_lr=min_lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
        )
        model = VTransformeSingleSurfUpd(
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

    def forward(self, x: Tensor) -> VerboseModelOutputDecoder:
        return self.model(x, out_dec_weights=True)