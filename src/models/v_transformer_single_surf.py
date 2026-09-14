import torch
from torch import Tensor

from src.models.utils import VerboseModelOutputDecoder
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

