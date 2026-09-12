import math

import torch
from torch import Tensor
from torch.nn import Module, TransformerDecoderLayer


class PositionalEncoding(Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch-first x shape: [batch_size, seq_len, embedding_dim]
        return x + self.pe[:, :x.size(1), :]


class VerboseTransformerDecoderLayer(TransformerDecoderLayer):
    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Tensor | None = None,
            memory_mask: Tensor | None = None,
            tgt_key_padding_mask: Tensor | None = None,
            memory_key_padding_mask: Tensor | None = None,
            tgt_is_causal: bool = False,
            memory_is_causal: bool = False,
            need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.
            need_weights: If specified, outputs self-attention and cross attention weights

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            sa_out = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal, need_weights)
            x = x + sa_out[0]
            mha_out = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal,
                                      need_weights)
            x = x + mha_out[0]
            x = x + self._ff_block(self.norm3(x))
        else:
            sa_out = self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal, need_weights)
            x = self.norm1(
                x + sa_out[0]
            )
            mha_out = self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal, need_weights)
            x = self.norm2(
                x
                + mha_out[0]
            )
            x = self.norm3(x + self._ff_block(x))

        if need_weights:
            return x, sa_out[1], mha_out[1]

        return x

    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Tensor | None,
            key_padding_mask: Tensor | None,
            is_causal: bool = False,
            need_weights: bool = False,
    ) -> tuple[Tensor, Tensor]:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights,
        )
        return self.dropout1(x[0]), x[1]

    def _mha_block(
            self,
            x: Tensor,
            mem: Tensor,
            attn_mask: Tensor | None,
            key_padding_mask: Tensor | None,
            is_causal: bool = False,
            need_weights: bool = False,
    ) -> tuple[Tensor, Tensor]:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights,
        )
        return self.dropout2(x[0]), x[1]
