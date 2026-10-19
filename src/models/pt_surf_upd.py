from typing import Any

import torch
import torch.nn.functional as F

from src.models.base_model import LitMixedLossModel
from src.models.prototypical_transformer import PrototypicalTransformer
from src.models.utils import BatchedGRUCell, LossTerm, VerboseModelOutputSurfEmbedding


class PrototypicalTransformerSurfUpdates(PrototypicalTransformer):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            in_mlp_hidden_dims: list[int],
            out_mlp_hidden_dims: list[int],
            n_pred_steps: int,
            n_classes: int,
            temperature: float,
            d_model: int,
            n_head: int,
            num_layers: int,
            dim_feedforward: int,
            activation: str = 'gelu',
            layer_norm_eps: float = 1e-5,
            dropout: float = 0.1,
            decoder_causal: bool = False,
    ):
        """Prototype-conditioned Transformer encoder-decoder for multi-step prediction

            The encoder receives a history of state and control observations and prepends
            a learnable surface token. The output corresponding to this token is the
            segment-level representation used by the prototype objective

            The decoder receives ``n_pred_steps`` learned future-query tokens and
            cross-attends to the complete encoder output. It predicts all future states
            in parallel

            :param input_dim: Dimension of the concatenated state and control vector
            :param output_dim: Dimension of the predicted state vector
            :param in_mlp_hidden_dims: Hidden dimensions of the encoder input projection MLP
            :param out_mlp_hidden_dims: Hidden dimensions of the decoder output MLP
            :param n_pred_steps: Number of future states predicted in one forward pass
            :param n_classes: Number of prototype classes
            :param temperature: Temperature applied to cosine similarities
            :param d_model: Transformer embedding dimension
            :param n_head: Number of attention heads
            :param num_layers: Number of Transformer encoder and decoder layers
            :param dim_feedforward: Hidden dimension of Transformer feed-forward blocks
            :param activation: Transformer activation function
            :param layer_norm_eps: Epsilon used by Transformer layer normalization
            :param dropout: Dropout probability
            :param decoder_causal: Whether decoder self-attention is causally masked
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            in_mlp_hidden_dims=in_mlp_hidden_dims,
            out_mlp_hidden_dims=out_mlp_hidden_dims,
            n_pred_steps=n_pred_steps,
            n_classes=n_classes,
            temperature=temperature,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            decoder_causal=decoder_causal,
        )

        self.gru_cell = BatchedGRUCell(
            input_size=self.d_model,
            hidden_size=self.d_model,
        )

    def forward(
            self,
            states: torch.Tensor,
            controls: torch.Tensor | None = None,
    ) -> VerboseModelOutputSurfEmbedding:
        """Run the prototype Transformer

        The state and control vectors are concatenated at every history
        timestep. A surface token is prepended before the encoder. The decoder
        then uses learned future queries to predict all requested future states

        :param states: Historical states with shape
            ``(batch, history_steps, state_dim)``
        :param controls: Historical controls with shape
            ``(batch, history_steps, control_dim)``
        :return: VerboseModelOutput containing predicted future states (reg_out) with shape
            ``(batch, n_pred_steps, output_dim)`` and the prototype logits (cls_out) with
            shape ``(batch, n_classes)``
        :raises ValueError: If state and control history lengths differ or their
            concatenated dimension does not match ``input_dim``
        """

        batch_size = states.size(0)
        ret: VerboseModelOutputSurfEmbedding = super().forward(states, controls)

        surf_embedding = ret.surf_embedding
        surf_embedding = surf_embedding.unsqueeze(1).expand(-1, self.prototypes.size(0), -1)

        gru_out = self.gru_cell(surf_embedding, self.prototypes.expand(batch_size, -1, -1))

        return VerboseModelOutputSurfEmbedding(
            reg_out=ret.reg_out,
            cls_out=self.prototype_logits(ret.surf_embedding, gru_out),
            surf_embedding=ret.surf_embedding,
        )

    def prototype_logits(
            self,
            surface_embedding: torch.Tensor,
            prototypes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute cosine-similarity logits against learnable prototypes

        This is kept separate from :meth:`forward` so the training module can
        decide how the prototype objective should be formulated.

        :param surface_embedding: Surface embeddings with shape
            ``(batch, d_model)``
        :param prototypes: Updated batched prototypes with shape ``(batch, n_classes, d_model)``
        :return: Prototype logits with shape ``(batch, n_classes)``
        """

        if prototypes is None:
            prototypes = self.prototypes
        if prototypes.ndim == 3:
            surface_embedding = surface_embedding.unsqueeze(1)

        embedding = F.normalize(surface_embedding, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)

        return embedding @ prototypes.transpose(-1, -2) / self.temperature


class LitPrototypicalTransformerSurfUpd(LitMixedLossModel):
    def __init__(
            self,
            in_dim: int,
            output_dim: int,
            in_mlp_hidden_dims: list[int],
            out_mlp_hidden_dims: list[int],
            n_pred_steps: int,
            n_classes: int,
            temperature: float,
            d_model: int,
            n_head: int,
            num_layers: int,
            dim_feedforward: int,
            activation: str,
            layer_norm_eps: float,
            dropout: float,
            decoder_causal: bool,
            reg_loss: LossTerm | None,
            cls_loss: LossTerm | None,
            additional_losses: dict[str, list[LossTerm]] | None,
            start_lr: float,
            min_lr: float,
            lr_patience: int,
            lr_factor: float,
    ):
        super().__init__(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
            additional_losses=additional_losses,
            start_lr=start_lr,
            min_lr=min_lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
        )

        model = PrototypicalTransformerSurfUpdates(
            input_dim=in_dim,
            output_dim=output_dim,
            in_mlp_hidden_dims=in_mlp_hidden_dims,
            out_mlp_hidden_dims=out_mlp_hidden_dims,
            n_pred_steps=n_pred_steps,
            n_classes=n_classes,
            temperature=temperature,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            decoder_causal=decoder_causal,
        )

        self.model = torch.compile(model)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
