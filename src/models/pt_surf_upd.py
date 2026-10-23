from typing import Any

import torch
import torch.nn as nn
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

        self.update_prototypes = False

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

        if self.update_prototypes:
            with torch.no_grad():
                updated_prototypes = gru_out.mean(dim=0, keepdims=False)
                self.prototypes.copy_(updated_prototypes)

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

        ret = embedding @ prototypes.transpose(-1, -2) / self.temperature
        if ret.ndim == 3:
            ret = ret.squeeze(1)
        return ret


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


class PrototypicalTransformerSurfUpdateEachStep(PrototypicalTransformerSurfUpdates):
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

        del self.prototypes, self.update_prototypes
        self.register_buffer('prototypes', torch.randn(self.n_classes, self.d_model))

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
        ret: VerboseModelOutputSurfEmbedding = super(PrototypicalTransformerSurfUpdates, self).forward(states, controls)

        surf_embedding = ret.surf_embedding
        surf_embedding = surf_embedding.unsqueeze(1).expand(-1, self.prototypes.size(0), -1)

        gru_out = self.gru_cell(surf_embedding, self.prototypes.expand(batch_size, -1, -1))

        updated_prototypes = gru_out.mean(dim=0, keepdims=False)
        self.prototypes = updated_prototypes.clone().detach()

        return VerboseModelOutputSurfEmbedding(
            reg_out=ret.reg_out,
            cls_out=self.prototype_logits(ret.surf_embedding, updated_prototypes),
            surf_embedding=ret.surf_embedding,
        )


class LitPrototypicalTransformerSurfUpdEachStep(LitMixedLossModel):
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

        model = PrototypicalTransformerSurfUpdateEachStep(
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


class PrototypicalTransformerSurfUpdateOne(PrototypicalTransformerSurfUpdates):
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

        self.gru_cell = nn.GRUCell(
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
        ret: VerboseModelOutputSurfEmbedding = super(PrototypicalTransformerSurfUpdates, self).forward(states, controls)

        argmax_cls = ret.cls_out.argmax(dim=-1)

        gru_out = self.gru_cell(ret.surf_embedding, self.prototypes[argmax_cls])
        out = self.prototypes.repeat(batch_size, 1, 1)
        batch_idx = torch.arange(batch_size, device=argmax_cls.device)
        out[batch_idx, argmax_cls] = gru_out

        if self.update_prototypes:
            with torch.no_grad():
                updated_prototypes = out.mean(dim=0, keepdims=False)
                self.prototypes.copy_(updated_prototypes)

        return VerboseModelOutputSurfEmbedding(
            reg_out=ret.reg_out,
            cls_out=self.prototype_logits(ret.surf_embedding, out),
            surf_embedding=ret.surf_embedding,
        )


class LitPrototypicalTransformerSurfUpdOne(LitMixedLossModel):
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

        model = PrototypicalTransformerSurfUpdateOne(
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


class PrototypicalTransformerSurfUpdateAll(PrototypicalTransformerSurfUpdates):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            in_mlp_hidden_dims: list[int],
            out_mlp_hidden_dims: list[int],
            n_pred_steps: int,
            n_classes: int,
            max_batch_size: int,
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
        self.max_batch_size = max_batch_size
        self.prototypes = nn.Parameter(
            torch.randn(self.max_batch_size, self.n_classes, self.d_model)
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
        ret: VerboseModelOutputSurfEmbedding = super(PrototypicalTransformerSurfUpdates, self).forward(states, controls)

        surf_embedding = ret.surf_embedding.unsqueeze(1)
        prototypes_batch = self.prototypes[:batch_size]

        surf_embedding_expanded = surf_embedding.expand(-1, self.prototypes.size(1), -1)
        gru_out = self.gru_cell(surf_embedding_expanded, prototypes_batch)

        self.prototypes.data[:batch_size] = gru_out.detach()

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
            batch_size = surface_embedding.size(0)
            prototypes = self.prototypes[:batch_size]
        return super().prototype_logits(surface_embedding, prototypes)


class LitPrototypicalTransformerSurfUpdAll(LitMixedLossModel):
    def __init__(
            self,
            in_dim: int,
            output_dim: int,
            in_mlp_hidden_dims: list[int],
            out_mlp_hidden_dims: list[int],
            n_pred_steps: int,
            n_classes: int,
            max_batch_size: int,
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

        model = PrototypicalTransformerSurfUpdateAll(
            input_dim=in_dim,
            output_dim=output_dim,
            in_mlp_hidden_dims=in_mlp_hidden_dims,
            out_mlp_hidden_dims=out_mlp_hidden_dims,
            n_pred_steps=n_pred_steps,
            n_classes=n_classes,
            max_batch_size=max_batch_size,
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

        self.model = torch.compile(model, disable=True)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)


class PrototypicalTransformerSurfUpdateGRU(PrototypicalTransformerSurfUpdates):
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

        del self.gru_cell
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=self.batch_first
        )

        self.update_prototypes = False

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
            ``(batch, seq_len, history_steps, state_dim)``
        :param controls: Historical controls with shape
            ``(batch, seq_len, history_steps, control_dim)``
        :return: VerboseModelOutput containing predicted future states (reg_out) with shape
            ``(batch, seq_len, n_pred_steps, output_dim)`` and the prototype logits (cls_out) with
            shape ``(batch, seq_len, n_classes)``
        :raises ValueError: If state and control history lengths differ or their
            concatenated dimension does not match ``input_dim``
        """

        B, S, H, _ = states.shape
        D = self.d_model
        N_proto = self.prototypes.size(0)

        states = states.reshape(B * S, H, -1)

        if controls is not None:
            controls = controls.reshape(B * S, H, -1)

        ret: VerboseModelOutputSurfEmbedding = (
            super(PrototypicalTransformerSurfUpdates, self).forward(
                states, controls
            )
        )

        # Shape: (B, S, D)
        surf_embedding = ret.surf_embedding.reshape(B, S, D)

        # Shape: (B * N_proto, S, D)
        prototype_input = surf_embedding.repeat_interleave(N_proto, dim=0)

        # Shape: (1, B, D)
        prototypes = self.prototypes.repeat(B, 1).unsqueeze(0)

        # Shape: (B, S, N_proto, D)
        gru_out = (
            self.gru(prototype_input, prototypes)[0]
            .reshape(B, N_proto, S, D)
            .transpose(1, 2)
        )

        if self.update_prototypes:
            # Shape: (N_proto, D)
            updated_prototypes = gru_out[:, -1, ...].mean(dim=0, keepdim=False)
            self.prototypes.copy_(updated_prototypes)

        # Shape: (B, S, 1, D)
        prototype_logits_input = surf_embedding.unsqueeze(2)

        return VerboseModelOutputSurfEmbedding(
            reg_out=ret.reg_out.reshape(B, S, -1),
            cls_out=self.prototype_logits(
                prototype_logits_input, gru_out
            ).squeeze(2),
            surf_embedding=surf_embedding,
        )


class LitPrototypicalTransformerSurfUpdGRU(LitMixedLossModel):
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

        model = PrototypicalTransformerSurfUpdateGRU(
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

class PrototypicalTransformerSurfUpdateOneGRU(PrototypicalTransformerSurfUpdates):
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

        del self.gru_cell
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=self.batch_first
        )

        self.update_prototypes = False

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
            ``(batch, seq_len, history_steps, state_dim)``
        :param controls: Historical controls with shape
            ``(batch, seq_len, history_steps, control_dim)``
        :return: VerboseModelOutput containing predicted future states (reg_out) with shape
            ``(batch, seq_len, n_pred_steps, output_dim)`` and the prototype logits (cls_out) with
            shape ``(batch, seq_len, n_classes)``
        :raises ValueError: If state and control history lengths differ or their
            concatenated dimension does not match ``input_dim``
        """

        B, S, H, _ = states.shape
        D = self.d_model
        N_proto = self.prototypes.size(0)

        states = states.reshape(B * S, H, -1)

        if controls is not None:
            controls = controls.reshape(B * S, H, -1)

        ret: VerboseModelOutputSurfEmbedding = (
            super(PrototypicalTransformerSurfUpdates, self).forward(
                states, controls
            )
        )

        # Shape: (B, S, D)
        surf_embedding = ret.surf_embedding.reshape(B, S, D)

        # Shape: (B * N_proto, S, D)
        prototype_input = surf_embedding.repeat_interleave(N_proto, dim=0)

        # Shape: (1, B, D)
        prototypes = self.prototypes.repeat(B, 1).unsqueeze(0)

        # Shape: (B, S, N_proto, D)
        gru_out = (
            self.gru(prototype_input, prototypes)[0]
            .reshape(B, N_proto, S, D)
            .transpose(1, 2)
        )

        prep_out = self.prototypes.repeat(B, S, 1, 1)
        argmax_cls = ret.cls_out.argmax(dim=-1).reshape(B, S)
        batch_idx = torch.arange(B, device=gru_out.device)[:, None]  # (B, 1)
        seq_idx = torch.arange(S, device=gru_out.device)[None, :]  # (1, S)
        prep_out[batch_idx, seq_idx, argmax_cls] = gru_out[batch_idx, seq_idx, argmax_cls]

        if self.update_prototypes:
            # Shape: (N_proto, D)
            updated_prototypes = prep_out[:, -1, ...].mean(dim=0, keepdim=False)
            self.prototypes.copy_(updated_prototypes)

        # Shape: (B, S, 1, D)
        prototype_logits_input = surf_embedding.unsqueeze(2)

        return VerboseModelOutputSurfEmbedding(
            reg_out=ret.reg_out.reshape(B, S, -1),
            cls_out=self.prototype_logits(
                prototype_logits_input, prep_out
            ).squeeze(2),
            surf_embedding=surf_embedding,
        )

class LitPrototypicalTransformerSurfUpdOneGRU(LitMixedLossModel):
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

        model = PrototypicalTransformerSurfUpdateOneGRU(
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


if __name__ == '__main__':
    in_dim = 10
    n_classes = 4
    d_model = 8
    model = PrototypicalTransformerSurfUpdateOneGRU(
        input_dim=in_dim,
        output_dim=10,
        in_mlp_hidden_dims=[],
        out_mlp_hidden_dims=[],
        n_pred_steps=1,
        n_classes=n_classes,
        temperature=0.1,
        d_model=d_model,
        n_head=1,
        num_layers=1,
        dim_feedforward=16,
        activation='gelu',
        layer_norm_eps=1e-6,
        dropout=0.1,
        decoder_causal=False,
    )
    model.update_prototypes = True
    model.eval()

    seq_len = 20
    consec_step = 10
    for _ in range(3):
        with torch.no_grad():
            in_data = torch.randn(150, consec_step, seq_len, in_dim)
            out = model(in_data)
        print(model.prototypes)