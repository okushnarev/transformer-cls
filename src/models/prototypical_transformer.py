from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import LitMixedLossModel
from src.models.modules import PositionalEncoding
from src.models.utils import LossTerm, VerboseModelOutput, build_mlp, init_weights


class PrototypicalTransformer(nn.Module):
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
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_mlp_hidden_dims = in_mlp_hidden_dims
        self.out_mlp_hidden_dims = out_mlp_hidden_dims
        self.n_pred_steps = n_pred_steps
        self.n_classes = n_classes
        self.temperature = temperature
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

        # Checks
        if self.temperature <= 0:
            raise ValueError('Temperature must be greater than zero')

        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.decoder_causal = decoder_causal

        self.batch_first = True
        self.norm_first = True
        self.bias = False
        self.enable_nested_tensor = False

        # Project concatenated state + control observations into model space
        self.in_proj = build_mlp(
            self.input_dim,
            self.in_mlp_hidden_dims,
            self.d_model,
            self.dropout,
        )

        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
        )

        # Learnable segment-level surface token
        self.surface_token = nn.Parameter(
            torch.randn(1, 1, self.d_model)
        )

        # One learnable prototype per known class
        self.prototypes = nn.Parameter(
            torch.randn(self.n_classes, self.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=self.batch_first,
            norm_first=self.norm_first,
            bias=self.bias,
        )

        self.in_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=self.enable_nested_tensor,
        )

        init_weights(self.in_encoder)

        self.future_queries = nn.Parameter(
            torch.randn(1, self.n_pred_steps, self.d_model)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=self.batch_first,
            norm_first=self.norm_first,
            bias=self.bias,
        )

        self.surf_models_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers,
        )

        init_weights(self.surf_models_decoder)

        # Map each decoder representation directly to a future state
        self.out_proj = build_mlp(
            self.d_model,
            self.out_mlp_hidden_dims,
            self.output_dim,
            self.dropout,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize learnable token and prototype parameters"""

        nn.init.normal_(self.surface_token, mean=0.0, std=0.02)
        nn.init.normal_(self.future_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    def forward(
            self,
            states: torch.Tensor,
            controls: torch.Tensor,
    ) -> VerboseModelOutput:
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

        if states.ndim != 3:
            raise ValueError(
                'States must have shape (batch, history_steps, state_dim)'
            )

        if controls.ndim != 3:
            raise ValueError(
                'Controls must have shape (batch, history_steps, control_dim)'
            )

        if states.shape[:2] != controls.shape[:2]:
            raise ValueError(
                'States and controls must have the same batch and history dimensions'
            )

        x = torch.cat((states, controls), dim=-1)

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f'Expected concatenated state/control dimension '
                f'{self.input_dim}, got {x.shape[-1]}'
            )

        batch_size = x.shape[0]

        x = self.in_proj(x)

        surface_token = self.surface_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat((surface_token, x), dim=1)

        encoder_input = self.pos_encoder(encoder_input)
        encoder_output = self.in_encoder(encoder_input)

        # Surface representation used by the prototype objective
        surface_embedding = encoder_output[:, 0]

        decoder_input = self.future_queries.expand(batch_size, -1, -1)

        decoder_output = self.surf_models_decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_is_causal=self.decoder_causal,
        )

        predictions = self.out_proj(decoder_output)

        return VerboseModelOutput(
            reg_out=predictions,
            cls_out=self.prototype_logits(surface_embedding)
        )

    def prototype_logits(
            self,
            surface_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine-similarity logits against learnable prototypes

        This is kept separate from :meth:`forward` so the training module can
        decide how the prototype objective should be formulated.

        :param surface_embedding: Surface embeddings with shape
            ``(batch, d_model)``
        :return: Prototype logits with shape ``(batch, n_classes)``
        """

        embedding = F.normalize(surface_embedding, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)

        return embedding @ prototypes.transpose(0, 1) / self.temperature


class LitPrototypicalTransformer(LitMixedLossModel):
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
            reg_loss: Callable | LossTerm | None,
            cls_loss: Callable | LossTerm | None,
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

        model = PrototypicalTransformer(
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
        self.model(*args, **kwargs)
