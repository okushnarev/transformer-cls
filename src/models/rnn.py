import torch.nn as nn
from torch import Tensor
from torch.nn import GRU, Module

from src.models.utils import build_mlp


class RNN(Module):
    def __init__(
            self,
            input_dim: int = 6,
            in_mlp_hidden_dims: list[int] = [],
            sequence_length: int = 10,
            out_dim_cls: int = 4,
            out_dim_reg: int = 3,
            out_reg_mlp_hidden_dims: list[int] = [],
            d_model: int = 128,
            hidden_dim: int = 128,
            num_layers: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.in_mlp_hidden_dims = in_mlp_hidden_dims
        self.out_reg_mlp_hidden_dims = out_reg_mlp_hidden_dims
        self.sequence_length = sequence_length

        self.batch_first = True
        self.norm_first = True
        self.bias = False
        self.enable_nested_tensor = False

        self.layer_norm_eps = 1e-5
        self.dropout = 0.1
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


        if self.in_mlp_hidden_dims:
            self.in_proj = build_mlp(self.input_dim, self.in_mlp_hidden_dims, d_model, self.dropout)
        else:
            self.in_proj = nn.Linear(self.input_dim, d_model)


        self.rnn = GRU(
            input_size=self.d_model,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
        )

        if self.out_reg_mlp_hidden_dims:
            self.reg_ffn = build_mlp(
                self.hidden_dim,
                self.out_reg_mlp_hidden_dims,
                out_dim_reg,
                self.dropout
            )
        else:
            self.reg_ffn = nn.Linear(self.hidden_dim, out_dim_reg)
        self.cls_ffn = nn.Linear(self.hidden_dim, out_dim_cls)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.in_proj(x)

        rnn_out, _ = self.rnn(x)
        last_time_step_out = rnn_out[:, -1, :]

        cls_out = self.cls_ffn(last_time_step_out)
        reg_out = self.reg_ffn(last_time_step_out)

        return cls_out, reg_out

