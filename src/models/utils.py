from torch import nn


def init_weights(component):
    for module in component.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
        else:
            continue
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def build_mlp(
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
        activation: nn.Module = nn.ReLU
) -> nn.Module:
    layers = []
    current_dim = input_dim
    for dim in hidden_dims:
        layers.extend([
            nn.Linear(current_dim, dim),
            activation(),
            nn.Dropout(dropout)
        ])
        current_dim = dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)
