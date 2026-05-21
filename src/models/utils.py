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
