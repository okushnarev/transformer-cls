from dataclasses import dataclass

import torch




@dataclass
class InferenceOut:
    in_data: torch.Tensor
    in_info: torch.Tensor

    out_reg_pred: torch.Tensor
    out_reg_true: torch.Tensor

    out_cls_pred: torch.Tensor
    out_cls_true: torch.Tensor

    surf_embeddings: torch.Tensor
    prototypes: torch.Tensor

