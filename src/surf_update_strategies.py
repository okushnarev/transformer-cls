import torch


def max_diff_surf_upd(old_models: torch.Tensor, new_models: torch.Tensor) -> torch.Tensor:
    dist_of_models = new_models - old_models
    distances = torch.linalg.vector_norm(dist_of_models.flatten(start_dim=1), dim=-1)
    max_dist_idx = torch.argmax(distances)
    return new_models[max_dist_idx]


def min_diff_surf_upd(old_models: torch.Tensor, new_models: torch.Tensor) -> torch.Tensor:
    dist_of_models = new_models - old_models
    distances = torch.linalg.vector_norm(dist_of_models.flatten(start_dim=1), dim=-1)
    max_dist_idx = torch.argmin(distances)
    return new_models[max_dist_idx]
