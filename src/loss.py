import torch
import torch.nn.functional as F


def cosine_loss(
        in_data: torch.Tensor,
        margin: float = 0.1
) -> torch.Tensor:
    """
        Penalize pairs of inputs whose absolute cosine similarity
        exceeds `margin`

        Args:
            in_data: Tensor of shape (..., n_vecs, d_model)
            margin: Allowed absolute cosine similarity

        Returns:
            Scalar diversity loss
    """
    batch_dims, n_vecs, d_model = in_data.shape

    if n_vecs < 2:
        return in_data.new_zeros(())

    # Cosine similarity as a dot product of normalized vectors
    in_data = F.normalize(in_data, p=2, dim=-1)
    cosine_sim = in_data @ in_data.transpose(-1, -2)

    # Select only i < j
    i, j = torch.triu_indices(
        n_vecs,
        n_vecs,
        offset=1,
        device=in_data.device,
    )

    pairwise_cos = cosine_sim[..., i, j]
    excess = torch.clamp(pairwise_cos.abs() - margin, min=0.0)
    loss = excess.square().mean()

    return loss


def sharp_attn_loss(
        in_data: torch.Tensor,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Loss used to Concentrate attention distribution
    :param in_data: Attention tensor [batch, heads, query, key]
    :param eps: Numerical stability constant
    """
    entropy = -(in_data * torch.log(in_data.clamp_min(eps))).sum(dim=-1)
    return entropy.mean()


def div_attn_loss(
        in_data: torch.Tensor,
        mean_over_dim: int = -2,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Loss used to Encourage different rows to use different keys
    :param in_data: Attention tensor [batch, heads, query, key]
    :param mean_over_dim: Dimension to mean attention over: Query (-2), Key (-1)
    :param eps: Numerical stability constant
    """
    mean_attn = in_data.mean(dim=mean_over_dim)

    keys_sz = mean_attn.size(-1)
    uniform = torch.full_like(mean_attn, 1.0 / keys_sz)

    div_loss = (
            mean_attn *
            (torch.log(mean_attn.clamp_min(eps)) - torch.log(uniform))
    ).sum(dim=-1).mean()

    return div_loss

def eye_loss(in_data: torch.Tensor) -> torch.Tensor:
    keys_sz = in_data.size(-1)
    tgt = torch.eye(keys_sz).to(in_data.device)
    return (tgt - in_data).square().mean()