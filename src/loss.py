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
