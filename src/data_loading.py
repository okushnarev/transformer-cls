import torch
from torch.nn.utils.rnn import pad_sequence


def pad_batch(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([x.size(0) for x in batch])
    padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
    max_len = padded.size(1)
    mask = torch.arange(max_len)[None, :] >= lengths[:, None]
    return padded, mask