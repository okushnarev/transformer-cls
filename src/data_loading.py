import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler


def pad_batch(batch: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    *xs, ys = zip(*batch)
    variable_sz_x = xs[0]
    lengths = torch.tensor([x.size(0) for x in variable_sz_x])
    padded = pad_sequence(variable_sz_x, batch_first=True, padding_value=0.0)
    max_len = padded.size(1)
    mask = torch.arange(max_len)[None, :] >= lengths[:, None]
    y = torch.stack(ys, dim=0)

    extras = []
    for x in xs[1:]:
        if all(isinstance(t, torch.Tensor) for t in x):
            extras.append(torch.stack(x, dim=0))
        else:
            extras.append(x)

    return padded, mask, *extras, y


class ConcatLengthAwareSampler(Sampler[list[int]]):
    """
    Yields batches drawn from a single sub-dataset at a time,
    so each batch has uniform sequence length, therefore no padding
    """

    def __init__(
            self,
            sizes: list[int],
            batch_size: int,
            shuffle: bool
    ):
        self.sizes = sizes
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Offsets of each sub-dataset
        self.offsets = [0]
        for s in sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)

    def __iter__(self):
        # Build one batch index list per sub-dataset
        all_batches = []
        for offset, size in zip(self.offsets, self.sizes):
            local = torch.randperm(size) if self.shuffle else torch.arange(size)
            for i in range(0, size, self.batch_size):
                chunk = local[i:i + self.batch_size]
                all_batches.append(chunk + offset)

        if self.shuffle:
            indices = torch.randperm(len(all_batches))
            all_batches = [all_batches[idx] for idx in indices]

        return iter(all_batches)

    def __len__(self):
        return sum((s + self.batch_size - 1) // self.batch_size for s in self.sizes)