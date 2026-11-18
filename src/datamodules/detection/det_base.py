from pathlib import Path

import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from src.data_loading import ConcatLengthAwareSampler, pad_batch
from src.data_processing import split_tensor_ds
from src.helpers import insert_to_dict
from src.inference import InferenceOut


class BaseDetectionDataModule(L.LightningDataModule):
    def __init__(
            self,
            predictions_path: Path,
            test_size: float,
            val_size: float,
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            seed: int = 69,
    ):
        super().__init__()

        self.predictions_path = predictions_path
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def setup(self, stage: str):
        predictions = InferenceOut.load(self.predictions_path)
        B, N, D = predictions.prototypes.shape

        idx_range = torch.arange(N).unsqueeze(0).repeat(B, 1)
        mask = torch.ones_like(idx_range, dtype=torch.bool)
        mask[torch.arange(B), predictions.out_cls_true.squeeze()] = False

        true_surf_proto = predictions.prototypes[~mask].view(B, 1, D)
        other_surf_proto = predictions.prototypes[mask].view(B, N - 1, D)
        tensors: dict[int, tuple[torch.Tensor, ...]] = {}
        for n_items in range(1, N):
            combs = torch.combinations(torch.arange(other_surf_proto.size(1)), n_items)
            X_part_true = other_surf_proto[:, combs].view(-1, n_items, D)
            y_part_true = torch.ones(X_part_true.size(0), 1)
            tensors = insert_to_dict(tensors, X_part_true, y_part_true)

            X_part_false = torch.cat((X_part_true, true_surf_proto.repeat_interleave(combs.size(0), dim=0)), dim=1)
            y_part_false = torch.zeros(X_part_false.size(0), 1)
            tensors = insert_to_dict(tensors, X_part_false, y_part_false)

        ds_parts = [TensorDataset(*v) for v in tensors.values()]
        self.full_ds = ConcatDataset(ds_parts)
        subsets = [
            split_tensor_ds(
                p,
                test_sz=self.test_size,
                val_sz=self.val_size,
                seed=self.seed
            ) for p in ds_parts
        ]
        self.train_ds, self.test_ds, self.val_ds = [ConcatDataset(ds) for ds in zip(*subsets)]

    def _prep_dataloader(self, ds: ConcatDataset, *, shuffle: bool, **dataloader_params):
        return DataLoader(
            ds,
            collate_fn=pad_batch,
            batch_sampler=ConcatLengthAwareSampler(
                sizes=[len(p) for p in ds.datasets],
                batch_size=self.batch_size,
                shuffle=shuffle,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **dataloader_params
        )

    def train_dataloader(self):
        return self._prep_dataloader(
            self.train_ds,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return self._prep_dataloader(
            self.test_ds,
            shuffle=False
        )

    def val_dataloader(self):
        return self._prep_dataloader(
            self.val_ds,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def full_dataloader(self):
        return self._prep_dataloader(
            self.full_ds,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
