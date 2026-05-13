import torch
from torch.utils.data import TensorDataset

from src.data_processing import create_sequences
from src.datamodules.base import BaseDataModule


class RegressionDataModule(BaseDataModule):
    def __init__(
            self,
            ds_name: str,
            features: list[str],
            reg_targets: list[str],
            group_cols: list[str],
            stratify_col: str,
            segment_size: int,
            sequence_length: int,
            test_size: float = 0.2,
            val_size: float = 0.1,
            batch_size: int = 4096,
            num_workers: int = 4,
            pin_memory: bool = True,
            seed: int = 69,
    ):
        super().__init__(
            ds_name=ds_name,
            features=features,
            group_cols=group_cols,
            stratify_col=stratify_col,
            segment_size=segment_size,
            sequence_length=sequence_length,
            test_size=test_size,
            val_size=val_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )

        self.reg_targets = reg_targets
        self._raw_cols = list(set(self.group_cols + self.features + self.reg_targets + [self.stratify_col]))
        self._scale_cols = self.features + self.reg_targets

    def _prep_dataset(self, df):
        X = torch.tensor(
            create_sequences(
                df,
                group_by=self.group_cols,
                cols=self.features,
                length=self.sequence_length,
                mode='full'
            ),
            dtype=torch.float
        )

        y_reg = torch.tensor(
            create_sequences(
                df,
                group_by=self.group_cols,
                cols=self.reg_targets,
                length=self.sequence_length,
                mode='last'

            ),
            dtype=torch.float
        )

        return TensorDataset(X, y_reg)
