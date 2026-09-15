from typing import Literal, Optional

import pandas as pd
import torch
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from src.data_processing import create_sequences
from src.datamodules.base import BaseDataModule


class BorealTCDataModule(BaseDataModule):
    def __init__(
            self,
            features: list[str],
            mode: Literal['cls', 'reg', 'cls+reg'],
            info_cols: list[str] | None,
            reg_targets: list[str] | None,
            segment_size: int,
            sequence_length: int,
            test_size: float,
            val_size: float,
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            seed: int,
    ):
        super().__init__(
            # Dataset-specific arguments
            ds_name='boreal_imu.csv',
            group_cols='exp_idx',
            stratify_col='surf',
            info_cols=info_cols,
            cls_target='surf' if 'cls' in mode else None,
            reg_targets=reg_targets if 'reg' in mode and reg_targets else [],
            # General arguments
            features=features,
            segment_size=segment_size,
            sequence_length=sequence_length,
            test_size=test_size,
            val_size=val_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )


class BorealTCFutureDataModule(BorealTCDataModule):
    def __init__(
            self,
            features: list[str],
            mode: Literal['cls', 'reg', 'cls+reg'],
            info_cols: list[str] | None,
            reg_targets: list[str] | None,
            segment_size: int,
            sequence_length: int,
            test_size: float,
            val_size: float,
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            seed: int,
    ):
        super().__init__(
            features=features,
            mode=mode,
            info_cols=info_cols,
            reg_targets=[f'{col}_next' for col in features],
            segment_size=segment_size,
            sequence_length=sequence_length,
            test_size=test_size,
            val_size=val_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )

    def apply_transforms(self) -> DataFrame:
        df = super().apply_transforms()
        df_next = df[self.features].copy()
        df_next = df_next.rename(columns=dict(zip(self.features, self.reg_targets)))
        df = pd.concat((df, df_next), axis=1)
        return df

    def setup(self, stage: str):
        super().setup(stage)
        for df in (self.df, self.df_train, self.df_test, self.df_val):
            df[self.reg_targets] = df.groupby(self.group_cols)[self.reg_targets].shift(-1)
            df.dropna(inplace=True)