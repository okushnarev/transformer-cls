from typing import Literal, Optional

import pandas as pd
import torch
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from src.data_processing import create_sequences
from src.datamodules.base import BaseDataModule


class BelyaevKushnarevDataModule(BaseDataModule):
    def __init__(
            self,
            features: list[str],
            mode: Literal['cls', 'reg', 'cls+reg'],
            info_cols: Optional[list[str]] = [],
            reg_targets: Optional[list[str]] = None,
            segment_size: int = 100,
            sequence_length: int = 10,
            test_size: float = 0.2,
            val_size: float = 0.1,
            batch_size: int = 4096,
            num_workers: int = 4,
            pin_memory: bool = True,
            seed: int = 69,
    ):
        super().__init__(
            # Dataset-specific arguments
            ds_name='belyaev_kushnarev.csv',
            group_cols='exp_idx',
            stratify_col='surf',
            info_cols=info_cols,
            cls_target='surf' if 'cls' in mode else None,
            reg_targets=reg_targets or ['dx', 'dy', 'dang'] if 'reg' in mode else [],
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

    def apply_transforms(self) -> DataFrame:
        df = super().apply_transforms()
        df[['dx', 'dy', 'dang']] = df.groupby(self.group_cols)[['xpos', 'ypos', 'ang']].diff().fillna(0)
        return df


class BelyaevKushnarevFutureDataModule(BelyaevKushnarevDataModule):

    def __init__(
            self,
            features: list[str],
            mode: Literal['reg', 'cls+reg'],
            info_cols: Optional[list[str]] = [],
            segment_size: int = 100,
            sequence_length: int = 10,
            test_size: float = 0.2,
            val_size: float = 0.1,
            batch_size: int = 4096,
            num_workers: int = 4,
            pin_memory: bool = True,
            seed: int = 69,
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


class BKFutureDataModuleWithActions(BelyaevKushnarevFutureDataModule):
    def __init__(
            self,
            features: list[str],
            actions: list[str],
            mode: Literal['reg', 'cls+reg'],
            info_cols: Optional[list[str]],
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
            segment_size=segment_size,
            sequence_length=sequence_length,
            test_size=test_size,
            val_size=val_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )

        self.actions = actions
        self._all_cols = list(set(
            self._all_cols
            + self.actions
        ))

    def setup(self, stage: str):
        super().setup(stage)
        self.actions_scaler = StandardScaler()
        self._transform_cols(self.actions_scaler, self.actions)

    def _prep_dataset(self, df) -> TensorDataset:
        orig_dataset = super()._prep_dataset(df)
        tensors = orig_dataset.tensors

        X_features = torch.tensor(
            create_sequences(
                df,
                group_by=self.group_cols,
                cols=self.features,
                length=self.sequence_length,
                mode='full'
            ),
            dtype=torch.float
        )

        X_actions = torch.tensor(
            create_sequences(
                df,
                group_by=self.group_cols,
                cols=self.actions,
                length=self.sequence_length,
                mode='full'
            ),
            dtype=torch.float
        )

        X_info = torch.tensor(
            create_sequences(
                df,
                group_by=self.group_cols,
                cols=self.info_cols,
                length=self.sequence_length,
                mode='full'
            ),
            dtype=torch.float
        ) if self.info_cols else None

        if self.info_cols:
            return TensorDataset(X_features, X_actions, X_info, *tensors[1:])
        return TensorDataset(X_features, X_actions, *tensors[1:])
