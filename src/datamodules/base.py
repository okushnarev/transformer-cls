import os
from typing import Optional

import lightning as L
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from pandas import DataFrame

from src.data_processing import create_sequences, segment_split
from src.paths import ProjectPaths


class BaseDataModule(L.LightningDataModule):
    def __init__(
            self,
            ds_name: str,
            features: list[str],
            group_cols: str | list[str],
            stratify_col: str,
            cls_target: Optional[str] = None,
            reg_targets: Optional[list[str]] = [],
            segment_size: int = 100,
            sequence_length: int = 10,
            test_size: float = 0.2,
            val_size: float = 0.1,
            batch_size: int = 4096,
            num_workers: int = 4,
            pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = ProjectPaths.get_raw_data_dir() / ds_name

        self.segment_size = segment_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = int(os.environ.get('PL_GLOBAL_SEED', 42))

        self.test_size = test_size
        self.val_size = val_size

        self.features = features
        self.group_cols = group_cols if isinstance(group_cols, list) else [group_cols]
        self.stratify_col = stratify_col
        self.cls_target = cls_target
        self.reg_targets = reg_targets

        self._all_cols = list(set(
            self.group_cols
            + self.features
            + self.reg_targets
            + ([self.cls_target] if self.cls_target else [])
            + [self.stratify_col]
        ))

        self._segment_col = 'segment_id'

    def apply_transforms(self) -> DataFrame:
        return pd.read_csv(self.data_dir)

    def setup(self, stage: str):
        self.df = self.apply_transforms()
        self.df = self.df[self._all_cols]
        self.df[self._segment_col] = self.df.groupby(self.group_cols).cumcount() // self.segment_size

        self.df_train, self.df_test = segment_split(
            self.df,
            group_cols=self.group_cols,
            stratify_col=self.stratify_col,
            segment_col=self._segment_col,
            # kwagrs
            test_size=self.test_size,
            random_state=self.seed,
        )
        self.df_train, self.df_val = segment_split(
            self.df_train,
            group_cols=self.group_cols,
            stratify_col=self.stratify_col,
            segment_col=self._segment_col,
            # kwagrs
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.seed,
        )

        self.group_cols =  self.group_cols + [self._segment_col]

        self.feature_scaler = StandardScaler()
        self._transform_cols(self.feature_scaler, self.features)

        if self.reg_targets:
            self.reg_scaler = StandardScaler()
            self._transform_cols(self.reg_scaler, self.reg_targets)

        if self.cls_target:
            self.label_encoder = LabelEncoder()
            self._transform_cols(self.label_encoder, self.cls_target)

    def _transform_cols(self, cls, cols: list[str] | str) -> None:
        """
        Method to apply transform (like Scaler or LabelEncoder) to columns.
        :param cls: Class instance that represents transform.
        :param cols: Cols to apply transform.
        """
        self.df_train[cols] = cls.fit_transform(self.df_train[cols])
        self.df_test[cols] = cls.transform(self.df_test[cols])
        self.df_val[cols] = cls.transform(self.df_val[cols])

    def _prep_dataset(self, df) -> TensorDataset:
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

        y_cls = torch.tensor(
            create_sequences(
                df,
                group_by=self.group_cols,
                cols=self.cls_target,
                length=self.sequence_length,
                mode='last'

            ),
            dtype=torch.long
        ) if self.cls_target else None

        y_reg = torch.tensor(
            create_sequences(
                df,
                group_by=self.group_cols,
                cols=self.reg_targets,
                length=self.sequence_length,
                mode='last'

            ),
            dtype=torch.float
        ) if self.reg_targets else None

        return TensorDataset(*(item for item in (X, y_cls, y_reg) if item is not None))

    def _prep_dataloader(self, df, **dataloader_params):
        dataset = self._prep_dataset(df)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **dataloader_params
        )

    def train_dataloader(self):
        return self._prep_dataloader(
            self.df_train,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return self._prep_dataloader(
            self.df_test,
            shuffle=False
        )

    def val_dataloader(self):
        return self._prep_dataloader(
            self.df_val,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return self._prep_dataloader(
            self.df_test,
            shuffle=False
        )
