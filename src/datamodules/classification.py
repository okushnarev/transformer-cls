import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset

from src.data_processing import create_sequences
from src.datamodules.base import BaseDataModule


class ClassificationDataModule(BaseDataModule):
    def __init__(
            self,
            ds_name: str,
            features: list[str],
            cls_target: str,
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

        self.cls_target = cls_target
        self._raw_cols = list(set(self.group_cols + self.features + [self.cls_target, self.stratify_col]))

    def setup(self, stage: str):
        super().setup(stage=stage)

        label_encoder = LabelEncoder()
        self.df_train[self.cls_target] = label_encoder.fit_transform(self.df_train[self.cls_target])
        self.df_test[self.cls_target] = label_encoder.transform(self.df_test[self.cls_target])
        self.df_val[self.cls_target] = label_encoder.transform(self.df_val[self.cls_target])

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

        y_cls = torch.tensor(
            create_sequences(
                df,
                group_by=self.group_cols,
                cols=self.cls_target,
                length=self.sequence_length,
                mode='last'

            ),
            dtype=torch.long
        )

        return TensorDataset(X, y_cls)
