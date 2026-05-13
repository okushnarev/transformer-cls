import lightning as L
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data_processing import segment_split
from src.paths import ProjectPaths


class BaseDataModule(L.LightningDataModule):
    def __init__(
            self,
            ds_name: str,
            features: list[str],
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
            **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.data_dir = ProjectPaths.get_raw_data_dir() / ds_name

        self.segment_size = segment_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        self.test_size = test_size
        self.val_size = val_size

        self.features = features
        self.group_cols = group_cols
        self.stratify_col = stratify_col
        self._raw_cols = list(set(self.group_cols + self.features + [self.stratify_col]))
        self._scale_cols = self.features

        self._segment_col = 'segment_id'

    def setup(self, stage: str):
        self.df_raw = pd.read_csv(self.data_dir)
        self.df_raw = self.df_raw[self._raw_cols]
        self.df_raw[self._segment_col] = self.df_raw.groupby(self.group_cols).cumcount() // self.segment_size

        self.df_train, self.df_test = segment_split(
            self.df_raw,
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

        self.group_cols = [self._segment_col] + self.group_cols

        scaler = StandardScaler()
        self.df_train[self._scale_cols] = scaler.fit_transform(self.df_train[self._scale_cols])
        self.df_test[self._scale_cols] = scaler.transform(self.df_test[self._scale_cols])
        self.df_val[self._scale_cols] = scaler.transform(self.df_val[self._scale_cols])

    def _prep_dataset(self, df) -> TensorDataset:
        ...

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
            drop_last=True,
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
            self.df_raw,
            shuffle=False
        )
