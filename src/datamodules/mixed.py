from torch.utils.data import TensorDataset

from src.datamodules.classification import ClassificationDataModule
from src.datamodules.regression import RegressionDataModule


class MixedDataModule(ClassificationDataModule, RegressionDataModule):
    def __init__(
            self,
            ds_name: str,
            features: list[str],
            cls_target: str,
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
            cls_target=cls_target,
            reg_targets=reg_targets,
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

        self._raw_cols = list(set(self.group_cols + self.features + self.reg_targets + [self.cls_target, self.stratify_col]))

    def _prep_dataset(self, df):
        X, y_cls = ClassificationDataModule._prep_dataset(self, df).tensors
        _, y_reg = RegressionDataModule._prep_dataset(self, df).tensors

        return TensorDataset(X, y_cls, y_reg)
