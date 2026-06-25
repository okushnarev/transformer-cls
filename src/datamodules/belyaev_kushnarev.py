from typing import Literal

from pandas import DataFrame

from src.datamodules.base import BaseDataModule


class BelyaevKushnarevDataModule(BaseDataModule):
    def __init__(
            self,
            features: list[str],
            mode: Literal['cls', 'reg', 'cls+reg'],
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
            cls_target='surf' if 'cls' in mode else None,
            reg_targets=['dx', 'dy', 'dang'] if 'reg' in mode else [],
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
