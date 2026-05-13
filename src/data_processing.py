from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def segment_split(
        df: pd.DataFrame,
        group_cols: list[str],
        stratify_col: str,
        segment_col: str,
        **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    # Prepare group cols
    group_cols = list(set(group_cols + [stratify_col, segment_col]))

    # Split
    segment_metadata = df[group_cols].drop_duplicates()
    train_ids, test_ids = train_test_split(
        segment_metadata,
        stratify=segment_metadata[stratify_col],
        **kwargs
    )

    # Compose dataframes
    train_df = df.merge(train_ids, on=group_cols, how='inner')
    test_df = df.merge(test_ids, on=group_cols, how='inner')

    return train_df, test_df


def create_sequences(
        df: pd.DataFrame,
        group_by: list[str] | str,
        cols: list[str] | str,
        length: int,
        mode: Literal['full', 'last'],
) -> np.ndarray:
    if isinstance(cols, str):
        cols = [cols]

    sequences = []
    grouped = df.groupby(group_by)

    for _, group in grouped:
        # Skip if too short
        if len(group) < length:
            continue

        # Manual sliding window
        feature_data = group[cols].values
        for i in range(len(group) - length + 1):
            if mode == 'full':
                seq = feature_data[i: i + length]
            elif mode == 'last':
                seq = feature_data[i + length - 1]
            else:
                raise ValueError(f'Unknown mode for sequence creation: {mode}')
            sequences.append(seq)

    return np.array(sequences)
