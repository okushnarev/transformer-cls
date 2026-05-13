from pandas import DataFrame
from sklearn.model_selection import train_test_split

def segment_split(
        df: DataFrame,
        group_cols: list[str],
        stratify_col: str,
        segment_col: str,
        **kwargs) -> tuple[DataFrame, DataFrame]:

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