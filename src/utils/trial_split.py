import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch

def split_by_trial(df, trial_column="trial_name", test_size=0.2, seed=42):
    """
    Splits the dataset into train and test sets based on unique trials.
    Ensures that all epochs from the same trial are grouped together.

    Args:
        df (pd.DataFrame): Input dataframe.
        trial_column (str): Name of the trial identifier column.
        test_size (float): Fraction of trials to use for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        train_df (pd.DataFrame): Training subset.
        test_df (pd.DataFrame): Testing subset.
    """
    df = df.copy()
    unique_trials = df[trial_column].unique()

    train_trials, test_trials = train_test_split(
        unique_trials, test_size=test_size, random_state=seed
    )

    train_df = df[df[trial_column].isin(train_trials)].reset_index(drop=True)
    test_df = df[df[trial_column].isin(test_trials)].reset_index(drop=True)

    return train_df, test_df

def prepare_splits(df, trial_column="trial_name", test_size=0.1, seed=42):
    """
    Returns a train_df_full, val_folds (list of 3 val dfs), and test_df using grouped splits.
    Final test set is 10%, and remaining 90% is used for rotating 3-fold validation.

    Args:
        df (pd.DataFrame): Full dataset.
        trial_column (str): Column to group by (e.g. 'trial_name').
        test_size (float): Fraction of data for test (typically 0.1).
        seed (int): Random seed.

    Returns:
        train_df_full (pd.DataFrame): All non-test data (~90% of original).
        val_folds (list of pd.DataFrame): Three validation sets (~10% each).
        test_df (pd.DataFrame): Final 10% test set.
    """
    df = df.copy()
    all_trials = df[trial_column].unique()
    all_trials.sort()
    np.random.seed(seed)
    np.random.shuffle(all_trials)

    total_trials = len(all_trials)
    n_test = int(test_size * total_trials)
    n_val = int(test_size * total_trials)  # Each val fold is also 10%

    test_trials = all_trials[:n_test]
    remaining_trials = all_trials[n_test:]  # ~90% left

    # Prepare 3 val folds: each of size ~10% of total
    val_folds = []
    for i in range(3):
        start = i * n_val
        end = start + n_val
        fold_trials = remaining_trials[start:end]
        val_folds.append(df[df[trial_column].isin(fold_trials)].reset_index(drop=True))

    # Final test df
    test_df = df[df[trial_column].isin(test_trials)].reset_index(drop=True)
    # Train set is all remaining (excluding test + current val) – done inside training loop

    train_df_full = df[df[trial_column].isin(remaining_trials)].reset_index(drop=True)
    return train_df_full, val_folds, test_df