import pandas as pd
import numpy as np

# Default columns to permanently drop from all feature sets
DEFAULT_DROP_COLUMNS = [
    "C3-A2_delta", "C4-A1_delta", "F3-A2_delta", "F4-A1_delta", "O1-A2_delta", "O2-A1_delta",
    "C3-A2_theta", "C4-A1_theta", "F3-A2_theta", "F4-A1_theta", "O1-A2_theta", "O2-A1_theta",
    "C3-A2_alpha", "C4-A1_alpha", "F3-A2_alpha", "F4-A1_alpha", "O1-A2_alpha", "O2-A1_alpha",
    "C3-A2_rel_beta", "C4-A1_rel_beta", "F3-A2_rel_beta", "F4-A1_rel_beta", "O1-A2_rel_beta", "O2-A1_rel_beta",
    "C3-A2_rel_delta", "C4-A1_rel_delta", "F3-A2_rel_delta", "F4-A1_rel_delta", "O1-A2_rel_delta", "O2-A1_rel_delta",
    "C3-A2_rel_theta", "C4-A1_rel_theta", "F3-A2_rel_theta", "F4-A1_rel_theta", "O1-A2_rel_theta", "O2-A1_rel_theta",
    "C3-A2_rel_alpha", "C4-A1_rel_alpha", "F3-A2_rel_alpha", "F4-A1_rel_alpha", "O1-A2_rel_alpha", "O2-A1_rel_alpha",
    "C3-A2_beta", "C4-A1_beta", "F3-A2_beta", "F4-A1_beta", "O1-A2_beta", "O2-A1_beta",
    "C3-A2_wavelet_level_0_mean", "C3-A2_wavelet_level_1_mean", "C3-A2_wavelet_level_2_mean",
    "C3-A2_wavelet_level_3_mean", "C3-A2_wavelet_level_4_mean",
    "C4-A1_wavelet_level_0_mean", "C4-A1_wavelet_level_1_mean", "C4-A1_wavelet_level_2_mean",
    "C4-A1_wavelet_level_3_mean", "C4-A1_wavelet_level_4_mean",
    "F3-A2_wavelet_level_0_mean", "F3-A2_wavelet_level_1_mean", "F3-A2_wavelet_level_2_mean",
    "F3-A2_wavelet_level_3_mean", "F3-A2_wavelet_level_4_mean",
    "F4-A1_wavelet_level_0_mean", "F4-A1_wavelet_level_1_mean", "F4-A1_wavelet_level_2_mean",
    "F4-A1_wavelet_level_3_mean", "F4-A1_wavelet_level_4_mean",
    "O1-A2_wavelet_level_0_mean", "O1-A2_wavelet_level_1_mean", "O1-A2_wavelet_level_2_mean",
    "O1-A2_wavelet_level_3_mean", "O1-A2_wavelet_level_4_mean",
    "O2-A1_wavelet_level_0_mean", "O2-A1_wavelet_level_1_mean", "O2-A1_wavelet_level_2_mean",
    "O2-A1_wavelet_level_3_mean", "O2-A1_wavelet_level_4_mean",
    "C3-A2_wavelet_mean","C3-A2_wavelet_std","C4-A1_wavelet_mean","C4-A1_wavelet_std","F3-A2_wavelet_mean","F3-A2_wavelet_std","F4-A1_wavelet_mean","F4-A1_wavelet_std","O1-A2_wavelet_mean","O1-A2_wavelet_std","O2-A1_wavelet_mean","O2-A1_wavelet_std","C3-A2_spectral_entropy","C3-A2_sample_entropy","C4-A1_spectral_entropy","C4-A1_sample_entropy","F3-A2_spectral_entropy","F3-A2_sample_entropy","F4-A1_spectral_entropy","F4-A1_sample_entropy","O1-A2_spectral_entropy","O1-A2_sample_entropy","O2-A1_spectral_entropy","O2-A1_sample_entropy"
    "C3-A2_spectral_entropy", "C3-A2_sample_entropy",
    "C4-A1_spectral_entropy", "C4-A1_sample_entropy",
    "F3-A2_spectral_entropy", "F3-A2_sample_entropy",
    "F4-A1_spectral_entropy", "F4-A1_sample_entropy",
    "O1-A2_spectral_entropy", "O1-A2_sample_entropy",
    "O2-A1_spectral_entropy", "O2-A1_sample_entropy"
]

def clean_features(df, drop_columns=None):
    """
    Cleans the input dataframe by:
    - Dropping specified bad feature columns
    - Leaving all other features untouched (no normalization)
    """
    df = df.copy()

    if drop_columns is None:
        drop_columns = DEFAULT_DROP_COLUMNS

    df.drop(columns=drop_columns, inplace=True, errors='ignore')

    return df

def remove_empty_vectors(df, vector_col="synset_vector"):
    """
    Removes rows where the synset_vector is missing or empty.
    """
    df = df.copy()
    df = df[df[vector_col].apply(lambda x: isinstance(x, str) and len(eval(x)) > 0)]
    return df

def normalize_features(df, mode="global", group_col="trial_name"):
    """
    Normalize EEG feature columns using z-score normalization.
    Args:
        mode: 'global' (default), 'trial', or 'epoch'
        group_col: column to group by if mode is 'trial' or 'epoch'
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if mode == "global":
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    elif mode in ["trial", "epoch"]:
        df[numeric_cols] = df.groupby(group_col)[numeric_cols].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    else:
        raise ValueError("Normalization mode must be one of: global, trial, epoch")

    return df
