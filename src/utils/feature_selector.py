import pandas as pd
import ast

CHANNEL_TO_REGION = {
    # Eye & muscle channels (typically not brain regions, but listed for completeness)
    "LOC-A2": "eye", "ROC-A1": "eye", "EMG 1-2": "muscle",

    # Prefrontal
    "Fpz": "prefrontal", "Fp1": "prefrontal", "Fp2": "prefrontal",

    # Frontal
    "Af7": "frontal", "Af8": "frontal", "Af3": "frontal", "Af4": "frontal",
    "F7": "frontal", "F5": "frontal", "F3": "frontal", "F1": "frontal",
    "Fz": "frontal", "F2": "frontal", "F4": "frontal", "F6": "frontal", "F8": "frontal",
    "Ft7": "frontal", "Ft8": "frontal",

    # Fronto-central
    "Fc5": "fronto-central", "Fc3": "fronto-central", "Fc1": "fronto-central",
    "Fcz": "fronto-central", "Fc2": "fronto-central", "Fc4": "fronto-central",
    "Fc6": "fronto-central",

    # Central
    "T7": "temporal", "C5": "central", "C3": "central", "C1": "central",
    "Cz": "central", "C2": "central", "C4": "central", "C6": "central",
    "T8": "temporal",

    # Centro-parietal
    "Tp7": "temporo-parietal", "Cp5": "centro-parietal", "Cp3": "centro-parietal",
    "Cp1": "centro-parietal", "Cp2": "centro-parietal", "Cp4": "centro-parietal",
    "Cp6": "centro-parietal", "Tp8": "temporo-parietal",

    # Parietal
    "P7": "parietal", "P5": "parietal", "P3": "parietal", "P1": "parietal",
    "Pz": "parietal", "P2": "parietal", "P4": "parietal", "P6": "parietal", "P8": "parietal",

    # Parieto-occipital
    "Po7": "parieto-occipital", "Po3": "parieto-occipital", "Po4": "parieto-occipital", "Po8": "parieto-occipital",

    # Occipital
    "O1": "occipital", "Oz": "occipital", "O2": "occipital",

    # Reference electrodes (if applicable)
    "A1": "reference", "A2": "reference"
}

def get_region_from_feature(col_name):
    """Extract EEG region from feature column using channel mapping."""
    for ch, region in CHANNEL_TO_REGION.items():
        if col_name.startswith(ch + "_"):
            return region
    return "unknown"

def get_feature_groups(df):
    feature_groups = {
        "bandpower": lambda col: any(col.endswith(f"_{b}") for b in ["delta", "theta", "alpha", "beta"]) and not col.startswith("region_") and not "_rel_" in col,
        "rel_bandpower": lambda col: any(f"_rel_{b}" in col for b in ["delta", "theta", "alpha", "beta"]),
        "wavelet": lambda col: "_wavelet_" in col,
        "spectral_entropy": lambda col: col.endswith("_spectral_entropy"),
        "sample_entropy": lambda col: col.endswith("_sample_entropy"),
        "spectral_centroid": lambda col: col == "spectral_centroid",
        "spectral_skewness": lambda col: col == "spectral_skewness",

        "topo_band_means": lambda col: "_mean_" in col and col.startswith("region_"),
        "topo_band_stds": lambda col: "_std_" in col and col.startswith("region_"),
        "topo_band_entropies": lambda col: "_entropy_" in col and col.startswith("region_"),

        "delta": lambda col: "_delta" in col and not col.startswith("region_"),
        "theta": lambda col: "_theta" in col and not col.startswith("region_"),
        "alpha": lambda col: "_alpha" in col and not col.startswith("region_"),
        "beta": lambda col: "_beta" in col and not col.startswith("region_"),
        "rel_delta": lambda col: "_rel_delta" in col and not col.startswith("region_"),
        "rel_theta": lambda col: "_rel_theta" in col and not col.startswith("region_"),
        "rel_alpha": lambda col: "_rel_alpha" in col and not col.startswith("region_"),
        "rel_beta": lambda col: "_rel_beta" in col and not col.startswith("region_"),

        "all": lambda col: (
            df[col].dtype in ['float64', 'int64'] and
            col not in ["epoch", "subject_id", "sleep_stage", "trial_name", "dream_text", "synsets", "Meta Topic ID", "meta_topic_id"]
        )
    }
    return feature_groups

def add_region_summary_features(df):
    """
    Aggregates features by EEG region (mean across features in same region).
    """
    region_features = {}

    for col in df.columns:
        parts = col.split("_")
        if len(parts) < 2:
            continue
        ch = parts[0]
        region = CHANNEL_TO_REGION.get(ch, None)
        if region:
            region_features.setdefault(region, []).append(col)

    region_summaries = {}
    for region, cols in region_features.items():
        if cols:
            region_summaries[f"region_{region}_mean"] = df[cols].mean(axis=1)

    return pd.concat([df, pd.DataFrame(region_summaries)], axis=1)


def select_features(
    df,
    group_names,
    sleep_stage=None,
    target_column="synset_vector",
    label_df=None,
    label_column="topic_label",
    merge_on="trial_name",
    include_region_summary=False
):
    df = df.copy()

    if label_df is not None:
        if merge_on not in label_df.columns:
            raise ValueError(f"'{merge_on}' column not found in label_df")

        # Clean label_df key for merge
        label_df = label_df.copy()
        label_df[merge_on] = label_df[merge_on].str.replace(".edf", "", regex=False)
        df = df.merge(label_df[[merge_on, label_column]], on=merge_on, how="inner")
        target_column = label_column

    if sleep_stage is not None:
        df = df[df["sleep_stage"] == sleep_stage]

    groups = get_feature_groups(df)

    if "all" in group_names:
        exclude_cols = ['subject_id', 'trial_name', 'sleep_stage', target_column, "Meta Topic ID", "epoch", "meta_topic_id"]
        selected_columns = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']
    else:
        selected_columns = []
        for group in group_names:
            if group not in groups:
                print(f"[Warning] Feature group '{group}' not found.")
                continue
            is_in_group = groups[group]
            selected_columns.extend([col for col in df.columns if is_in_group(col)])

    X = df[selected_columns].copy()

    if include_region_summary:
        X = add_region_summary_features(X)

    y = df[target_column].copy() if target_column in df.columns else None
    return X, y

def load_report_embeddings(path="data/processed/merged_embedded_reports.csv", vector_column="cleaned_vector"):
    """
    Loads report embeddings and returns a dataframe with trial_name and selected vector_column.

    Args:
        path (str): Path to the CSV file with embedded reports.
        vector_column (str): One of "cleaned_vector", "biasfree_vector", "trimmed_vector".

    Returns:
        pd.DataFrame with columns ["trial_name", vector_column] where vectors are parsed as lists.
    """
    df = pd.read_csv(path)
    df["trial_name"] = df["trial_name"].str.replace(".edf", "", regex=False)
    df[vector_column] = df[vector_column].apply(ast.literal_eval)
    return df[["trial_name", vector_column]]