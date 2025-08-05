import pandas as pd

def balance_trials_by_topic(df, topic_col="topic_label", trial_col="trial_name", seed=42):
    """
    Oversample trials so that each topic has equal representation by duplicating entire trials.
    Each duplicated trial's epoch rows are copied as-is.

    Args:
        df (pd.DataFrame): Full feature DataFrame (e.g. multiple rows per trial).
        topic_col (str): Column name for topic labels.
        trial_col (str): Column name for trial identifiers (to duplicate whole trials).
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Oversampled, balanced DataFrame.
    """
    import numpy as np
    np.random.seed(seed)

    trial_topics = df[[trial_col, topic_col]].drop_duplicates()
    topic_counts = trial_topics[topic_col].value_counts()
    target_n = topic_counts.max()

    balanced_dfs = []

    for topic, count in topic_counts.items():
        topic_trials = trial_topics[trial_topics[topic_col] == topic][trial_col].tolist()
        n_needed = target_n - count
        n_clones = int(np.ceil(n_needed / count))

        original = df[df[topic_col] == topic]
        balanced_dfs.append(original)

        if n_needed > 0:
            clones = []
            while len(clones) < n_needed:
                sampled_trials = np.random.choice(topic_trials, size=n_needed, replace=True)
                for t in sampled_trials:
                    trial_df = df[df[trial_col] == t].copy()
                    trial_df[trial_col] = trial_df[trial_col] + f"_dup{np.random.randint(10000)}"
                    clones.append(trial_df)
                    if len(clones) >= n_needed:
                        break
            balanced_dfs.extend(clones)

    full_df = pd.concat(balanced_dfs, ignore_index=True)
    return full_df.sample(frac=1, random_state=seed).reset_index(drop=True)
