import os
from datetime import datetime
import csv
import pandas as pd, os
import numpy as np
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def safe_mean(key, logs, default=0.0):
    values = [f[key] for f in logs if f.get(key) is not None]
    return round(np.mean(values), 6) if values else default

def log_train(model_name, epochs, train_loss, val_loss, train_size, val_size, hidden_size, batch_size, lr, save_path, feature_groups, sleep_stage, target_column):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(LOG_DIR, "train_log.txt"), "a") as f:
        f.write(f"[{now}]\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Feature Groups: {feature_groups}\n")
        f.write(f"Sleep Stage Filter: {sleep_stage}\n")
        f.write(f"Target Column: {target_column}\n")
        f.write(f"Epochs: {epochs if epochs is not None else 'N/A (classic model)'}\n")
        f.write(f"Train Loss: {train_loss:.4f}\n" if train_loss is not None else "Train Loss: N/A (classic model)\n")
        f.write(f"Val Loss: {val_loss:.4f}\n" if val_loss is not None else "Val Loss: N/A (classic model)\n")
        f.write(f"Train Size: {train_size}, Val Size: {val_size}\n")
        f.write(f"Hidden Size: {hidden_size if hidden_size is not None else 'N/A'}, Batch Size: {batch_size if batch_size is not None else 'N/A'}, Learning Rate: {lr if lr is not None else 'N/A'}\n")
        f.write(f"Saved to: {save_path}\n")
        f.write("\n")

def log_eval(model_name, top1, top5, avg_cosine, log_file, predictions_path=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(log_file, "a") as f:
        f.write(f"[{now}]\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Top-1 Accuracy: {top1:.4f}\n")
        f.write(f"Top-5 Accuracy: {top5:.4f}\n")
        f.write(f"Average Cosine Similarity: {avg_cosine:.4f}\n")
        if predictions_path:
            f.write(f"Predictions Saved At: {predictions_path}\n")
        f.write("="*40 + "\n")

def log_experiment(
    log_path,
    experiment_id,
    model_name,
    hidden_dims,
    dropout_rate,
    feature_groups,
    sleep_stage,
    target_column,
    train_loss,
    val_loss,
    top1,
    top5,
    avg_cosine,
    notes=""
):
    file_exists = os.path.isfile(log_path)

    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "experiment_id",
                "timestamp",
                "model_name",
                "hidden_dims",
                "dropout_rate",
                "feature_groups",
                "sleep_stage",
                "target_column",
                "train_loss",
                "val_loss",
                "top1_accuracy",
                "top5_accuracy",
                "average_cosine",
                "notes"
            ])

        writer.writerow([
            experiment_id,
            datetime.now().isoformat(),
            model_name,
            str(hidden_dims) if hidden_dims else "N/A",
            dropout_rate if dropout_rate is not None else "N/A",
            str(feature_groups),
            sleep_stage if sleep_stage else "None",
            target_column,
            round(train_loss, 4) if train_loss is not None else "N/A",
            round(val_loss, 4) if val_loss is not None else "N/A",
            round(top1, 4),
            round(top5, 4),
            round(avg_cosine, 4),
            notes
        ])

def log_full_experiment(
    experiment_id,
    model_name,
    feature_groups,
    train_loss,
    val_loss,
    top1_acc,
    top3_acc, 
    f1_score,
    cosine_sim,
    train_size,
    val_size,
    hidden_dims,
    dropout_rate,
    classic_params,
    batch_size,
    lr,
    epochs,
    sleep_stage,
    target_column,
    notes,
    log_path="logs/experiments/model_exp_log.csv"
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    new_entry = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "feature_groups": ", ".join(feature_groups) if isinstance(feature_groups, list) else feature_groups,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,  
        "f1_score": f1_score,
        "cosine_similarity": cosine_sim,
        "train_size": train_size,
        "val_size": val_size,
        "hidden_dims": str(hidden_dims) if hidden_dims else None,
        "dropout_rate": dropout_rate,
        "classic_params": str(classic_params) if classic_params else None,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "sleep_stage": sleep_stage,
        "target_column": target_column,
        "notes": notes
    }

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)

        if experiment_id in df["experiment_id"].values:
            idx = df.index[df["experiment_id"] == experiment_id][0]
            for key, value in new_entry.items():
                if pd.notna(value):
                    df.at[idx, key] = value
            print(f"Updated log for existing experiment_id: {experiment_id}")
        else:
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            print(f"[+] Logged new experiment: {experiment_id}")
    else:
        df = pd.DataFrame([new_entry])
        print(f"[+] Created log and added experiment: {experiment_id}")

    df.to_csv(log_path, index=False)

def log_trainval_fold(
    experiment_id,
    fold_id,
    model_name,
    feature_groups,
    train_loss,
    val_loss,
    top1_acc,
    top3_acc,
    f1_score,
    manhattan_dist,
    train_size,
    val_size,
    hidden_dims,
    dropout_rate,
    classic_params,
    batch_size,
    lr,
    epochs,
    sleep_stage,
    target_column,
    notes,
    log_path="logs/experiments/trainval_multifold_log.csv"
):
    row = {
        "experiment_id": experiment_id,
        "fold_id": fold_id,
        "model_name": model_name,
        "feature_groups": "|".join(feature_groups) if isinstance(feature_groups, list) else feature_groups,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "f1_score": f1_score,
        "manhattan_dist": manhattan_dist,
        "train_size": train_size,
        "val_size": val_size,
        "hidden_dims": hidden_dims,
        "dropout_rate": dropout_rate,
        "classic_params": str(classic_params),
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "sleep_stage": sleep_stage,
        "target_column": target_column,
        "notes": notes
    }

    
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)
        df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
    else:
        df_log = pd.DataFrame([row])

    df_log.to_csv(log_path, index=False)

def log_trainval_average(
    experiment_id,
    model_name,
    fold_logs,
    feature_groups,
    hidden_dims,
    dropout_rate,
    classic_params,
    batch_size,
    lr,
    epochs,
    sleep_stage,
    target_column,
    notes,
    log_path="logs/experiments/multi_model_exp_log.csv"
):
    base_log = fold_logs[0]
    avg_row = {
    "experiment_id": experiment_id,
    "model_name": model_name,
    "feature_groups": feature_groups if isinstance(feature_groups, str) else ",".join(feature_groups),
    "train_loss": safe_mean("train_loss", fold_logs),
    "val_loss": safe_mean("val_loss", fold_logs),
    "top1_acc": safe_mean("top1_acc", fold_logs),
    "top3_acc": safe_mean("top3_acc", fold_logs),
    "f1_score": safe_mean("f1_score", fold_logs),
    "manhattan_dist": safe_mean("manhattan_dist", fold_logs),
    "train_size": "",  # optional to fill
    "val_size": sum([f["val_size"] for f in fold_logs]),
    "hidden_dims": base_log.get("hidden_dims", ""),
    "dropout_rate": base_log.get("dropout_rate", ""),
    "classic_params": base_log.get("classic_params", ""),
    "batch_size": base_log.get("batch_size", ""),
    "lr": base_log.get("lr", ""),
    "epochs": base_log.get("epochs", ""),
    "sleep_stage": base_log.get("sleep_stage", ""),
    "target_column": base_log.get("target_column", ""),
    "notes": notes
}
    # Load or create log file
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)

        if experiment_id in df["experiment_id"].values:
            idx = df.index[df["experiment_id"] == experiment_id][0]
            for key, value in avg_row.items():
                df.at[idx, key] = value
            print(f"[↑] Updated average log for experiment_id: {experiment_id}")
        else:
            df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
            print(f"[+] Logged new average for experiment_id: {experiment_id}")
    else:
        df = pd.DataFrame([avg_row])
        print(f"[+] Created new log file and logged experiment_id: {experiment_id}")

    df.to_csv(log_path, index=False)