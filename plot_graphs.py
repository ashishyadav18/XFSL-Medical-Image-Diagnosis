import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Academic styling for publication
sns.set_theme(style="whitegrid", context="talk")

LOG_DIR = "logs"
OUTPUT_DIR = "outputs/graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# SAFE CSV LOADER
# =========================
def safe_read(path):
    if not os.path.exists(path):
        print(f"Skipping missing file: {path}")
        return None

    df = pd.read_csv(path, sep=None, engine="python")
    # Clean column names to lowercase and remove spaces for easy targeting
    df.columns = df.columns.str.strip().str.replace("\ufeff", "").str.lower()
    return df

# =========================
# 1. TRAINING CURVES
# =========================
def plot_training():
    df = safe_read(f"{LOG_DIR}/training_log.csv")
    if df is None:
        return

    # Plot 1: Acc & Loss
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(df["epoch"], df["train_acc"], marker='o', label="Train Acc", color='blue')
    ax1.plot(df["epoch"], df["val_acc"], marker='s', label="Val Acc", color='cyan')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right')
    plt.title("Training vs Validation Accuracy")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_training_accuracy.png", dpi=300)
    plt.close()

    # Plot 2: Advanced Metrics (F1, MCC, AUC)
    if "f1" in df.columns and "mcc" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df["epoch"], df["f1"], marker='o', label="Macro F1", color='purple')
        ax.plot(df["epoch"], df["mcc"], marker='s', label="MCC", color='orange')
        ax.plot(df["epoch"], df["auc"], marker='^', label="AUC", color='green')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.title("Advanced Validation Metrics over Epochs")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/2_training_advanced_metrics.png", dpi=300)
        plt.close()

# =========================
# 2. ABLATION CURVES
# =========================
def plot_ablation():
    df = safe_read(f"{LOG_DIR}/ablation_results.csv")
    if df is None:
        return

    # Accuracy with Error Bars
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(df["k_shots"], df["mean_accuracy"], yerr=df["std_deviation"], marker='o', capsize=5, label="Mean Acc", color='blue')
    ax.set_xlabel("Number of Shots (K)")
    ax.set_ylabel("Accuracy (%)")
    plt.title("Effect of Support Set Size on Accuracy")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_ablation_accuracy.png", dpi=300)
    plt.close()

    # Advanced Metrics Ablation (Acc vs F1 vs MCC)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["k_shots"], df["mean_accuracy"] / 100, marker='o', label="Accuracy", color='blue') # Normalized to 0-1
    ax.plot(df["k_shots"], df["mean_f1"], marker='s', label="Macro F1", color='purple')
    ax.plot(df["k_shots"], df["mean_mcc"], marker='^', label="MCC", color='orange')
    ax.set_xlabel("Number of Shots (K)")
    ax.set_ylabel("Score (0.0 - 1.0)")
    plt.title("Model Robustness vs. Shot Size")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_ablation_advanced_metrics.png", dpi=300)
    plt.close()

# =========================
# 3. PER-CLASS ABLATION
# =========================
def plot_per_class_ablation():
    df = safe_read(f"{LOG_DIR}/per_class_ablation.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="k_shots", y="accuracy", hue="class", marker="o", palette="tab10", ax=ax)
    ax.set_xlabel("Number of Shots (K)")
    ax.set_ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy vs Shot Size")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_per_class_ablation.png", dpi=300)
    plt.close()

# =========================
# 4. OOD THRESHOLD
# =========================
def plot_ood():
    df = safe_read(f"{LOG_DIR}/ood_results.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["threshold"], df["far_ood_detection (%)"], marker='o', label="Far-OOD", color='red')
    ax.plot(df["threshold"], df["near_ood_detection (%)"], marker='s', label="Near-OOD", color='darkorange')
    ax.set_xlabel("Cosine Similarity Threshold")
    ax.set_ylabel("Detection Rate (%)")
    plt.title("Out-of-Distribution Detection vs Threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_ood_threshold.png", dpi=300)
    plt.close()

# =========================
# 5. K=20 MASTER PLOTS
# =========================
def plot_k20_analysis():
    df = safe_read(f"{LOG_DIR}/k20_detailed_logs.csv")
    if df is None:
        return

    # Check if correct columns exist
    if "true_class" not in df.columns or "predicted_class" not in df.columns:
        print("Skipping K=20 plots: missing true/predicted columns.")
        return

    # Create a 'Correct' boolean column for splitting distributions
    df["correct"] = df["true_class"] == df["predicted_class"]

    # --- A. Pooled Confusion Matrix ---
    labels = sorted(set(df["true_class"]).union(set(df["predicted_class"])))
    cm = confusion_matrix(df["true_class"], df["predicted_class"], labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.title("Pooled Confusion Matrix (Optimal K=20 Baseline)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/7_confusion_matrix.png", dpi=300)
    plt.close()

    # --- B. Confidence Distribution (Correct vs Incorrect) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(data=df, x="confidence", hue="correct", fill=True, common_norm=False, palette={True: "green", False: "red"}, ax=ax)
    ax.set_xlabel("Cosine Similarity Confidence")
    ax.set_ylabel("Density")
    plt.title("Confidence Distribution: Correct vs Incorrect")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/8_confidence_distribution.png", dpi=300)
    plt.close()

    # --- C. Similarity Gap Distribution (Correct vs Incorrect) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(data=df, x="similarity_gap", hue="correct", fill=True, common_norm=False, palette={True: "blue", False: "red"}, ax=ax)
    ax.set_xlabel("Similarity Gap (\u0394)") # Delta symbol
    ax.set_ylabel("Density")
    plt.title("Similarity Gap (\u0394): Correct vs Incorrect")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/9_similarity_gap.png", dpi=300)
    plt.close()

# =========================
# MAIN
# =========================
def main():
    print("Generating Academic Publication Graphs...\n")

    plot_training()
    plot_ablation()
    plot_per_class_ablation()
    plot_ood()
    plot_k20_analysis()

    print("\n✅ All graphs successfully saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()