import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_confusion_matrix_matplotlib(matrix, labels, title):
    """
    Plot confusion matrix using matplotlib (no seaborn).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Heatmap with colorbar
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_title(title, fontsize=12, pad=15)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate cells with adaptive text color
    for i in range(len(labels)):
        for j in range(len(labels)):
            # White text on dark cells, black on light
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, matrix[i, j],
                    ha="center", va="center",
                    fontweight="bold", color=color)

    plt.tight_layout()
    return fig


def calculate_metrics(conf_matrix, labels):
    """
    Calculate precision, recall, F1-score per class.
    """
    metrics = {}

    for i, label in enumerate(labels):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        metrics[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3)
        }

    return metrics


def print_metrics_table(metrics, labels, matrix):
    """
    Print metrics in table format.
    """
    df = pd.DataFrame(metrics).T
    print("\nClassification Metrics (Test Set):\n")
    print(df)


def print_business_insights(metrics):
    """
    Print short business interpretation.
    """
    print("\nBusiness Insights:")
    for label, scores in metrics.items():
        if scores["recall"] < 0.8:
            print(f"- {label}: lower recall may cause misrouting delays")
        else:
            print(f"- {label}: reliably detected")


def plot_metrics_bar_chart(metrics, save_path=None):
    """
    Create grouped bar chart for precision, recall, F1 per class.

    Parameters:
    - metrics: dict from calculate_metrics()
    - save_path: str, optional path to save figure
    """
    categories = list(metrics.keys())
    precision = [metrics[cat]["precision"] for cat in categories]
    recall = [metrics[cat]["recall"] for cat in categories]
    f1 = [metrics[cat]["f1"] for cat in categories]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#4A90E2')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#50C878')
    bars3 = ax.bar(x + width, f1, width, label='F1-score', color='#FF6B6B')

    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel('Categories')
    ax.set_ylabel('Score')
    ax.set_title('Figure 4: Per-class performance metrics\n(Precision, Recall, F1-score)', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

    return fig