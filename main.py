from src.model import train_and_evaluate
from src.data_loader import load_data
from src.utils import (
    plot_confusion_matrix_matplotlib,
    calculate_metrics,
    print_metrics_table,
    print_business_insights,
    plot_metrics_bar_chart
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

def main():
    print("\n" + "=" * 70)
    print("Figure 2: Confusion Matrix from Trained Model")
    print("=" * 70)

    # Load data and train model
    print("Loading data and training model...")
    df = load_data()
    results = train_and_evaluate(df)

    y_test = results["y_test"]
    y_pred = results["y_pred"]
    labels = results["categories"]

    # Compute confusion matrix
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    # Plot confusion matrix (Figure 2)
    fig = plot_confusion_matrix_matplotlib(
        matrix,
        labels,
        title="Figure 2: Confusion Matrix\nTest Set (n=100)"
    )

    # Metrics
    metrics = calculate_metrics(matrix, labels)

    #Accuracy calculation and output
    total_correct = sum(matrix[i][i] for i in range(len(labels)))
    total_samples = matrix.sum()
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.3f} ({int(total_correct)}/{total_samples} correct)")

    print_metrics_table(metrics, labels, matrix)
    print_business_insights(metrics)

    # Generation of Figure 4
    fig4 = plot_metrics_bar_chart(metrics)
    fig4.savefig(
        "reports/figures/figure_4_metrics.png",
        dpi=300,
        bbox_inches="tight"
    )
    print("\n Figure 4 saved to reports/figures/")

    # Save confusion matrix figure (Figure 2)
    os.makedirs("reports/figures", exist_ok=True)
    fig.savefig(
        "reports/figures/figure_2_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    print("\n Figure 2 saved to reports/figures/")
    plt.show()


if __name__ == "__main__":
    main()
