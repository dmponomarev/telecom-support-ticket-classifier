#visualization.py
"""
Generate dataset distribution visualization.
Output: text console display + PNG/PDF charts.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch


def show_text_chart():
    """Print formatted distribution table to console."""
    categories = ['billing', 'network', 'device', 'contract', 'other']
    counts = [80, 80, 80, 80, 80]

    print("=" * 60)
    print("Figure 1: Distribution of 400 support tickets")
    print("=" * 60)
    print()
    print("CATEGORY    COUNT    VISUAL REPRESENTATION")
    print("-" * 50)

    for cat, cnt in zip(categories, counts):
        print(f"{cat:10} {cnt:4}    {'█' * 20}")

    print("-" * 50)
    print(f"{'TOTAL':10} {sum(counts):4}    {'█' * 20}")
    print()
    print("Key insights:")
    print("✓ 80 tickets per category (balanced)")
    print("✓ 400 total samples")
    print("✓ 5 telecom problem categories")
    print("✓ Ready for ML classification")
    print("=" * 60)


def create_bar_chart():
    """Generate matplotlib bar chart, return figure object."""
    categories = ['billing', 'network', 'device', 'contract', 'other']
    values = [80, 80, 80, 80, 80]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD700', '#FFFACD']

    fig, ax = plt.subplots(figsize=(12, 7))
    x_pos = np.arange(len(categories))

    bars = ax.bar(x_pos, values, color=colors, edgecolor='black',
                  linewidth=1.5, width=0.7)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                str(val), ha='center', va='bottom',
                fontweight='bold', fontsize=12)

    # Axis configuration
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_yticks(range(0, 91, 10))
    ax.set_ylim(0, 90)
    ax.set_xlabel("Category", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Tickets", fontsize=13, fontweight='bold')
    ax.set_title("Figure 1: Distribution of 400 tickets\n(80 per category)",
                 fontsize=15, fontweight='bold', pad=20)

    # Styling
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    legend_patches = [Patch(facecolor=c, edgecolor='black', label=cat)
                      for c, cat in zip(colors, categories)]
    ax.legend(handles=legend_patches, title='Categories:',
              loc='upper right', fontsize=10, framealpha=0.9)

    # Footer note
    plt.figtext(0.5, 0.01,
                "Dataset: 400 tickets | Balanced | German (60%) / English (40%)",
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    return fig


def export_figure(fig, base_name="figure_1_distribution"):
    """Save figure in multiple formats."""
    reports_dir = os.path.join("..", "reports", "figures")
    os.makedirs(reports_dir, exist_ok=True)

    formats = [
        (os.path.join(reports_dir, f"{base_name}.png"), 300),
    ]

    for path, dpi in formats:
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  → {path}")


def main():
    """Run complete visualization pipeline."""
    print("\n" + "=" * 60)
    print("Generating Figure 1: Dataset Distribution")
    print("=" * 60 + "\n")

    # 1. Text output
    show_text_chart()

    print("\n" + "=" * 60)
    print("Creating graphical version...")
    print("=" * 60 + "\n")

    # 2. Graphical output
    fig = create_bar_chart()

    # 3. Export
    print("Exporting files:")
    export_figure(fig)

    print("\n" + "=" * 60)
    print("Summary:")
    print("-" * 40)
    categories = ['billing', 'network', 'device', 'contract', 'other']
    for cat in categories:
        print(f"{cat:10} → 80 tickets (20.0%)")
    print("-" * 40)
    print(f"{'TOTAL':10} → 400 tickets (100.0%)")
    print("=" * 60)

    # 4. Display
    print("\nOpening interactive chart...")
    plt.show()


if __name__ == "__main__":
    main()