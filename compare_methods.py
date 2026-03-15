from pathlib import Path
import os
import matplotlib.pyplot as plt

from model_utils import build_training_database, evaluate_folder

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FOLDER = BASE_DIR / "dataset_FVC2000_DB4_B" / "dataset" / "train_data"
TEST_FOLDER = BASE_DIR / "dataset_FVC2000_DB4_B" / "dataset" / "real_data"
RESULTS_DIR = BASE_DIR / "results"

METHODS = [
    ("SIFT+KNN", "sift"),
    ("Gaussian+SIFT", "gaussian_sift"),
    ("Gabor+SIFT", "gabor_sift"),
    ("LBP+SIFT", "lbp_sift"),
]


RESNET18_REFERENCE = 83.16


def compare_methods(ratio=0.70, include_resnet=True):
    scores = []

    for display_name, method_key in METHODS:
        print(f"\n[INFO] Running method: {display_name}")
        database = build_training_database(str(TRAIN_FOLDER), method=method_key)
        accuracy, _ = evaluate_folder(str(TEST_FOLDER), database, ratio=ratio, method=method_key)
        percent = accuracy * 100
        scores.append((display_name, percent))
        print(f"[RESULT] {display_name}: {percent:.2f}%")

    if include_resnet:
        scores.insert(1, ("ResNet18", RESNET18_REFERENCE))

    return scores


def plot_results(scores):
    if not RESULTS_DIR.exists():
        os.makedirs(RESULTS_DIR)

    labels = [x[0] for x in scores]
    values = [x[1] for x in scores]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(labels, values, color="#4C72B0", edgecolor="black")

    ax.set_title(
        "Fingerprint Recognition Methods Comparison",
        fontsize=14,
        pad=20,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.xticks(fontsize=11)
    plt.tight_layout()

    output_path = RESULTS_DIR / "method_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"[INFO] Plot saved in: {output_path}")


def main():
    scores = compare_methods(ratio=0.70, include_resnet=True)
    plot_results(scores)


if __name__ == "__main__":
    main()
