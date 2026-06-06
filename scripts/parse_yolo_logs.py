import re
import numpy as np
import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_yolo_log(file_path):
    """
    Parses a YOLOv8 training log file and extracts training metrics per epoch for multiple folds.
    It takes the last recorded value for each epoch. A new fold is detected when the epoch number drops.
    """
    pattern = re.compile(
        r"^\s*(\d+)/\d+\s+[\d.]+[a-zA-Z]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+:"
    )

    folds_data = []
    current_epoch_data = {}
    last_epoch = -1

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                box_loss = float(match.group(2))
                cls_loss = float(match.group(3))
                dfl_loss = float(match.group(4))

                # If epoch number is smaller than last_epoch, we've started a new fold
                if epoch < last_epoch:
                    folds_data.append(current_epoch_data)
                    current_epoch_data = {}

                current_epoch_data[epoch] = {
                    "box_loss": box_loss,
                    "cls_loss": cls_loss,
                    "dfl_loss": dfl_loss,
                }
                last_epoch = epoch

    if current_epoch_data:
        folds_data.append(current_epoch_data)

    return folds_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse YOLO training logs into numpy arrays for all CV folds"
    )
    parser.add_argument(
        "--log-file", type=str, required=True, help="Path to the runs.txt log file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/yolo_metrics.npz",
        help="Output npz file path",
    )
    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"Error: Log file {args.log_file} does not exist.")
        exit(1)

    folds_data = parse_yolo_log(args.log_file)

    if not folds_data:
        print("Failed to extract any metrics. Check your log file format.")
        exit(1)

    print(f"Successfully extracted metrics for {len(folds_data)} folds.")

    npz_dict = {}

    # Calculate grid size for plots
    num_folds = len(folds_data)
    cols = 2
    rows = (num_folds + 1) // 2
    plt.figure(figsize=(12, 5 * rows))

    for i, fold_data in enumerate(folds_data):
        fold_idx = i + 1
        epochs = np.array(sorted(list(fold_data.keys())))
        box_losses = np.array([fold_data[e]["box_loss"] for e in epochs])
        cls_losses = np.array([fold_data[e]["cls_loss"] for e in epochs])
        dfl_losses = np.array([fold_data[e]["dfl_loss"] for e in epochs])

        npz_dict[f"fold{fold_idx}_epochs"] = epochs
        npz_dict[f"fold{fold_idx}_box_loss"] = box_losses
        npz_dict[f"fold{fold_idx}_cls_loss"] = cls_losses
        npz_dict[f"fold{fold_idx}_dfl_loss"] = dfl_losses

        # Quick summary stats
        print(f"\nFold {fold_idx} - Final Epoch ({epochs[-1]}):")
        print(f"  Box Loss: {box_losses[-1]:.4f}")
        print(f"  Cls Loss: {cls_losses[-1]:.4f}")
        print(f"  Dfl Loss: {dfl_losses[-1]:.4f}")

        # Add to subplot
        plt.subplot(rows, cols, fold_idx)
        plt.plot(epochs, box_losses, label="Box Loss", linewidth=2)
        plt.plot(epochs, cls_losses, label="Class Loss", linewidth=2)
        plt.plot(epochs, dfl_losses, label="DFL Loss", linewidth=2)

        plt.title(f"Fold {fold_idx} Training Losses", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.7)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **npz_dict)
    print(f"\nSaved numpy arrays to {args.output}")
    print(f"Keys available in npz: {list(npz_dict.keys())}")

    plt.tight_layout()
    plot_path = Path(args.output).parent / "yolo_training_curves_cv.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved training curves plot to {plot_path}")
