import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe on headless cluster
import matplotlib.pyplot as plt
import numpy as np

from config import Config


def setup_output_dir(cfg: Config, config_path: Optional[str] = None) -> logging.Logger:
    """Create the output directory and return a logger writing to file + console.

    Output folder layout:
        checkpoint_dir/
        ├── config.yaml          (copy of the run config)
        ├── train.log
        ├── best_model.pt
        └── training_curves.png
    """
    save_dir = Path(cfg.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if config_path is not None:
        shutil.copyfile(config_path, save_dir / Path(config_path).name)

    logger = logging.getLogger("ConditionalINR")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(save_dir / "train.log", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def save_training_plots(
    history: Dict,
    save_dir: str,
    feature_supervision: bool = False,
    tv_weight: float = 0.0,
    class_names: Optional[List[str]] = None,
    class_colors: Optional[np.ndarray] = None,
) -> None:
    """Save (and overwrite) training_curves.png with loss, validation, and per-class Dice."""
    epochs     = history["epoch"]
    val_epochs = history["val_epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    # ------------------------------------------------------------------ #
    # Panel 1 — training loss components                                  #
    # ------------------------------------------------------------------ #
    ax = axes[0]
    ax.plot(epochs, history["train_loss"],  label="Total",      linewidth=2)
    ax.plot(epochs, history["ce_loss"],     label="CE",         linestyle="--")
    ax.plot(epochs, history["dice_loss"],   label="Dice",       linestyle="--")
    ax.plot(epochs, history["smooth_loss"], label="Smoothness", linestyle="--")
    if feature_supervision:
        ax.plot(epochs, history["feat_sup_loss"], label="FeatSup", linestyle="--")
    if tv_weight > 0 and "tv_loss" in history:
        ax.plot(epochs, history["tv_loss"], label="TV", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Panel 2 — val mean Dice (left) + val loss (right)                   #
    # ------------------------------------------------------------------ #
    ax2 = axes[1]
    ax2.plot(val_epochs, history["val_dice_mean"],
             color="steelblue", linewidth=2, label="Val Dice (mean)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice", color="steelblue")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.set_title("Validation")
    ax2.grid(True, alpha=0.3)

    ax3 = ax2.twinx()
    ax3.plot(val_epochs, history["val_loss"],
             color="coral", linestyle=":", linewidth=1.5, label="Val loss")
    ax3.set_ylabel("Loss", color="coral")
    ax3.tick_params(axis="y", labelcolor="coral")

    lines = ax2.get_legend_handles_labels()[0] + ax3.get_legend_handles_labels()[0]
    lbls  = ax2.get_legend_handles_labels()[1] + ax3.get_legend_handles_labels()[1]
    ax2.legend(lines, lbls, loc="lower right")

    # ------------------------------------------------------------------ #
    # Panel 3 — per-class Dice over validation epochs                     #
    # ------------------------------------------------------------------ #
    ax4 = axes[2]
    if history["val_dice_per_class"]:
        val_dice_pc = np.array(history["val_dice_per_class"])  # [V, num_classes]
        num_classes = val_dice_pc.shape[1]
        for c in range(num_classes):
            class_dice = val_dice_pc[:, c]
            if np.all(np.isnan(class_dice)):
                continue
            name  = class_names[c]  if class_names  is not None else f"Class {c}"
            color = tuple(class_colors[c]) if class_colors is not None else None
            ax4.plot(val_epochs, class_dice, label=name, color=color, linewidth=1.5)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Dice")
    ax4.set_title("Per-class Dice (validation)")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(fontsize=7, loc="lower right", ncol=2)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(save_dir) / "training_curves.png", dpi=120)
    plt.close(fig)