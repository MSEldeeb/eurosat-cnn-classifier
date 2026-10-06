"""Exploratory data analysis: class balance, example tiles and colour statistics.

Example:
    python -m eurosat_cnn.eda
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import CLASSES, DATA_DIR, FIGURES_DIR, REPORTS_DIR
from .data import list_images, stratified_split


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    args = p.parse_args(argv)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    paths, labels = list_images(args.data_dir)
    counts = np.bincount(labels, minlength=len(CLASSES))
    train, val, test = stratified_split(paths, labels)

    # 1) class balance
    fig, ax = plt.subplots(figsize=(9, 4))
    order = np.argsort(counts)[::-1]
    ax.bar([CLASSES[i] for i in order], counts[order], color="#2b8a9e")
    for k, i in enumerate(order):
        ax.text(k, counts[i] + 30, str(counts[i]), ha="center", fontsize=8)
    ax.set(ylabel="Number of images", title=f"EuroSAT RGB: {len(labels):,} images in {len(CLASSES)} classes")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 2) example tiles: 5 per class
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(len(CLASSES), 5, figsize=(6.5, 13))
    for r, name in enumerate(CLASSES):
        choice = rng.choice(np.where(labels == r)[0], 5, replace=False)
        for c, i in enumerate(choice):
            axes[r, c].imshow(Image.open(paths[i]))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("Five random tiles per class (64 x 64 px, 10 m per pixel)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sample_grid.png", dpi=150)
    plt.close(fig)

    # 3) mean colour per class (sample of 300 images each)
    means = []
    for r in range(len(CLASSES)):
        idx = rng.choice(np.where(labels == r)[0], 300, replace=False)
        arr = np.stack([np.asarray(Image.open(paths[i]), dtype=np.float32) for i in idx])
        means.append(arr.mean(axis=(0, 1, 2)) / 255)
    means = np.array(means)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(CLASSES))
    for k, (ch, col) in enumerate(zip("RGB", ["#d62728", "#2ca02c", "#1f77b4"])):
        ax.bar(x + (k - 1) * 0.27, means[:, k], 0.27, label=ch, color=col)
    ax.set_xticks(x, CLASSES, rotation=40, ha="right")
    ax.set(ylabel="Mean pixel value (0-1)", title="Average colour per class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mean_colour_per_class.png", dpi=150)
    plt.close(fig)

    summary = {
        "total_images": int(len(labels)),
        "image_size": "64x64 RGB",
        "images_per_class": {c: int(n) for c, n in zip(CLASSES, counts)},
        "imbalance_ratio_max_min": round(float(counts.max() / counts.min()), 2),
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "mean_rgb_per_class": {c: [round(float(v), 3) for v in m] for c, m in zip(CLASSES, means)},
    }
    (REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
