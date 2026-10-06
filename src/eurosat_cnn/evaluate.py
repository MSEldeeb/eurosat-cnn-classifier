"""Evaluate trained models on the held-out test set and create report figures.

Example:
    python -m eurosat_cnn.evaluate --models baseline improved
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import keras

from .config import CLASSES, DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from .data import list_images, make_dataset, stratified_split
from .models import RandomRot90  # noqa: F401  (registers the custom layer for loading)


def plot_history(names: list[str]):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in names:
        path = REPORTS_DIR / f"{name}_history.csv"
        if not path.exists():
            continue
        h = pd.read_csv(path)
        epochs = h["epoch"] + 1
        axes[0].plot(epochs, h["accuracy"], "--", label=f"{name} train")
        axes[0].plot(epochs, h["val_accuracy"], label=f"{name} validation")
        axes[1].plot(epochs, h["loss"], "--", label=f"{name} train")
        axes[1].plot(epochs, h["val_loss"], label=f"{name} validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, name: str):
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set(xlabel="Predicted class", ylabel="True class", title=f"Confusion matrix ({name} model, test set)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of true class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)


def plot_examples(images, y_true, y_pred, probs, name: str, wrong: bool):
    idx = np.where((y_true != y_pred) if wrong else (y_true == y_pred))[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(idx, size=min(12, len(idx)), replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(13, 5))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(images[i])
        ax.axis("off")
        colour = "firebrick" if wrong else "darkgreen"
        ax.set_title(f"true: {CLASSES[y_true[i]]}\npred: {CLASSES[y_pred[i]]} ({probs[i].max():.0%})",
                     fontsize=7.5, color=colour)
    for ax in list(axes.flat)[len(idx):]:
        ax.axis("off")
    kind = "misclassified" if wrong else "correct"
    fig.suptitle(f"{kind.capitalize()} test images ({name} model)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{kind}_examples_{name}.png", dpi=150)
    plt.close(fig)


def evaluate(name: str, data_dir: str):
    paths, labels = list_images(data_dir)
    _, _, test = stratified_split(paths, labels)
    test_ds = make_dataset(test, batch_size=128)

    model = keras.models.load_model(MODELS_DIR / f"{name}.keras")
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = test.labels

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {name} model: test accuracy {acc:.2%} on {len(y_true)} images ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

    images = np.concatenate([x.numpy() for x, _ in test_ds])
    plot_confusion(cm, name)
    plot_examples(images, y_true, y_pred, probs, name, wrong=True)
    plot_examples(images, y_true, y_pred, probs, name, wrong=False)

    # most frequent confusions
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], CLASSES[i], CLASSES[j]) for i in range(len(CLASSES)) for j in range(len(CLASSES))),
                   reverse=True)[:5]
    metrics = {
        "model": name,
        "test_images": int(len(y_true)),
        "test_accuracy": round(float(acc), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "per_class_f1": {c: round(float(report[c]["f1-score"]), 4) for c in CLASSES},
        "top_confusions": [{"true": t, "predicted": p, "count": int(n)} for n, t, p in pairs if n],
        "confusion_matrix": cm.tolist(),
    }
    (REPORTS_DIR / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["baseline", "improved"])
    p.add_argument("--data-dir", default=str(DATA_DIR))
    args = p.parse_args(argv)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = [evaluate(m, args.data_dir) for m in args.models if (MODELS_DIR / f"{m}.keras").exists()]
    plot_history(args.models)
    print("\nSummary")
    for r in results:
        print(f"  {r['model']:<9} accuracy={r['test_accuracy']:.2%}  macro-F1={r['macro_f1']:.3f}")
    write_results_markdown(results)


def write_results_markdown(results: list[dict]):
    """Write reports/results.md: a ready-to-paste results table for the README."""
    import json as _json

    lines = ["| Model | Parameters | Epochs | Training time | Test accuracy | Macro F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        info_path = REPORTS_DIR / f"{r['model']}_training.json"
        info = _json.loads(info_path.read_text()) if info_path.exists() else {}
        lines.append(
            f"| {r['model'].capitalize()} CNN | {info.get('parameters', 0):,} | {info.get('epochs_run', '-')} | "
            f"{info.get('training_minutes', '-')} min ({info.get('hardware', '-')}) | "
            f"**{r['test_accuracy']:.2%}** | {r['macro_f1']:.3f} |"
        )
    if results:
        best = max(results, key=lambda r: r["test_accuracy"])
        lines += ["", f"Most frequent confusions ({best['model']} model):", ""]
        lines += [f"- {c['true']} predicted as {c['predicted']}: {c['count']} images" for c in best["top_confusions"][:3]]
    table = "\n".join(lines) + "\n"
    (REPORTS_DIR / "results.md").write_text(table)
    print(f"\nResults table written to {REPORTS_DIR / 'results.md'}")

    # Also update the README between the RESULTS markers, if present.
    from .config import PROJECT_ROOT

    readme = PROJECT_ROOT / "README.md"
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if start in text and end in text:
            head, rest = text.split(start, 1)
            _, tail = rest.split(end, 1)
            readme.write_text(f"{head}{start}\n{table}{end}{tail}", encoding="utf-8")
            print("README.md results section updated")


if __name__ == "__main__":
    main()
