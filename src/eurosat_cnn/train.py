"""Train a CNN on EuroSAT.

Example:
    python -m eurosat_cnn.train --model improved --epochs 30
"""
from __future__ import annotations

import argparse
import json
import time

import tensorflow as tf
import keras

from .config import DATA_DIR, MODELS_DIR, REPORTS_DIR, SEED
from .data import load_splits
from .models import build_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["baseline", "improved"], default="improved")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=6, help="early-stopping patience (epochs)")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="use only this share of train/val images, e.g. 0.1 for a quick check")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    keras.utils.set_random_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    (train, train_ds), (val, val_ds), (test, _) = load_splits(args.data_dir, args.batch_size, args.fraction)
    print(f"Images: train={len(train)}, val={len(val)}, test={len(test)}")

    model = build_model(args.model, args.learning_rate)
    model.summary()

    ckpt_path = MODELS_DIR / f"{args.model}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        keras.callbacks.CSVLogger(REPORTS_DIR / f"{args.model}_history.csv"),
    ]

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
    minutes = (time.time() - start) / 60

    summary = {
        "model": args.model,
        "parameters": int(model.count_params()),
        "epochs_run": len(history.history["loss"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "training_minutes": round(minutes, 1),
        "hardware": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        "settings": vars(args),
    }
    (REPORTS_DIR / f"{args.model}_training.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Best model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
