"""Loading, splitting and batching the EuroSAT RGB images."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import CLASSES, IMG_SIZE, SEED, TEST_FRACTION, VAL_FRACTION

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif"}


@dataclass
class Split:
    paths: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def list_images(data_dir: str | Path, classes: list[str] = CLASSES) -> tuple[np.ndarray, np.ndarray]:
    """Return image paths and integer labels from a folder with one sub-folder per class."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_dir}. Run `python -m eurosat_cnn.download` first "
            "or pass --data-dir pointing to the extracted EuroSAT_RGB folder."
        )
    paths, labels = [], []
    for idx, name in enumerate(classes):
        files = sorted(p for p in (data_dir / name).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"No images found for class '{name}' in {data_dir / name}")
        paths += [str(p) for p in files]
        labels += [idx] * len(files)
    return np.array(paths), np.array(labels, dtype=np.int32)


def stratified_split(
    paths: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = SEED,
) -> tuple[Split, Split, Split]:
    """Split into train/val/test, keeping the class proportions identical in each part."""
    p_train, p_test, y_train, y_test = train_test_split(
        paths, labels, test_size=test_fraction, stratify=labels, random_state=seed
    )
    rel_val = val_fraction / (1.0 - test_fraction)
    p_train, p_val, y_train, y_val = train_test_split(
        p_train, y_train, test_size=rel_val, stratify=y_train, random_state=seed
    )
    return Split(p_train, y_train), Split(p_val, y_val), Split(p_test, y_test)


def subsample(split: Split, fraction: float, seed: int = SEED) -> Split:
    """Keep a stratified fraction of a split (useful for quick test runs)."""
    if fraction >= 1.0:
        return split
    keep, _, y_keep, _ = train_test_split(
        split.paths, split.labels, train_size=fraction, stratify=split.labels, random_state=seed
    )
    return Split(keep, y_keep)


def _load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))          # no-op for 64x64 tiles
    img = tf.cast(img, tf.float32) / 255.0                      # scale pixels to [0, 1]
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img, label


def make_dataset(split: Split, batch_size: int = 64, shuffle: bool = False, seed: int = SEED) -> tf.data.Dataset:
    """Build a cached, batched tf.data pipeline for one split."""
    ds = tf.data.Dataset.from_tensor_slices((split.paths, split.labels))
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if shuffle:
        ds = ds.shuffle(len(split), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_splits(data_dir: str | Path, batch_size: int = 64, fraction: float = 1.0):
    """Convenience wrapper: list files, split them and build the three datasets.

    ``fraction`` < 1 trains and validates on a stratified subset for quick checks;
    the test split is never reduced, so final scores are always comparable.
    """
    paths, labels = list_images(data_dir)
    train, val, test = stratified_split(paths, labels)
    train, val = subsample(train, fraction), subsample(val, fraction)
    return (
        (train, make_dataset(train, batch_size, shuffle=True)),
        (val, make_dataset(val, batch_size)),
        (test, make_dataset(test, batch_size)),
    )
