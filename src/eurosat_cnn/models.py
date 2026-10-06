"""Two CNN architectures: a simple baseline and an improved, regularised model."""
from __future__ import annotations

import tensorflow as tf
import keras
from keras import layers

from .config import CLASSES, IMG_SIZE, NUM_CHANNELS

INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, NUM_CHANNELS)


@keras.saving.register_keras_serializable(package="eurosat_cnn")
class RandomRot90(layers.Layer):
    """Rotate each image by a random multiple of 90 degrees during training.

    Satellite tiles have no 'up' direction, so a rotated field is still a field.
    Unlike arbitrary-angle rotation, 90-degree steps need no interpolation or padding.
    """

    def call(self, images, training=None):
        if not training:
            return images
        # Pick an independent rotation (0, 90, 180 or 270 degrees) for every image in the batch.
        rotations = tf.stack([tf.image.rot90(images, k=k) for k in range(4)], axis=1)
        k = tf.random.uniform([tf.shape(images)[0]], minval=0, maxval=4, dtype=tf.int32)
        return tf.gather(rotations, k, axis=1, batch_dims=1)


def build_baseline(num_classes: int = len(CLASSES)) -> keras.Model:
    """A classic small CNN: three conv + pooling stages followed by a dense classifier."""
    inputs = keras.Input(INPUT_SHAPE)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def _conv_block(x, filters: int, n_convs: int = 2):
    for _ in range(n_convs):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    return layers.MaxPooling2D()(x)


def build_improved(num_classes: int = len(CLASSES)) -> keras.Model:
    """Deeper CNN with data augmentation, batch normalisation, global pooling and dropout."""
    inputs = keras.Input(INPUT_SHAPE)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = RandomRot90()(x)
    x = layers.RandomContrast(0.1)(x)
    # One conv at full 64x64 resolution (the most expensive stage), then two per block.
    x = _conv_block(x, 32, n_convs=1)
    x = _conv_block(x, 64)
    x = _conv_block(x, 128)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="improved_cnn")


MODELS = {"baseline": build_baseline, "improved": build_improved}


def build_model(name: str, learning_rate: float = 1e-3) -> keras.Model:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(MODELS)}.")
    model = MODELS[name]()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
