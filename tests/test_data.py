import numpy as np

from eurosat_cnn.config import CLASSES
from eurosat_cnn.data import list_images, make_dataset, stratified_split


def test_list_images_finds_every_class(fake_dataset):
    paths, labels = list_images(fake_dataset)
    assert len(paths) == 20 * len(CLASSES)
    assert set(labels) == set(range(len(CLASSES)))


def test_split_is_stratified_and_disjoint(fake_dataset):
    paths, labels = list_images(fake_dataset)
    train, val, test = stratified_split(paths, labels, val_fraction=0.2, test_fraction=0.2)
    assert len(train) + len(val) + len(test) == len(paths)
    assert not set(train.paths) & set(test.paths)
    assert not set(val.paths) & set(test.paths)
    # every class keeps the same share in every split
    for split in (train, val, test):
        counts = np.bincount(split.labels, minlength=len(CLASSES))
        assert counts.min() == counts.max()


def test_split_is_reproducible(fake_dataset):
    paths, labels = list_images(fake_dataset)
    a = stratified_split(paths, labels)[2]
    b = stratified_split(paths, labels)[2]
    assert list(a.paths) == list(b.paths)


def test_dataset_batches_are_scaled(fake_dataset):
    paths, labels = list_images(fake_dataset)
    train, _, _ = stratified_split(paths, labels)
    images, batch_labels = next(iter(make_dataset(train, batch_size=8)))
    assert images.shape == (8, 64, 64, 3)
    assert float(images.numpy().min()) >= 0.0 and float(images.numpy().max()) <= 1.0
    assert batch_labels.shape == (8,)
