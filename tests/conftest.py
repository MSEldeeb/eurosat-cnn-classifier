import numpy as np
import pytest
from PIL import Image

from eurosat_cnn.config import CLASSES


@pytest.fixture
def fake_dataset(tmp_path):
    """A tiny EuroSAT-like folder: 20 random 64x64 JPEGs per class."""
    rng = np.random.default_rng(0)
    for name in CLASSES:
        (tmp_path / name).mkdir()
        for i in range(20):
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(tmp_path / name / f"{name}_{i}.jpg")
    return tmp_path
