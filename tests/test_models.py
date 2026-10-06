import numpy as np
import keras
import pytest

from eurosat_cnn.config import CLASSES
from eurosat_cnn.models import RandomRot90, build_model


@pytest.mark.parametrize("name", ["baseline", "improved"])
def test_model_output_is_a_probability_distribution(name):
    model = build_model(name)
    x = np.random.rand(4, 64, 64, 3).astype("float32")
    probs = model.predict(x, verbose=0)
    assert probs.shape == (4, len(CLASSES))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("name", ["baseline", "improved"])
def test_model_can_take_a_training_step(name):
    model = build_model(name)
    x = np.random.rand(8, 64, 64, 3).astype("float32")
    y = np.random.randint(0, len(CLASSES), 8)
    loss = model.train_on_batch(x, y)
    assert np.isfinite(loss).all()


def test_rot90_only_changes_images_in_training():
    layer = RandomRot90()
    x = np.random.rand(16, 8, 8, 3).astype("float32")
    np.testing.assert_array_equal(layer(x, training=False), x)
    out = layer(x, training=True).numpy()
    # each output image is one of the four 90-degree rotations of its input
    for i in range(len(x)):
        assert any(np.allclose(out[i], np.rot90(x[i], k)) for k in range(4))


def test_improved_model_round_trips_through_keras_file(tmp_path):
    model = build_model("improved")
    path = tmp_path / "m.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    x = np.random.rand(2, 64, 64, 3).astype("float32")
    np.testing.assert_allclose(model.predict(x, verbose=0), loaded.predict(x, verbose=0), rtol=1e-5)
