from __future__ import annotations

from typing import Tuple, Protocol
import numpy as np
from tensorflow.keras.utils import to_categorical


class KerasDatasetModule(Protocol):
    """Minimal protocol for tf.keras.datasets.* modules."""
    @staticmethod
    def load_data():
        ...


def import_dataset(
    dataset_module: KerasDatasetModule,
    *,
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    normalize: bool = True,
    dtype: np.dtype = np.float32,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load and preprocess an image classification dataset from a tf.keras.datasets module.

    Parameters
    ----------
    dataset_module:
        A module like `tensorflow.keras.datasets.fashion_mnist` providing `load_data()`.
    input_shape:
        Expected (H, W, C). For Fashion-MNIST, default is (28, 28, 1).
    num_classes:
        Number of classes for one-hot encoding.
    normalize:
        If True, scales pixel values to [0, 1].
    dtype:
        Floating dtype to use for images.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x_* shape: (N, H, W, C)
        y_* shape: (N, num_classes) one-hot encoded
    """
    (x_train, y_train), (x_test, y_test) = dataset_module.load_data()

    if x_train.ndim not in (3, 4):
        raise ValueError(f"Expected x_train to have 3 or 4 dims, got shape={x_train.shape}")
    if x_test.ndim not in (3, 4):
        raise ValueError(f"Expected x_test to have 3 or 4 dims, got shape={x_test.shape}")

    h, w, c = input_shape

    # Ensure channel dimension exists
    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]
    if x_test.ndim == 3:
        x_test = x_test[..., np.newaxis]

    if x_train.shape[1:] != (h, w, c) or x_test.shape[1:] != (h, w, c):
        raise ValueError(
            f"Unexpected image shape. Expected {(h, w, c)} but got "
            f"train={x_train.shape[1:]}, test={x_test.shape[1:]}"
        )

    x_train = x_train.astype(dtype, copy=False)
    x_test = x_test.astype(dtype, copy=False)

    if normalize:
        x_train /= dtype.type(255.0)
        x_test /= dtype.type(255.0)

    # One-hot encoding
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return (x_train, y_train), (x_test, y_test)


def flatten_images(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten image tensors for models that expect vectors (e.g., MLP).

    Accepts either (N, H, W, C) or (N, H, W).
    Returns (N, H*W*C).
    """
    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]
    if x_test.ndim == 3:
        x_test = x_test[..., np.newaxis]

    if x_train.ndim != 4 or x_test.ndim != 4:
        raise ValueError(f"Expected 4D tensors. Got train={x_train.shape}, test={x_test.shape}")

    return (
        x_train.reshape(x_train.shape[0], -1),
        x_test.reshape(x_test.shape[0], -1),
    )