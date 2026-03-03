"""
models.py

Keras model architectures for Fashion-MNIST (and similar 2D image datasets).

Design goals:
- Keep model builders pure (no compilation, no training side effects)
- Make input_shape / num_classes configurable
- Minimize repetition via small helper blocks
"""

from __future__ import annotations

from typing import Tuple, Optional

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten,
    Conv2D, MaxPool2D, BatchNormalization, Activation
)


def _conv_block(
    x,
    filters: int,
    kernel_size: int = 3,
    *,
    padding: str = "same",
    activation: Optional[str] = "relu",
    use_bn: bool = False,
    pool: bool = True,
    pool_size: int = 2,
    pool_strides: Tuple[int, int] = (1, 1),
    name_prefix: str = "block",
    kernel_initializer: str = "he_uniform",
):
    """A small reusable Conv -> (BN) -> (Act) -> (Pool) block."""
    x = Conv2D(
        filters,
        kernel_size,
        strides=(1, 1),
        padding=padding,
        kernel_initializer=kernel_initializer,
        name=f"{name_prefix}_conv",
    )(x)

    if use_bn:
        x = BatchNormalization(name=f"{name_prefix}_bn")(x)

    if activation is not None:
        x = Activation(activation, name=f"{name_prefix}_act")(x)

    if pool:
        x = MaxPool2D(
            pool_size=pool_size,
            strides=pool_strides,
            padding="valid" if padding == "valid" else "same",
            name=f"{name_prefix}_pool",
        )(x)

    return x


def _dense_block(
    x,
    units: int,
    *,
    activation: str = "relu",
    dropout_rate: float = 0.0,
    use_bn: bool = False,
    name_prefix: str = "dense",
    kernel_initializer: str = "he_uniform",
):
    """Dense -> (BN) -> Act -> (Dropout) block."""
    x = Dense(units, kernel_initializer=kernel_initializer, name=f"{name_prefix}_fc")(x)

    if use_bn:
        x = BatchNormalization(name=f"{name_prefix}_bn")(x)

    x = Activation(activation, name=f"{name_prefix}_act")(x)

    if dropout_rate and dropout_rate > 0:
        x = Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)

    return x


def bcnn(
    *,
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    filters: int = 32,
    units: int = 100,
) -> Model:
    """Baseline CNN: Conv -> Pool -> Flatten -> Dense -> Softmax."""
    inp = Input(shape=input_shape, name="input")

    x = _conv_block(
        inp, filters=filters, kernel_size=3,
        padding="valid", activation="relu",
        use_bn=False, pool=True, name_prefix="bcnn1"
    )

    x = Flatten(name="flatten")(x)
    x = Dense(units, activation="relu", kernel_initializer="he_uniform", name="fc1")(x)
    out = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=out, name="BCNN")


def modified_padding_bcnn(
    *,
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    filters: int = 32,
    units: int = 100,
) -> Model:
    """Baseline CNN but with 'same' padding in the first conv layer."""
    inp = Input(shape=input_shape, name="input")

    x = _conv_block(
        inp, filters=filters, kernel_size=3,
        padding="same", activation="relu",
        use_bn=False, pool=True, name_prefix="mpbcnn1"
    )

    x = Flatten(name="flatten")(x)
    x = Dense(units, activation="relu", kernel_initializer="he_uniform", name="fc1")(x)
    out = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=out, name="ModifiedPaddingBCNN")


def deep_modified_padding_bcnn(
    *,
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    filters: int = 64,
    units: int = 100,
) -> Model:
    """Deeper CNN with three convolutional blocks (mostly 'same' padding)."""
    inp = Input(shape=input_shape, name="input")

    x = _conv_block(inp, filters=filters, padding="same", name_prefix="dmpbcnn1")
    x = _conv_block(x, filters=filters, padding="same", name_prefix="dmpbcnn2")
    x = _conv_block(x, filters=filters, padding="same", name_prefix="dmpbcnn3")

    x = Flatten(name="flatten")(x)
    x = Dense(units, activation="relu", kernel_initializer="he_uniform", name="fc1")(x)
    x = Dense(units, activation="relu", kernel_initializer="he_uniform", name="fc2")(x)
    out = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=out, name="DeepModifiedPaddingBCNN")


def regularized_bcnn(
    *,
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    filters1: int = 32,
    filters2: int = 64,
    units: int = 100,
) -> Model:
    """CNN with BatchNorm in conv and dense blocks (no dropout)."""
    inp = Input(shape=input_shape, name="input")

    x = _conv_block(inp, filters=filters1, padding="same", use_bn=True, name_prefix="reg1")
    x = _conv_block(x, filters=filters2, padding="same", use_bn=True, name_prefix="reg2")
    x = _conv_block(x, filters=filters2, padding="valid", use_bn=True, name_prefix="reg3")

    x = Flatten(name="flatten")(x)
    x = _dense_block(x, units=units, use_bn=True, dropout_rate=0.0, name_prefix="reg_fc1")
    x = _dense_block(x, units=units, use_bn=True, dropout_rate=0.0, name_prefix="reg_fc2")
    out = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=out, name="RegularizedBCNN")


def drop_bcnn(
    *,
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    filters1: int = 32,
    filters2: int = 64,
    units: int = 100,
    dropout_rate: float = 0.1,
) -> Model:
    """CNN with dropout after conv blocks and between dense layers."""
    inp = Input(shape=input_shape, name="input")

    x = _conv_block(inp, filters=filters1, padding="same", name_prefix="drop1")
    x = Dropout(dropout_rate, name="drop_conv1")(x)

    x = _conv_block(x, filters=filters2, padding="same", name_prefix="drop2")
    x = Dropout(dropout_rate, name="drop_conv2")(x)

    x = _conv_block(x, filters=filters2, padding="same", name_prefix="drop3")
    x = Dropout(dropout_rate, name="drop_conv3")(x)

    x = Flatten(name="flatten")(x)
    x = _dense_block(x, units=units, dropout_rate=dropout_rate, name_prefix="drop_fc1")
    x = Dense(units, activation="relu", kernel_initializer="he_uniform", name="fc2")(x)
    out = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=out, name="DropBCNN")