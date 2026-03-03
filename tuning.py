"""
tuning.py

Hyperparameter tuning utilities using Keras Tuner (Bayesian Optimization).

- build_model(hp): defines the search space + builds a compiled model
- CustomTuner: injects batch_size as a tunable hyperparameter
- run_tuning(...): executes the tuning loop and returns the best model
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import SGD

# Prefer modern package name; keep backward compatibility.
try:
    import keras_tuner as kt
except Exception:  # pragma: no cover
    import kerastuner as kt  # type: ignore


@dataclass(frozen=True)
class TuningConfig:
    """Configuration for the hyperparameter search."""
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    num_classes: int = 10

    objective: str = "val_accuracy"
    max_trials: int = 10
    epochs: int = 10
    validation_split: float = 0.1

    project_dir: Path = Path("logs")
    project_name: str = "keras_tuner"

    overwrite: bool = True


def build_initial_model(
    hp: Optional[Any] = None,
    *,
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
) -> Model:
    """
    Build and compile a CNN model. If `hp` is provided, it defines the search space.

    Search space:
      - filters: {32, 64}
      - kernel size: {3, 5}
      - number of conv blocks: {2,3,4}
      - number of hidden layers: {1..5}
      - hidden units: {80,90,100,110,120}
      - lr: [0.001, 0.01]
      - momentum: [0.7, 1.0]
    """
    # Clean graph state between trials
    tf.keras.backend.clear_session()

    # Defaults
    filters = 32
    kernel_size = 3
    conv_blocks = 2
    hidden_layers = 2
    hidden_units = 75
    lr = 0.001
    momentum = 0.7

    if hp is not None:
        filters = hp.Choice("filters", values=[32, 64])
        kernel_size = hp.Choice("kernel_size", values=[3, 5])
        conv_blocks = hp.Choice("conv_blocks", values=[2, 3, 4])

        hidden_layers = hp.Choice("hidden_layers", values=list(range(1, 6)))
        hidden_units = hp.Choice("hidden_units", values=list(range(80, 130, 10)))

        lr = hp.Float("lr", min_value=0.001, max_value=0.01, sampling="linear")
        momentum = hp.Float("momentum", min_value=0.7, max_value=1.0, sampling="linear")

    inputs = Input(shape=input_shape, name="input")

    x = Conv2D(filters, kernel_size, padding="same", name="conv_1")(inputs)
    x = BatchNormalization(name="bn_1")(x)
    x = Activation("relu", name="act_1")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="valid", name="pool_1")(x)

    for i in range(2, conv_blocks + 1):
        x = Conv2D(filters, kernel_size, padding="same", name=f"conv_{i}")(x)
        x = BatchNormalization(name=f"bn_{i}")(x)
        x = Activation("relu", name=f"act_{i}")(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="valid", name=f"pool_{i}")(x)

    x = Flatten(name="flatten")(x)

    for j in range(1, hidden_layers + 1):
        x = Dense(hidden_units, name=f"fc_{j}")(x)
        x = BatchNormalization(name=f"fc_bn_{j}")(x)
        x = Activation("relu", name=f"fc_act_{j}")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="tuned_cnn")

    opt = SGD(learning_rate=lr, momentum=momentum)
    model.compile(
        loss="CategoricalCrossentropy",
        optimizer=opt,
        metrics=["accuracy", "MSE"],
    )
    return model


class CustomTuner(kt.tuners.BayesianOptimization):
    """
    BayesianOptimization tuner that also tunes batch_size per trial.
    """
    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Int("batch_size", min_value=32, max_value=128, step=32)
        return super().run_trial(trial, *args, **kwargs)


def run_tuning(
    x_train,
    y_train,
    *,
    cfg: TuningConfig = TuningConfig(),
):
    """
    Run Bayesian hyperparameter tuning and return the best model.

    Returns
    -------
    best_model: tf.keras.Model
    """
    cfg.project_dir.mkdir(parents=True, exist_ok=True)

    tuner = CustomTuner(
        hypermodel=lambda hp: build_initial_model(hp, input_shape=cfg.input_shape, num_classes=cfg.num_classes),
        objective=cfg.objective,
        max_trials=cfg.max_trials,
        directory=str(cfg.project_dir),
        project_name=cfg.project_name,
        overwrite=cfg.overwrite,
    )

    tuner.search_space_summary()
    tuner.search(
        x_train,
        y_train,
        validation_split=cfg.validation_split,
        epochs=cfg.epochs,
        verbose=0,
    )
    tuner.results_summary(num_trials=1)

    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model