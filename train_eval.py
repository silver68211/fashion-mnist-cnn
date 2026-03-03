"""
train_eval.py

Training and evaluation utilities:
- model compilation
- single-run training
- k-fold cross validation with checkpointing
- best-fold selection
- saved-model evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

from utils_io import ensure_dir, get_model_name, remove_file


ModelBuilder = Callable[[], Any]


@dataclass(frozen=True)
class CompileConfig:
    """Model compilation configuration."""
    learning_rate: float = 0.01
    momentum: float = 0.9
    loss: str = "CategoricalCrossentropy"
    metrics: Tuple[str, ...] = ("accuracy", "MSE")


@dataclass(frozen=True)
class KFoldConfig:
    """K-fold training configuration."""
    k_folds: int = 5
    epochs: int = 10
    batch_size: int = 32
    seed: int = 1
    verbose: int = 0
    monitor: str = "val_accuracy"
    mode: str = "max"


def compile_model(model: Any, cfg: CompileConfig) -> Any:
    """
    Compile a Keras model with SGD and standard metrics.
    """
    opt = SGD(learning_rate=cfg.learning_rate, momentum=cfg.momentum)
    model.compile(loss=cfg.loss, optimizer=opt, metrics=list(cfg.metrics))
    return model


def train_single(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    verbose: int = 1,
    callbacks: Optional[Sequence[Callback]] = None,
):
    """
    Train a model for a single train/validation split.
    Returns the Keras History object.
    """
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=list(callbacks) if callbacks else None,
    )


def kfold_evaluate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_builder: ModelBuilder,
    savedir: Union[str, Path],
    kcfg: KFoldConfig = KFoldConfig(),
    ccfg: CompileConfig = CompileConfig(),
) -> Dict[str, Any]:
    """
    Run k-fold cross validation.

    For each fold:
    - build a fresh model
    - compile
    - fit with ModelCheckpoint (best val metric)
    - reload best weights
    - evaluate on validation set

    Returns
    -------
    dict with keys:
      - scores: np.ndarray of fold accuracies
      - val_losses: np.ndarray of fold validation losses
      - histories: list of History objects
      - fold_results: list of dicts containing metrics per fold
    """
    savedir = Path(savedir).expanduser().resolve()
    ensure_dir(str(savedir))

    kfold = KFold(n_splits=kcfg.k_folds, shuffle=True, random_state=kcfg.seed)

    histories: List[Any] = []
    fold_results: List[Dict[str, float]] = []
    scores: List[float] = []
    val_losses: List[float] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kfold.split(x), start=1):
        ckpt_name = get_model_name(fold_idx)
        ckpt_path = savedir / ckpt_name

        # Remove old checkpoint if exists
        remove_file(str(savedir) + "/", ckpt_name)

        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_va, y_va = x[va_idx], y[va_idx]

        model = model_builder()
        compile_model(model, ccfg)

        checkpoint_cb = ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor=kcfg.monitor,
            save_best_only=True,
            mode=kcfg.mode,
            verbose=0,
        )

        history = train_single(
            model,
            x_tr,
            y_tr,
            x_va,
            y_va,
            epochs=kcfg.epochs,
            batch_size=kcfg.batch_size,
            verbose=kcfg.verbose,
            callbacks=[checkpoint_cb],
        )

        # Evaluate best checkpoint on validation set
        model.load_weights(str(ckpt_path))
        eval_values = model.evaluate(x_va, y_va, verbose=1)
        metrics = dict(zip(model.metrics_names, eval_values))

        histories.append(history)
        fold_results.append({k: float(v) for k, v in metrics.items()})

        # Common keys (robust fallback if user changes metric names)
        acc_key = "accuracy" if "accuracy" in metrics else model.metrics_names[1]
        loss_key = "loss"

        scores.append(float(metrics[acc_key]))
        val_losses.append(float(metrics[loss_key]))

    return {
        "scores": np.asarray(scores, dtype=float),
        "val_losses": np.asarray(val_losses, dtype=float),
        "histories": histories,
        "fold_results": fold_results,
    }


def best_model_from_scores(scores: Sequence[float]) -> Tuple[int, str]:
    """
    Return (best_fold_index, best_checkpoint_filename).
    best_fold_index is 1-based (matches checkpoint naming).
    """
    scores = np.asarray(scores, dtype=float)
    best_idx = int(np.argmax(scores))
    best_fold = best_idx + 1
    return best_fold, get_model_name(best_fold)


def evaluate_saved_model(
    savedir: Union[str, Path],
    model_filename: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    verbose: int = 1,
):
    """
    Load a saved Keras model (H5 or SavedModel) and evaluate it on a test set.
    """
    savedir = Path(savedir).expanduser().resolve()
    model_path = savedir / model_filename
    model = load_model(str(model_path))
    return model.evaluate(x_test, y_test, verbose=verbose)