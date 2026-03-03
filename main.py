"""
main.py

Entry point for running Fashion-MNIST experiments:
- load data
- run k-fold validation for multiple architectures
- select best fold checkpoint
- evaluate on test set
- plot curves and summarize results
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

from config import DATASET, TRAIN, SAVEDIR
from data import import_dataset
from models import (
    bcnn,
    modified_padding_bcnn,
    deep_modified_padding_bcnn,
    regularized_bcnn,
    drop_bcnn,
)
from train_eval import (
    model_evaluation_kfold,
    best_model_from_scores,
    evaluate_saved_model,
)
from metrics_viz import show_image_grid, plot_histories, summarize_kfold_scores


ModelFn = Callable[..., Any]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single experiment run."""
    k_folds: int = 5
    verbose: int = 0


def _print_dataset_info(x_train: np.ndarray, y_train: np.ndarray,
                        x_test: np.ndarray, y_test: np.ndarray) -> None:
    print("Dataset shapes:")
    print(f"  x_train: {x_train.shape} | y_train: {y_train.shape}")
    print(f"  x_test : {x_test.shape} | y_test : {y_test.shape}")


def run_experiment(
    name: str,
    model_builder: Callable[[], Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    exp_cfg: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Run k-fold evaluation, select best fold checkpoint, evaluate on test set, and plot results.

    Returns a dict containing:
      - name
      - scores (k-fold accuracies)
      - best_fold, best_file
      - test_eval (loss, acc, mse)
      - histories, val_losses
    """
    scores, histories, val_losses = model_evaluation_kfold(
        x_train,
        y_train,
        model_fn=model_builder,
        savedir=str(SAVEDIR) + "/",   # train_eval expects string + filename; keep consistent
        k_folds=exp_cfg.k_folds,
        epochs=TRAIN.epochs,
        batch_size=TRAIN.batch_size,
        lr=TRAIN.learning_rate,
        mom=TRAIN.momentum,
        verbose=exp_cfg.verbose,
        seed=TRAIN.seed,
    )

    best_fold, best_file = best_model_from_scores(scores)
    test_eval = evaluate_saved_model(str(SAVEDIR) + "/", best_file, x_test, y_test, verbose=1)

    print(f"\n=== Experiment: {name} ===")
    print(f"  k-fold accuracies: {np.round(scores, 4)}")
    print(f"  best fold: {best_fold} | checkpoint: {best_file}")
    print(f"  test eval (loss, acc, mse): {test_eval}")

    summarize_kfold_scores(scores, title=f"{name} Accuracy")
    plot_histories(histories, "loss", title=f"{name} - Loss")
    plot_histories(histories, "accuracy", title=f"{name} - Accuracy")

    return {
        "name": name,
        "scores": scores,
        "best_fold": best_fold,
        "best_file": best_file,
        "test_eval": test_eval,
        "histories": histories,
        "val_losses": val_losses,
    }


def main() -> Dict[str, Dict[str, Any]]:
    """
    Main entry point. Loads data, shows a sample grid, and runs all experiments.

    Returns a results dictionary keyed by experiment name.
    """
    (x_train, y_train), (x_test, y_test) = import_dataset(
        fashion_mnist,
        input_shape=DATASET.input_shape,
        num_classes=DATASET.num_classes,
        normalize=True,
    )

    _print_dataset_info(x_train, y_train, x_test, y_test)
    show_image_grid(x_train, grid_size=4, seed=TRAIN.seed)

    # Registry of experiments (single place to add/remove models)
    experiments: Dict[str, Callable[[], Any]] = {
        "BCNN": lambda: bcnn(input_shape=DATASET.input_shape, num_classes=DATASET.num_classes),
        "ModifiedPaddingBCNN": lambda: modified_padding_bcnn(input_shape=DATASET.input_shape, num_classes=DATASET.num_classes),
        "DeepModifiedPaddingBCNN": lambda: deep_modified_padding_bcnn(input_shape=DATASET.input_shape, num_classes=DATASET.num_classes),
        "RegularizedBCNN": lambda: regularized_bcnn(input_shape=DATASET.input_shape, num_classes=DATASET.num_classes),
        "DropBCNN": lambda: drop_bcnn(input_shape=DATASET.input_shape, num_classes=DATASET.num_classes),
    }

    exp_cfg = ExperimentConfig(k_folds=5, verbose=0)

    results: Dict[str, Dict[str, Any]] = {}
    for name, builder in experiments.items():
        results[name] = run_experiment(
            name=name,
            model_builder=builder,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            exp_cfg=exp_cfg,
        )

    return results


if __name__ == "__main__":
    main()