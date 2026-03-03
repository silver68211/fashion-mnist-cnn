"""
metrics_viz.py

Plotting utilities and evaluation helpers for classification experiments.
- image grid visualization
- training history plots
- predictions and confusion matrix plots
- k-fold performance summary
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


ArrayLike = Union[np.ndarray]


def _ensure_nhw(images: np.ndarray) -> np.ndarray:
    """
    Ensure images are shaped (N, H, W) for visualization.
    Accepts (N, H, W) or (N, H, W, C). If C exists, uses channel 0.
    """
    if images.ndim == 3:
        return images
    if images.ndim == 4:
        return images[..., 0]
    raise ValueError(f"Expected images with shape (N,H,W) or (N,H,W,C). Got {images.shape}")


def show_image_grid(
    images: np.ndarray,
    *,
    grid_size: int = 4,
    seed: Optional[int] = None,
    cmap: str = "binary",
    figsize: Tuple[int, int] = (5, 5),
    show: bool = True,
):
    """
    Display a grid of randomly sampled images.

    Parameters
    ----------
    images:
        Image array with shape (N,H,W) or (N,H,W,C).
    grid_size:
        Grid is grid_size x grid_size.
    seed:
        Optional random seed for reproducible sampling.
    cmap:
        Matplotlib colormap.
    figsize:
        Figure size.
    show:
        If True, calls plt.show().

    Returns
    -------
    (fig, axs)
        The Matplotlib figure and axes array.
    """
    x = _ensure_nhw(np.asarray(images))
    n = grid_size

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.shape[0], size=n * n)

    fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=figsize)

    for k, i in enumerate(idx):
        ax = axs[k // n, k % n]
        ax.imshow(x[i], cmap=cmap)
        ax.axis("off")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axs


def plot_history(
    history,
    metric: str,
    *,
    title: Optional[str] = None,
    show_validation: bool = True,
    show: bool = True,
):
    """
    Plot a single Keras History object.

    Parameters
    ----------
    history:
        Keras History (the object returned by model.fit()).
    metric:
        Metric key, e.g., "loss", "accuracy", "MSE".
    title:
        Optional plot title. Defaults to `metric`.
    show_validation:
        If True, plots 'val_<metric>' if available.
    show:
        If True, calls plt.show().
    """
    hist = getattr(history, "history", None)
    if not isinstance(hist, dict):
        raise ValueError("Expected a Keras History object (with `.history` dict).")

    if metric not in hist:
        raise KeyError(f"Metric '{metric}' not found in history. Available keys: {list(hist.keys())}")

    plt.figure()
    plt.plot(hist[metric], label=metric)

    val_key = f"val_{metric}"
    if show_validation and val_key in hist:
        plt.plot(hist[val_key], label=val_key)

    plt.title(title or metric)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()


def plot_histories(
    histories: Sequence,
    metric: str,
    *,
    title: Optional[str] = None,
    show_validation: bool = True,
    show: bool = True,
):
    """
    Plot multiple Keras History objects (e.g., k-fold histories) on one figure.
    """
    plt.figure()
    for h in histories:
        hist = getattr(h, "history", None)
        if not isinstance(hist, dict) or metric not in hist:
            continue
        plt.plot(hist[metric])

        val_key = f"val_{metric}"
        if show_validation and val_key in hist:
            plt.plot(hist[val_key])

    plt.title(title or metric)
    plt.xlabel("Epoch")
    plt.tight_layout()

    if show:
        plt.show()


def predict_labels(model, x: np.ndarray, y_onehot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict class labels and return (preds, true_labels) as integer arrays.
    """
    probs = model.predict(x, verbose=0)
    preds = np.argmax(probs, axis=1)
    labels = np.argmax(y_onehot, axis=1)
    return preds, labels


def confusion_matrix_from_model(
    model,
    x: np.ndarray,
    y_onehot: np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix from a model and one-hot labels."""
    preds, labels = predict_labels(model, x, y_onehot)
    return confusion_matrix(labels, preds)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    *,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 8),
    cmap=plt.cm.Blues,
    show: bool = True,
):
    """
    Plot confusion matrix.

    Parameters
    ----------
    cm:
        Confusion matrix (raw counts).
    class_names:
        List of class names.
    normalize:
        If True, normalize rows to sum to 1 (per-true-class).
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected square confusion matrix. Got shape={cm.shape}")

    display = cm.astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display = np.round(display / row_sums, 2)

    fig = plt.figure(figsize=figsize)
    plt.imshow(display, interpolation="nearest", cmap=cmap)
    plt.title(title + (" (Normalized)" if normalize else ""))
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = display.max() / 2.0 if display.size else 0.0
    for i, j in itertools.product(range(display.shape[0]), range(display.shape[1])):
        color = "white" if display[i, j] > threshold else "black"
        plt.text(j, i, display[i, j], ha="center", va="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def summarize_kfold_scores(scores: Sequence[float], *, title: str = "Accuracy", show: bool = True):
    """
    Print and visualize k-fold scores (e.g., accuracies).

    Parameters
    ----------
    scores:
        Iterable of fold scores in [0,1] or percentages (if you pass percent, set title accordingly).
    """
    scores = np.asarray(list(scores), dtype=float)
    if scores.size == 0:
        raise ValueError("scores is empty.")

    mean = float(np.mean(scores))
    std = float(np.std(scores))

    print(f"{title}: mean={mean:.4f}, std={std:.4f}, n={scores.size}")

    plt.figure()
    plt.boxplot(scores, notch=False)
    plt.title(f"{title} (k-fold)")
    plt.ylabel(title)
    plt.tight_layout()

    if show:
        plt.show()