"""
config.py

Central configuration for the Fashion-MNIST training pipeline.
Designed for local execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# ============================================================
# Dataset Configuration
# ============================================================

@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-related configuration."""
    num_classes: int = 10
    input_shape: Tuple[int, int, int] = (28, 28, 1)

    class_names: Tuple[str, ...] = (
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    )


# ============================================================
# Training Configuration
# ============================================================

@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""
    seed: int = 1
    learning_rate: float = 0.01
    momentum: float = 0.9
    epochs: int = 10
    batch_size: int = 32


# ============================================================
# Paths
# ============================================================

def get_project_root() -> Path:
    """
    Returns the project root directory.
    Assumes this file lives inside the project folder.
    """
    return Path(__file__).resolve().parent


def get_savedir() -> Path:
    """
    Directory where trained models and checkpoints are stored.
    Default: ./saved_models
    Can be overridden with environment variable:
        export SAVEDIR=/custom/path
    """
    default_path = get_project_root() / "saved_models"
    path = Path(os.environ.get("SAVEDIR", default_path)).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# Public Configuration Objects
# ============================================================

DATASET = DatasetConfig()
TRAIN = TrainConfig()
SAVEDIR: Path = get_savedir()