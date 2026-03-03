"""
utils_io.py

File and path utility helpers for experiment management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """
    Ensure a directory exists. Creates it if necessary.

    Parameters
    ----------
    path : str | Path
        Directory path.

    Returns
    -------
    Path
        Resolved Path object.
    """
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_file(dir_path: PathLike, filename: str, *, silent: bool = True) -> bool:
    """
    Remove a file if it exists.

    Parameters
    ----------
    dir_path : str | Path
        Directory containing the file.
    filename : str
        File name to remove.
    silent : bool
        If False, raises FileNotFoundError when file does not exist.

    Returns
    -------
    bool
        True if file was removed, False if it did not exist.
    """
    file_path = Path(dir_path).expanduser().resolve() / filename

    if file_path.exists():
        file_path.unlink()
        return True

    if not silent:
        raise FileNotFoundError(f"{file_path} does not exist.")

    return False


def get_model_name(k: int, *, prefix: str = "model", ext: str = ".h5") -> str:
    """
    Generate a standardized checkpoint filename.

    Parameters
    ----------
    k : int
        Fold index (1-based).
    prefix : str
        Filename prefix.
    ext : str
        File extension.

    Returns
    -------
    str
        Formatted filename, e.g. "model_1.h5"
    """
    if k <= 0:
        raise ValueError("Fold index must be >= 1.")

    return f"{prefix}_{k}{ext}"