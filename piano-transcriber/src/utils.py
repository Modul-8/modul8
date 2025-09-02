"""
utils.py
--------
Contains helper functions for configuration loading, logging, feature
normalization and other shared utilities across scripts.
"""

import numpy as np

def load_config(path: str):
    """
    Load configuration file.

    Args:
        path (str): Path to JSON config file.

    Returns:
        dict: Loaded configuration parameters.
    """


def setup_logging(log_dir: str):
    """
    Set up logging for training.

    Args:
        log_dir (str): Directory where logs will be stored.

    Returns:
        Logger: Configured logger object
    """


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    """
    Normalize spectogram features.

    Args:
        feature (np.ndarray): Input specrogram.

    Returns:
        np.ndarray: Normalized spectrogram.
    """
