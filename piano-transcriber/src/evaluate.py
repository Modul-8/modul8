"""
evaluate.py
-----------
Provides evaluation metrics for piano transcription performance, including
frame-level F1, note-level F1 (onset and onset+offset) and velocity accuracy.
"""

import numpy as np
import torch


def compute_frame_f1(predictions: np.ndarray, targets: np.ndarray):
    """
    Compute frame-level F1 score.

    Args:
        predictions (np.ndarray): Binary frame predictions.
        targets (np.ndarray): Ground-truth frame labels.

    Returns:
        float: Frame-level F1 score.
    """


def compute_note_f1(predictions, targets, onset_only=False):
    """
    Compute note-level F1 score.

    Args:
        predictions (List[Tuple]): Predicted note events.
        targets (List[Tuple]): Ground-truth note events.
        onset_only (bool): Whether to evaluate only onsets or onsets+offsets.

    Returns:
        float: Note-level F1 score.
    """


def evaluate_model(model: torch.nn.Module, dataloader, device):
    """
    Evaluate model across a dataset.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Data loader for evaluation set.
        device (torch.device): Device for running evaluation.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
