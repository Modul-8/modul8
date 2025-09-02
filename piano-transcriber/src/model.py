"""
model.py
--------
Defines the neural network architecture for piano transcription:
    - CNN + BiLSTM for local feature extraction.
    - Transformer encoder for long-range musical context.
    - Seperate output heads for onset, frame, offset and velocity predictions.
"""

import torch


class CRNNBackbone:
    """
    CNN + BiLSTM feature extractor for spectrogram inputs.

    Args:
        n_bins (int): Number of frequency bins in spectrogram.
        hidden_size (int): Size of LSTM hidden states.
    """

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Input features of shape (batch, 1 freq_bins,
            time_frames).

        Returns:
            Tensor: Output feature sequence for subsequent Transformer layers.
        """


class TransformerEncoder:
    """
    Transformer-based encoder for long-range temporal modeling.

    Args:
        hidden_size (int): Input/output dimensionality.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
    """

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Input sequence of shape (batch, time, hidden_size).

        Returns:
            Tensor: Context-enhanced sequence of shape (batch, time,
            hidden-size).
        """


class PianoTranscriptionModel:
    """
    Full model with CRNN backbone, Transformer encoder and output heads.

    Args:
        n_bins (int): Number of frequency bins.
        n_pitches (int): Number of piano pitches (usually 88).
        hidden_size (int): Size of hidden layers.
    """

    def forward(self, x: torch.Tensor):
        """
        Run forward pass of the model.

        Args:
            x (Tensor): Input spectrogram batch.

        Returns:
            Dict[str, Tensor]:
                Predictions for onset, frame, offset and velocity.
        """
