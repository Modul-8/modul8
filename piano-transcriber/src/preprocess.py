"""
preprocess.py
-------------
Precomputes CQT features and frame-wise note labels from audio and MIDI files.
Saves results as compressed arrays for efficient loading during training.
"""

import argparse
import numpy as np
import librosa

import config


def audio_to_cqt(path: str) -> np.ndarray:
    """
    Load audio and compute log-magnitude CQT.

    Args:
        audio_path (str): Path to audio file (WAV or FLAC).

    Returns:
        np.ndarray: 2D array (freq_bins x time_frames) of log-CQT values.
    """

    y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    C = np.abs(librosa.cqt(
        y,
        sr=config.SAMPLE_RATE,
        hop_length=config.HOP_LENGTH,
        bins_per_octave=config.BINS_PER_OCTAVE,
        n_bins=config.N_BINS,
        fmin=config.FMIN
        ))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    return C_db.astype(np.float32)


def midi_to_frame_labels(midi_path: str, n_frames: int) -> np.ndarray:
    """
    Convert MIDI to onset/frame/offset/velocity labels.

    Args:
        midi_path (str): Path to MIDI file.
        n_frames (int): Number of frames in corresponding spectrogram.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            onset, frame, offset, velocity arrays (time_frames x pitches)
    """

    return None


def process_file(audio_path: str, midi_path: str, out_id: str):
    """
    Process one audio-MIDI pair into features and labels, then save to disk.

    Args:
        audio_path (str): Path to audio file.
        midi_path (str): Path to MIDI file.
        output_id (str): Unique identifier for saving output files.
    """

    return None


def main(dataset_dir: str):
    """
    Process entire dataset directory of audio/MIDI pairs.

    Args:
        dataset_dir (str): Path to dataset root containing audio/ and midi/ subfolders.
    """

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=config.RAW_DATASET_DIR, help="Path to dataset root containing audio/ and midi/")
    args = parser.parse_args()
    main(args.dataset_dir)
