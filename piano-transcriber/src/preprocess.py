"""
preprocess.py
-------------
Precomputes CQT features and frame-wise note labels from audio and MIDI files.
Saves results as compressed arrays for efficient loading during training.
"""

import argparse
import numpy as np
import librosa
import pretty_midi
import os
import glob
from tqdm import tqdm

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


def midi_to_frame_labels(midi_path: str, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert MIDI to onset/frame/offset/velocity labels.

    Args:
        midi_path (str): Path to MIDI file.
        n_frames (int): Number of frames in corresponding spectrogram.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            onset, frame, offset, velocity arrays (time_frames x pitches)
    """

    pm = pretty_midi.PrettyMIDI(midi_path)

    onset = np.zeros((n_frames, config.N_PITCHES), dtype=np.float32)
    frame = np.zeros((n_frames, config.N_PITCHES), dtype=np.float32)
    offset = np.zeros((n_frames, config.N_PITCHES), dtype=np.float32)
    velocity = np.zeros((n_frames, config.N_PITCHES), dtype=np.float32)

    for note in pm.instruments[0].notes:
        if note.pitch < 21 or note.pitch > 108:
            continue
        pitch_idx = note.pitch - 21

        start_frame = int(note.start * config.SAMPLE_RATE / config.HOP_LENGTH)
        end_frame = int(note.end * config.SAMPLE_RATE / config.HOP_LENGTH)

        if start_frame >= n_frames:
            continue
        end_frame = min(end_frame, n_frames - 1)

        onset[start_frame, pitch_idx] = 1.0
        offset[end_frame, pitch_idx] = 1.0
        frame[start_frame:end_frame + 1, pitch_idx] = 1.0
        velocity[start_frame:end_frame + 1, pitch_idx] = note.velocity / 127.0

    return onset, frame, offset, velocity


def process_file(audio_path: str, midi_path: str, out_id: str):
    """
    Process one audio-MIDI pair into features and labels, then save to disk.

    Args:
        audio_path (str): Path to audio file.
        midi_path (str): Path to MIDI file.
        output_id (str): Unique identifier for saving output files.
    """

    # Extract features
    C = audio_to_cqt(audio_path)
    n_frames = C.shape[1]

    # Extract labels
    onset, frame, offset, velocity = midi_to_frame_labels(midi_path, n_frames)

    # Save features
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(config.FEATURES_DIR, f"{out_id}.npz"), cqt=C)

    # Save labels
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    np.savez_compressed(
            os.path.join(config.LABELS_DIR, f"{out_id}.npz"),
            onset=onset,
            frame=frame,
            offset=offset,
            velocity=velocity
    )


def main(dataset_dir: str):
    """
    Process entire dataset directory of audio/MIDI pairs.

    Args:
        dataset_dir (str): Path to dataset root containing audio/ and midi/
        subfolders.
    """

    audio_files = sorted(glob.glob(os.path.join(dataset_dir, "audio", "*.wav")) +
                         glob.glob(os.path.join(dataset_dir, "audio", "*.flac")))

    for audio_path in tqdm(audio_files, desc="Processing files"):
        file_id = os.path.splitext(os.path.basename(audio_path))[0]
        midi_path = os.path.join(dataset_dir, "midi", f"{file_id}.midi")
        if not os.path.exists(midi_path):
            midi_path = os.path.join(dataset_dir, "midi", f"{file_id}.mid")
        if not os.path.exists(midi_path):
            print(f"ERROR: Skipping {file_id} (no matching MIDI found)")
            continue

        process_file(audio_path, midi_path, file_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=config.RAW_DATASET_DIR, help="Path to dataset root containing audio/ and midi/")
    args = parser.parse_args()
    main(args.dataset_dir)
