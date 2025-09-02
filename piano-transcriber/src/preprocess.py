"""
preprocess.py
-------------
Precomputes CQT features and frame-wise note labels from audio and MIDI files.
Saves results as compressed arrays for efficient loading during training.
"""

import argparse


def audio_to_cqt(path):
    """
    Load audio and compute log-magnitude CQT.
    """

    return None


def midi_to_frame_labels(midi_path, n_frames):
    """
    Convert MIDI to onset/frame/offset/velocity labels.
    """

    return None


def process_file(audio_path, midi_path, out_id):
    """
    Process one audio-MIDI pair.
    """

    return None


def main(dataset_dir):

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/maestro", help="Path to dataset root containing audio/ and midi/")
    args = parser.parse_args()
    main(args.dataset_dir)
