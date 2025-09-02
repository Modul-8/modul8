"""
decode.py
---------
Converts model predictions (frame-wise probabilities) into discrete note
events. Outputs final transcription as a MIDI file for listening or further
processing.
"""


def decode_predictions(pred_dict, sr: int, hop_length):
    """
    Decode raw model predictions into note events.

    Args:
        pred_dict (Dict[str, np.ndarray]): Model predictions for onset, frame,
        offset, velocity.
        sr (int): Sample rate used in preprocessing.
        hop_length (int): Hop length used for spectrogram frames.

    Returns:
        List[Tuple[int, float, float, int]]:
            List of note events (pitch, onset_time, duration, velocity).
    """


def save_midi(note_events, output_path: str):
    """
    Save note events to a MIDI file.

    Args:
        note_events (List[Tuple[int, float, float, int]]): Decoded note events.
        output_path (str): File path to save MIDI file.
    """


def main(audio_file, checkpoint, output_dir):
    """
    Run inference on a single audio file and produce MIDI transcription.

    Args:
        audio_file (str): Path to input audio file.
        checkpoint (str): Path to trained model checkpoint.
        output_dir (str): Directory to save output MIDI.
    """
