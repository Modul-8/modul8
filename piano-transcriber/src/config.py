"""
config.py
---------
Holds global configuration variables, such as:
    - Sample rate, hop length, bins per octave.
    - Dataset directories.
    - Training hyperparameters (batch size, learning rate, etc.).
"""

# --------------- |
# AUDIO PARAMETER |
# --------------- |

SAMPLE_RATE = 16000
HOP_LENGTH = 512
BINS_PER_OCTAVE = 36
N_BINS = 229
FMIN = 27.5

# ----------- |
# LABEL SPACE |
# ----------- |

N_PITCHES = 88

# ------------------- |
# DATASET DIRECTORIES |
# ------------------- |

RAW_DATASET_DIR = "data/maestro"
FEATURES_DIR = "data/processed/features"
LABELS_DIR = "data/processed/labels"

# -------------------------------------- |
# TRAINING PARAMETERS (example defaults) |
# -------------------------------------- |

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 50
