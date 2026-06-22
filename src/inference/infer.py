"""
Inference module for Engine Whisperer.

Provides two public entry points:

  classify_clip(model, path)           — classify from a file on disk
  classify_live(model, audio, sr)      — classify from a NumPy audio array

Both apply the identical preprocessing + feature-extraction pipeline used
during training, then run a majority vote across all 1-second frames.
"""

import os
import numpy as np
import joblib
from collections import Counter

from src.preprocessing.preprocess import preprocess_clip, preprocess_array
from src.feature_extraction.extract import extract_features

CLASSES = ["healthy", "worn", "critical"]


# ─────────────────────────────────────────────────────────────────────────────
# Model I/O
# ─────────────────────────────────────────────────────────────────────────────
def load_model(path: str):
    """
    Load a trained model from disk.

    Raises
    ------
    FileNotFoundError  if the model file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run:  python -m src.training.train_runner --synthetic"
        )
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Single-frame prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, feature_vec: np.ndarray) -> tuple:
    """
    Classify one feature vector.

    Parameters
    ----------
    model       : fitted RandomForestClassifier
    feature_vec : 1-D float array, shape (FEATURE_SIZE,)

    Returns
    -------
    (label: str, confidence: float)
      label      — 'healthy' | 'worn' | 'critical'
      confidence — percentage in [0, 100], how strongly the model believes
                   in this prediction relative to alternatives
    """
    vec   = np.asarray(feature_vec, dtype=np.float32).reshape(1, -1)
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    conf  = float(np.max(proba)) * 100.0
    return label, conf


# ─────────────────────────────────────────────────────────────────────────────
# Internal: aggregate frame predictions
# ─────────────────────────────────────────────────────────────────────────────
def _classify_frames(model, frames: list, sr: int) -> dict:
    """
    Extract features from every frame, classify each, and return an aggregated
    result via majority vote.

    Returns
    -------
    dict with keys:
      label             — majority-vote class
      confidence        — average confidence across frames (%)
      features          — mean feature vector across frames
      frame_labels      — per-frame predicted labels
      frame_confidences — per-frame confidence percentages
    """
    if not frames:
        return {
            "label":             "unknown",
            "confidence":         0.0,
            "features":           None,
            "frame_labels":       [],
            "frame_confidences":  [],
        }

    feature_vecs   = [extract_features(f, sr) for f in frames]
    frame_labels   = []
    frame_confs    = []

    for vec in feature_vecs:
        lbl, conf = predict(model, vec)
        frame_labels.append(lbl)
        frame_confs.append(conf)

    majority_label = Counter(frame_labels).most_common(1)[0][0]
    avg_confidence = float(np.mean(frame_confs))
    mean_features  = np.mean(feature_vecs, axis=0)

    return {
        "label":             majority_label,
        "confidence":         avg_confidence,
        "features":           mean_features,
        "frame_labels":       frame_labels,
        "frame_confidences":  frame_confs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public pipelines
# ─────────────────────────────────────────────────────────────────────────────
def classify_clip(model, audio_path: str, sr: int = 22_050) -> dict:
    """
    Full pipeline: audio file path → classification result.

    Parameters
    ----------
    model      : fitted model
    audio_path : path to WAV / MP3 / FLAC
    sr         : target sample rate

    Returns
    -------
    dict (see ``_classify_frames``)
    """
    frames, sr_ = preprocess_clip(audio_path, sr=sr)
    return _classify_frames(model, frames, sr_)


def classify_live(model, audio_array: np.ndarray, sr: int = 22_050) -> dict:
    """
    Full pipeline: in-memory NumPy audio array → classification result.

    Parameters
    ----------
    model       : fitted model
    audio_array : 1-D float32 NumPy array (e.g., from live mic or synthetic)
    sr          : sample rate of the array

    Returns
    -------
    dict (see ``_classify_frames``)
    """
    frames, sr_ = preprocess_array(audio_array, sr=sr)
    return _classify_frames(model, frames, sr_)
