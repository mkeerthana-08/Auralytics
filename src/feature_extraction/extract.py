"""
Feature Extraction module for Engine Whisperer.

For every preprocessed 1-second frame, three descriptors are computed
and concatenated into a single flat feature vector:

  ┌──────────────────────────────┬──────┬─────────────────────────────────────┐
  │ Descriptor                   │ Dims │ What it captures                    │
  ├──────────────────────────────┼──────┼─────────────────────────────────────┤
  │ MFCC (time-averaged mean)    │  13  │ Overall frequency "texture"          │
  │ ZCR  (mean)                  │   1  │ Rhythmicity / erraticness            │
  │ Spectral Centroid (mean, Hz) │   1  │ Energy centre-of-mass ↑ with wear   │
  └──────────────────────────────┴──────┴─────────────────────────────────────┘

  Total feature vector length = 15 values per frame.

Why these three?  Each descriptor captures a different axis of "faultiness":
  • MFCC:             texture / shape of the frequency spectrum
  • ZCR:              time-domain chaos / knocking
  • Spectral Centroid: where in the frequency spectrum most energy lives

Together they give the Random Forest enough discriminative information to
separate healthy, worn, and critical signatures — even with a small dataset.
"""

import os
import numpy as np
import librosa
import pandas as pd

from src.preprocessing.preprocess import preprocess_clip

# ─── Constants ───────────────────────────────────────────────────────────────
N_MFCC       = 13
FEATURE_SIZE = N_MFCC + 2   # 13 MFCC + 1 ZCR + 1 Centroid = 15

FEATURE_COLS = (
    [f"mfcc_{i}" for i in range(N_MFCC)]
    + ["zcr", "spectral_centroid"]
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame descriptors
# ─────────────────────────────────────────────────────────────────────────────
def compute_mfcc(frame: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Compute the time-averaged MFCC vector for one audio frame.

    The MFCC (Mel-Frequency Cepstral Coefficients) represent the sound's
    frequency texture by:
      1. Computing the FFT spectrum
      2. Warping to the Mel scale (matches human hearing sensitivity)
      3. Applying a DCT to compress to n_mfcc numbers

    Returns
    -------
    np.ndarray, shape (n_mfcc,)  — mean across the time axis
    """
    mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1).astype(np.float32)


def compute_zcr(frame: np.ndarray) -> float:
    """
    Compute the mean Zero Crossing Rate for one audio frame.

    ZCR counts how many times the waveform crosses the zero-amplitude line
    per sample.  A rhythmic, healthy engine has a low, regular ZCR.
    A knocking or erratic engine has a high, chaotic ZCR.

    Returns
    -------
    float  — mean ZCR across the frame
    """
    zcr = librosa.feature.zero_crossing_rate(frame)
    return float(np.mean(zcr))


def compute_spectral_centroid(frame: np.ndarray, sr: int) -> float:
    """
    Compute the mean Spectral Centroid (Hz) for one audio frame.

    The spectral centroid is the "centre of mass" of the frequency spectrum —
    the frequency at which most of the sound's energy is concentrated.
    As bearing wear progresses, vibration energy shifts toward higher
    frequencies, so this value increases with fault severity.

    Returns
    -------
    float  — mean spectral centroid in Hz
    """
    centroid = librosa.feature.spectral_centroid(y=frame, sr=sr)
    return float(np.mean(centroid))


# ─────────────────────────────────────────────────────────────────────────────
# Combined feature vector
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(frame: np.ndarray, sr: int) -> np.ndarray:
    """
    Combine all descriptors into a single 1-D feature vector.

    Parameters
    ----------
    frame : np.ndarray  1-second audio frame (float32)
    sr    : int         Sample rate

    Returns
    -------
    np.ndarray, shape (FEATURE_SIZE,) = (15,)
    """
    mfcc     = compute_mfcc(frame, sr)
    zcr      = compute_zcr(frame)
    centroid = compute_spectral_centroid(frame, sr)
    return np.concatenate([mfcc, [zcr, centroid]]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_matrix(data_dir: str, sr: int = 22_050) -> pd.DataFrame:
    """
    Walk  data_dir/{healthy,worn,critical}/  and extract features from
    every audio clip.

    Each 1-second frame of each clip becomes one row in the returned DataFrame.
    The 'label' column contains the folder name: 'healthy', 'worn', or 'critical'.

    Parameters
    ----------
    data_dir : str  Path to the  data/raw/  directory.
    sr       : int  Target sample rate.

    Returns
    -------
    pd.DataFrame with columns  FEATURE_COLS + ['label'].
    """
    classes = ["healthy", "worn", "critical"]
    rows: list = []

    for label in classes:
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found, skipping: {folder}")
            continue

        clip_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
        ]
        if not clip_files:
            print(f"[WARN] No audio files in {folder}")
            continue

        print(f"[INFO] Processing {len(clip_files)} '{label}' clips …")
        for fname in clip_files:
            fpath = os.path.join(folder, fname)
            try:
                frames, sr_ = preprocess_clip(fpath, sr=sr)
                for frame in frames:
                    vec = extract_features(frame, sr_)
                    rows.append(list(vec) + [label])
            except Exception as exc:
                print(f"[ERROR] {fpath}: {exc}")

    columns = FEATURE_COLS + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    print(f"[INFO] Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols")
    return df
