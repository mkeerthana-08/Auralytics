"""
Preprocessing module for Engine Whisperer.

Pipeline stages applied identically to every audio clip
(both during training and live inference):

  Load ──► Denoise ──► Normalise ──► Frame

Keeping this pipeline strictly identical between training and inference is
critical: the classifier can only recognise a live sound correctly if that
sound was converted into numbers the exact same way the training sounds were.
"""

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

# ── Optional: noisereduce back-end ────────────────────────────────────────────
try:
    import noisereduce as nr
    _NR_AVAILABLE = True
except ImportError:
    _NR_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Load
# ─────────────────────────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = 22_050) -> tuple:
    """
    Load an audio file and return (signal, sample_rate).

    Parameters
    ----------
    path : str  Absolute or relative path to WAV / MP3 / FLAC / OGG.
    sr   : int  Target sample rate (default 22 050 Hz — librosa standard).

    Returns
    -------
    (y, sr)  y is a 1-D float32 NumPy array.
    """
    y, sample_rate = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sample_rate


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Denoise
# ─────────────────────────────────────────────────────────────────────────────
def reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Remove background hiss / hum from the signal.

    Uses ``noisereduce`` (spectral subtraction) when available, otherwise
    falls back to a 4th-order Butterworth high-pass filter at 20 Hz that
    strips DC offset and infra-low noise.

    Why this matters: without denoising, room acoustics and microphone hiss
    can become spurious classifier features — the model would learn about the
    room, not the engine.
    """
    if _NR_AVAILABLE:
        try:
            denoised = nr.reduce_noise(
                y=y, sr=sr, stationary=False, prop_decrease=0.75
            )
            return denoised.astype(np.float32)
        except Exception:
            pass  # fall through to the filter fallback

    # Fallback: high-pass filter to strip DC / infra-low noise
    sos = butter(4, 20.0 / (sr / 2.0), btype="high", output="sos")
    return sosfilt(sos, y).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Normalise
# ─────────────────────────────────────────────────────────────────────────────
def normalize_amplitude(y: np.ndarray) -> np.ndarray:
    """
    Peak-normalise the signal to the range [-1, 1].

    Why this matters: a loud recording and a quiet recording of the same fault
    type must produce the same feature values.  Without normalisation, volume
    becomes a confounding signal that the classifier learns instead of the
    actual fault pattern.
    """
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        return (y / peak).astype(np.float32)
    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Frame
# ─────────────────────────────────────────────────────────────────────────────
def split_frames(
    y: np.ndarray,
    sr: int,
    frame_sec: float = 1.0,
    overlap: float = 0.5,
) -> list:
    """
    Slice the signal into overlapping fixed-length windows.

    Parameters
    ----------
    y          : 1-D signal array
    sr         : sample rate
    frame_sec  : window length in seconds  (default 1.0 s)
    overlap    : fractional overlap  (default 0.5 = 50 %)

    Returns
    -------
    List of 1-D NumPy float32 arrays, each of length  frame_sec * sr  samples.

    Why 50 % overlap?  Faults are not always uniformly distributed across all
    5 seconds of a clip.  Overlapping windows give the model multiple looks at
    each part of the signal and double the effective training set size per clip.
    """
    frame_len = int(frame_sec * sr)
    hop_len   = int(frame_len * (1.0 - overlap))
    frames    = []

    start = 0
    while start + frame_len <= len(y):
        frames.append(y[start : start + frame_len].copy())
        start += hop_len

    # If the whole clip is shorter than one frame, zero-pad it
    if not frames and len(y) > 0:
        padded = np.zeros(frame_len, dtype=np.float32)
        padded[: len(y)] = y
        frames.append(padded)

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Master Pipelines
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_clip(path: str, sr: int = 22_050) -> tuple:
    """
    Full preprocessing pipeline for a file on disk.

    Returns
    -------
    (frames, sr)  frames is a list of 1-D float32 arrays.
    """
    y, sr_ = load_audio(path, sr=sr)
    y = reduce_noise(y, sr_)
    y = normalize_amplitude(y)
    frames = split_frames(y, sr_)
    return frames, sr_


def preprocess_array(y: np.ndarray, sr: int = 22_050) -> tuple:
    """
    Full preprocessing pipeline for an in-memory NumPy array
    (e.g., live microphone capture or synthetic test signal).

    Returns
    -------
    (frames, sr)  frames is a list of 1-D float32 arrays.
    """
    y = np.asarray(y, dtype=np.float32)
    y = reduce_noise(y, sr)
    y = normalize_amplitude(y)
    frames = split_frames(y, sr)
    return frames, sr
