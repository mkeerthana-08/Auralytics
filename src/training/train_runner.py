"""
Training runner for Engine Whisperer.

Usage (run from project root):
    python -m src.training.train_runner --synthetic        # synthetic data
    python -m src.training.train_runner                    # real data in data/raw/

The runner:
  1. Generates (or loads) the feature matrix.
  2. Trains the Random Forest.
  3. Reports accuracy and confusion matrix.
  4. Saves the model to  models/model.pkl.
"""

import sys
import os

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import numpy as np

from src.feature_extraction.extract import extract_features, N_MFCC, build_feature_matrix
from src.preprocessing.preprocess import split_frames
from src.training.train import load_features, train_model, evaluate_model, save_model


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Audio Generator
# ─────────────────────────────────────────────────────────────────────────────
def _generate_clip(fault_type: str, sr: int, duration: float, seed: int) -> np.ndarray:
    """
    Generate one synthetic engine audio clip for ``fault_type``.

    Acoustic design rationale
    -------------------------
    Healthy   — fundamental 70 Hz + 2 quiet harmonics + tiny noise.
                Produces: low ZCR, stable centroid, smooth MFCC.

    Worn      — fundamental + bearing-defect harmonics at 230/460 Hz
                + moderate noise + occasional low impulses.
                Produces: moderate ZCR, elevated centroid, mildly irregular MFCC.

    Critical  — distorted signal with strong high-frequency components at
                500/1000/2000 Hz + heavy noise + frequent large impulses.
                Produces: high ZCR, sharply elevated centroid, chaotic MFCC.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, duration, int(duration * sr), dtype=np.float32)

    if fault_type == "healthy":
        y = (
            0.80 * np.sin(2 * np.pi *  70 * t).astype(np.float32)
          + 0.15 * np.sin(2 * np.pi * 140 * t).astype(np.float32)
          + 0.05 * np.sin(2 * np.pi * 210 * t).astype(np.float32)
        )
        y += rng.standard_normal(len(t)).astype(np.float32) * 0.010

    elif fault_type == "worn":
        y = (
            0.60 * np.sin(2 * np.pi *  70 * t).astype(np.float32)
          + 0.10 * np.sin(2 * np.pi * 140 * t).astype(np.float32)
          + 0.30 * np.sin(2 * np.pi * 230 * t).astype(np.float32)   # bearing defect
          + 0.15 * np.sin(2 * np.pi * 460 * t).astype(np.float32)   # 2× bearing freq
        )
        y += rng.standard_normal(len(t)).astype(np.float32) * 0.055
        # Occasional low-amplitude shock impulses
        idx = rng.integers(0, len(t), size=30)
        y[idx] += rng.standard_normal(30).astype(np.float32) * 0.22

    elif fault_type == "critical":
        y = (
            0.40 * np.sin(2 * np.pi *   70 * t).astype(np.float32)
          + 0.30 * np.sin(2 * np.pi *  500 * t).astype(np.float32)
          + 0.20 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
          + 0.15 * np.sin(2 * np.pi * 2000 * t).astype(np.float32)
        )
        y += rng.standard_normal(len(t)).astype(np.float32) * 0.120
        # Heavy shock impulses
        idx  = rng.integers(0, len(t), size=100)
        sign = rng.choice([-1.0, 1.0], size=100).astype(np.float32)
        y[idx] += sign * 0.70
    else:
        raise ValueError(f"Unknown fault type: {fault_type!r}")

    # Peak-normalise
    peak = float(np.max(np.abs(y)))
    return (y / peak).astype(np.float32) if peak > 1e-6 else y


def generate_synthetic_dataset(
    n_per_class: int   = 300,
    sr:          int   = 22_050,
    duration:    float = 3.0,
    seed:        int   = 42,
) -> tuple:
    """
    Generate synthetic feature vectors for all three classes in memory.

    Parameters
    ----------
    n_per_class : audio clips generated per class
    sr          : sample rate
    duration    : duration of each synthetic clip (seconds)
    seed        : base random seed

    Returns
    -------
    X : float32 array, shape (3 * n_per_class * n_frames_per_clip, FEATURE_SIZE)
    y : str array,     shape (same first dim,)
    """
    classes = ["healthy", "worn", "critical"]
    rows, labels = [], []

    for label in classes:
        print(f"[SYNTH] Generating {n_per_class} '{label}' clips …")
        for i in range(n_per_class):
            clip   = _generate_clip(label, sr, duration, seed=seed * 1_000 + i)
            frames = split_frames(clip, sr)
            for frame in frames:
                vec = extract_features(frame, sr)
                rows.append(vec)
                labels.append(label)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels)
    print(f"[SYNTH] Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Public convenience — called by the Streamlit app auto-trainer
# ─────────────────────────────────────────────────────────────────────────────
def train_on_synthetic(
    save_path:   str = None,
    n_per_class: int = 300,
    seed:        int = 42,
):
    """
    Generate synthetic data, train a Random Forest, optionally save it,
    and return the fitted model.

    Called by ``app/streamlit_app/app.py`` when  models/model.pkl  is absent.
    """
    print("[INFO] Training on synthetic data …")
    X, y = generate_synthetic_dataset(n_per_class=n_per_class, seed=seed)
    model, _, y_test, y_pred = train_model(X, y)
    metrics = evaluate_model(y_test, y_pred)
    acc = metrics["accuracy"] * 100
    print(f"[INFO] Synthetic test accuracy: {acc:.1f} %")

    if save_path:
        try:
            save_model(model, save_path)
        except OSError:
            # Read-only filesystem (e.g., Streamlit Cloud) — skip saving
            print(f"[WARN] Cannot save model to {save_path} — will re-train each session.")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train Engine Whisperer Random Forest classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Train on synthetic data (no audio files required)",
    )
    parser.add_argument(
        "--data-dir", default=os.path.join(ROOT, "data", "raw"),
        help="Path to data/raw/ directory (ignored with --synthetic)",
    )
    parser.add_argument(
        "--features-csv", default=os.path.join(ROOT, "data", "processed", "features.csv"),
        help="Where to save / load the feature matrix CSV",
    )
    parser.add_argument(
        "--model-out", default=os.path.join(ROOT, "models", "model.pkl"),
        help="Output path for the saved model",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=300,
        help="Synthetic clips per class (only with --synthetic)",
    )
    args = parser.parse_args()

    if args.synthetic:
        print("[INFO] Mode: synthetic data")
        train_on_synthetic(save_path=args.model_out, n_per_class=args.n_per_class)

    else:
        print(f"[INFO] Mode: real data from  {args.data_dir}")
        df = build_feature_matrix(args.data_dir)
        if df.empty:
            print("[ERROR] No features extracted. Check your data/raw/ folders.")
            sys.exit(1)

        os.makedirs(os.path.dirname(os.path.abspath(args.features_csv)), exist_ok=True)
        df.to_csv(args.features_csv, index=False)
        print(f"[INFO] Feature CSV saved to: {args.features_csv}")

        X, y = load_features(args.features_csv)
        model, _, y_test, y_pred = train_model(X, y)
        metrics = evaluate_model(y_test, y_pred)
        acc = metrics["accuracy"] * 100
        print(f"[INFO] Test accuracy: {acc:.1f} %")
        print(f"[INFO] Confusion matrix:\n{metrics['confusion_matrix']}")
        save_model(model, args.model_out)


if __name__ == "__main__":
    main()
