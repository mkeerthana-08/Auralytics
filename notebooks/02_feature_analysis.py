"""
Feature analysis script — Engine Whisperer
Plot MFCC heatmaps and ZCR/Centroid distributions per fault class.

Run from project root:
    python notebooks/02_feature_analysis.py
"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.training.train_runner import _generate_clip
from src.feature_extraction.extract import (
    compute_zcr, compute_spectral_centroid, N_MFCC
)

SR = 22_050
DURATION = 3.0
configs = [
    ("healthy",  "#10b981"),
    ("worn",     "#f59e0b"),
    ("critical", "#ef4444"),
]
BG = "#060b18"
CARD = "#0d1421"
GRID = "#1a2540"
TXT  = "#64748b"

# ── Figure 1: MFCC Heatmaps ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=BG)
fig.suptitle("MFCC Heatmaps by Fault Class", color="#e2e8f0", fontsize=13, fontweight="bold")

for ax, (label, color) in zip(axes, configs):
    y    = _generate_clip(label, SR, DURATION, seed=42)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    ax.set_facecolor(CARD)
    img = librosa.display.specshow(mfcc, sr=SR, x_axis="time", ax=ax, cmap="magma")
    plt.colorbar(img, ax=ax, format="%+.0f").ax.tick_params(colors=TXT, labelsize=7)
    ax.set_title(f"{label.upper()}", color=color, fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (s)", color=TXT, fontsize=8)
    ax.set_ylabel("MFCC Coeff", color=TXT, fontsize=8)
    ax.tick_params(colors=TXT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(GRID)

fig.tight_layout()
plt.savefig("notebooks/mfcc_heatmaps.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()

# ── Figure 2: ZCR + Centroid per class ───────────────────────────────────────
labels_list, zcr_list, centroid_list = [], [], []
for label, _ in configs:
    for i in range(20):
        y        = _generate_clip(label, SR, DURATION, seed=i)
        zcr_list.append(compute_zcr(y))
        centroid_list.append(compute_spectral_centroid(y, SR))
        labels_list.append(label)

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
fig2.suptitle("Feature Distributions by Fault Class", color="#e2e8f0", fontsize=13, fontweight="bold")

for ax, data, title, unit in [
    (ax1, zcr_list, "Zero Crossing Rate", "crossings/sample"),
    (ax2, centroid_list, "Spectral Centroid", "Hz"),
]:
    ax.set_facecolor(CARD)
    for j, (label, color) in enumerate(configs):
        vals = [d for d, l in zip(data, labels_list) if l == label]
        ax.scatter([j + 1] * len(vals), vals, color=color, alpha=0.6, s=40, label=label)
        ax.plot([j + 0.7, j + 1.3], [np.mean(vals)] * 2, color=color, linewidth=2)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["HEALTHY", "WORN", "CRITICAL"], color=TXT, fontsize=8)
    ax.set_title(title, color="#e2e8f0", fontsize=10, fontweight="bold")
    ax.set_ylabel(unit, color=TXT, fontsize=8)
    ax.tick_params(colors=TXT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.4, axis="y")

fig2.tight_layout()
plt.savefig("notebooks/feature_distributions.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()

print("Saved: notebooks/mfcc_heatmaps.png")
print("Saved: notebooks/feature_distributions.png")
