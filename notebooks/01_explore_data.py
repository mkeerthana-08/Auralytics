"""
Exploration script — Engine Whisperer
Visualise waveforms from each fault class.

Run from project root:
    python notebooks/01_explore_data.py
"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from src.training.train_runner import _generate_clip

SR = 22_050
DURATION = 3.0

fig, axes = plt.subplots(3, 1, figsize=(12, 8), facecolor="#060b18")
fig.suptitle("Engine Whisperer — Waveform Comparison by Fault Class",
             color="#e2e8f0", fontsize=14, fontweight="bold")

configs = [
    ("healthy",  "#10b981", "Healthy Engine — Smooth, Regular Rhythm"),
    ("worn",     "#f59e0b", "Worn (Early Bearing Wear) — Moderate Distortion"),
    ("critical", "#ef4444", "Critical Fault — Chaotic, High-Energy Signal"),
]

for ax, (label, color, title) in zip(axes, configs):
    y = _generate_clip(label, SR, DURATION, seed=42)
    t = np.linspace(0, DURATION, len(y))
    ax.set_facecolor("#0d1421")
    ax.plot(t, y, color=color, linewidth=0.6, alpha=0.9)
    ax.fill_between(t, y, alpha=0.12, color=color)
    ax.set_title(title, color=color, fontsize=10, pad=4)
    ax.set_ylabel("Amplitude", color="#64748b", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#1a2540")
    ax.grid(color="#1a2540", linewidth=0.4)
    ax.set_xlim(0, DURATION)

axes[-1].set_xlabel("Time (s)", color="#64748b", fontsize=8)
fig.tight_layout()
plt.savefig("notebooks/waveform_comparison.png", dpi=150, bbox_inches="tight",
            facecolor="#060b18")
plt.show()
print("Saved: notebooks/waveform_comparison.png")
