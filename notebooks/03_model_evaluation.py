"""
Model evaluation script — Engine Whisperer
Train on synthetic data and show confusion matrix + accuracy.

Run from project root:
    python notebooks/03_model_evaluation.py
"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from src.training.train_runner import generate_synthetic_dataset
from src.training.train import train_model, evaluate_model

BG   = "#060b18"
CARD = "#0d1421"
GRID = "#1a2540"
TXT  = "#64748b"
CLASSES = ["healthy", "worn", "critical"]

print("[INFO] Generating synthetic dataset …")
X, y = generate_synthetic_dataset(n_per_class=200, seed=42)

print("[INFO] Training …")
model, X_test, y_test, y_pred = train_model(X, y)
metrics = evaluate_model(y_test, y_pred)

acc = metrics["accuracy"] * 100
cm  = metrics["confusion_matrix"]
rep = metrics["classification_report"]

print(f"\n[RESULT] Test Accuracy: {acc:.2f} %")
print(f"[RESULT] Confusion Matrix:\n{cm}")

# ── Visualise ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5), facecolor=BG)
spec = gs.GridSpec(1, 2, figure=fig, wspace=0.35)

# Confusion matrix
ax1 = fig.add_subplot(spec[0])
ax1.set_facecolor(CARD)
cmap_colors = plt.cm.YlOrRd(np.linspace(0.1, 0.95, 256))
import matplotlib.colors as mcolors
cmap = mcolors.ListedColormap(cmap_colors)
im = ax1.imshow(cm, interpolation="nearest", cmap=cmap)
plt.colorbar(im, ax=ax1).ax.tick_params(colors=TXT, labelsize=8)
ax1.set_xticks(range(len(CLASSES)))
ax1.set_yticks(range(len(CLASSES)))
ax1.set_xticklabels([c.upper() for c in CLASSES], color=TXT, fontsize=8, rotation=20)
ax1.set_yticklabels([c.upper() for c in CLASSES], color=TXT, fontsize=8)
ax1.set_xlabel("Predicted", color=TXT, fontsize=9)
ax1.set_ylabel("Actual", color=TXT, fontsize=9)
ax1.set_title(f"Confusion Matrix (Accuracy: {acc:.1f}%)", color="#e2e8f0", fontsize=10, fontweight="bold")
for sp in ax1.spines.values():
    sp.set_color(GRID)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="#fff" if cm[i, j] > cm.max() / 2 else "#333", fontsize=12, fontweight="bold")

# Per-class bar chart
ax2 = fig.add_subplot(spec[1])
ax2.set_facecolor(CARD)
class_colors = {"healthy": "#10b981", "worn": "#f59e0b", "critical": "#ef4444"}
metrics_keys = ["precision", "recall", "f1-score"]
x = np.arange(len(CLASSES))
width = 0.25
for mi, mk in enumerate(metrics_keys):
    vals = [rep.get(c, {}).get(mk, 0) for c in CLASSES]
    bars = ax2.bar(x + (mi - 1) * width, vals, width, label=mk.replace("-", " ").title(),
                   alpha=0.85, color=["#60a5fa", "#a78bfa", "#f472b6"][mi])
ax2.set_xticks(x)
ax2.set_xticklabels([c.upper() for c in CLASSES], color=TXT, fontsize=8)
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("Score", color=TXT, fontsize=9)
ax2.set_title("Per-Class Metrics", color="#e2e8f0", fontsize=10, fontweight="bold")
ax2.tick_params(colors=TXT, labelsize=8)
ax2.legend(fontsize=8, facecolor=CARD, labelcolor="#e2e8f0", edgecolor=GRID)
for sp in ax2.spines.values():
    sp.set_color(GRID)
ax2.grid(color=GRID, linewidth=0.4, axis="y")

fig.suptitle("Engine Whisperer — Model Evaluation Report", color="#e2e8f0",
             fontsize=13, fontweight="bold")
fig.tight_layout()
plt.savefig("notebooks/model_evaluation.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("\nSaved: notebooks/model_evaluation.png")
