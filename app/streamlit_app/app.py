"""
Engine Whisperer — Auralytics Diagnostic App
============================================
Sound-based engine fault diagnosis system.

Modes:
  🎯 Demo Mode   — Analyse a pre-generated synthetic engine clip
  📁 Upload Mode — Analyse a user-supplied WAV / MP3 / FLAC file

Auto-trains on synthetic data when models/model.pkl is absent.
Generates a downloadable PDF diagnostic report with embedded visualisations.
"""

# ── Python path setup (must be first, before any project imports) ─────────────
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Standard library ─────────────────────────────────────────────────────────
import io
import time
import tempfile
from datetime import datetime
from collections import Counter

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib

# ── Optional: sounddevice for local mic recording ─────────────────────────────
try:
    import sounddevice as sd
    _SD_OK = True
except Exception:
    _SD_OK = False

# ── Project modules ───────────────────────────────────────────────────────────
from src.preprocessing.preprocess import preprocess_array
from src.feature_extraction.extract import N_MFCC
from src.reporting.report import generate_report
from src.inference.infer import classify_live

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(ROOT, "models", "model.pkl")
SR           = 22_050
REC_DURATION = 5  # seconds for live recording

# Status colour palettes
_COLORS = {
    "healthy":  {"accent": "#10b981", "bg": "#022c22", "light": "#d1fae5", "badge": "#065f46", "glow": "#10b98133"},
    "worn":     {"accent": "#f59e0b", "bg": "#2d1500", "light": "#fef3c7", "badge": "#92400e", "glow": "#f59e0b33"},
    "critical": {"accent": "#ef4444", "bg": "#2d0000", "light": "#fee2e2", "badge": "#991b1b", "glow": "#ef444433"},
    "unknown":  {"accent": "#6b7280", "bg": "#111827", "light": "#f3f4f6", "badge": "#374151", "glow": "#6b728033"},
}
_ICONS = {"healthy": "✅", "worn": "⚠️", "critical": "🚨", "unknown": "❓"}
_STATUS_RGB = {
    "healthy":  (16, 185, 129),
    "worn":     (245, 158, 11),
    "critical": (239, 68, 68),
    "unknown":  (107, 114, 128),
}

_DEMO_OPTIONS = {
    "🟢 Healthy Engine — Normal Operation": "healthy",
    "🟡 Early Bearing Wear — Worn State":   "worn",
    "🔴 Critical Fault — Seized Bearing":   "critical",
    "🎲 Random (Surprise Me!)":             None,
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Engine Whisperer | Auralytics",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Engine Whisperer — Auralytics v1.0**\n\n"
            "Sound-based engine fault diagnosis using MFCC, ZCR, and "
            "Spectral Centroid features classified by a Random Forest.\n\n"
            "Inspired by Caterpillar TIS/ADSD diagnostic workflows."
        )
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Base ─────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #060b18 !important;
    font-family: 'Inter', system-ui, sans-serif;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #080e1c !important;
    border-right: 1px solid #1a2540;
}
[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
[data-testid="stDecoration"] { display: none; }
.main .block-container { padding-top: 1rem; max-width: 1200px; }

/* ── Header ───────────────────────────────────────────────── */
.ew-header { text-align: center; padding: 1.5rem 0 0.5rem; }
.ew-title {
    font-size: 2.8rem; font-weight: 800; letter-spacing: -2px;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; line-height: 1.1;
}
.ew-sub {
    font-size: 0.9rem; color: #475569; margin-top: 0.4rem;
    letter-spacing: 0.04em; text-transform: uppercase;
}

/* ── Status card ──────────────────────────────────────────── */
.status-card {
    border-radius: 20px; padding: 2.2rem 1.5rem;
    text-align: center; margin: 1.5rem 0;
    position: relative; overflow: hidden;
}
.status-icon { font-size: 3.8rem; display: block; margin-bottom: 0.6rem; }
.status-lbl {
    font-size: 2.2rem; font-weight: 800;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.6rem;
}
.status-desc { font-size: 0.88rem; line-height: 1.7; max-width: 480px; margin: 0 auto 1rem; }
.status-badge {
    display: inline-block; padding: 0.3rem 1.2rem;
    border-radius: 99px; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
}

/* ── Confidence bar ───────────────────────────────────────── */
.conf-wrap { margin: 1.2rem 0; }
.conf-header { display: flex; justify-content: space-between; margin-bottom: 6px; }
.conf-label { color: #94a3b8; font-size: 0.82rem; }
.conf-value { font-weight: 700; font-size: 0.9rem; }
.conf-bg { background: #1a2540; border-radius: 99px; height: 10px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 99px; transition: width 0.9s cubic-bezier(.4,0,.2,1); }

/* ── Metric boxes ─────────────────────────────────────────── */
.metric-row { display: flex; gap: 1rem; margin: 1.2rem 0; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 120px;
    background: #0d1421; border: 1px solid #1a2540;
    border-radius: 12px; padding: 1rem; text-align: center;
}
.metric-val {
    font-size: 1.35rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; color: #60a5fa;
}
.metric-lbl {
    font-size: 0.7rem; color: #4b5563;
    text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px;
}

/* ── Info cards ───────────────────────────────────────────── */
.info-card {
    background: #0d1421; border: 1px solid #1a2540;
    border-radius: 14px; padding: 1.25rem 1.4rem; margin-bottom: 0.8rem;
}
.info-card-title {
    font-size: 0.75rem; color: #4b5563;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.5rem; font-weight: 600;
}
.info-card-content { font-size: 0.9rem; color: #e2e8f0; line-height: 1.6; }

/* ── Action box ───────────────────────────────────────────── */
.action-box {
    border-radius: 12px; padding: 1.1rem 1.3rem;
    line-height: 1.75; font-size: 0.88rem; margin-top: 0.5rem;
}

/* ── Steps (sidebar) ──────────────────────────────────────── */
.step { display: flex; align-items: flex-start; gap: 10px; padding: 7px 0; }
.step-n {
    width: 24px; height: 24px; border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; color: #ffffff; flex-shrink: 0;
    box-shadow: 0 0 8px rgba(99, 102, 241, 0.3);
}
.step-t { font-size: 0.82rem; color: #cbd5e1; line-height: 1.45; }

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.55rem 1.8rem !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Download button ──────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(16,185,129,0.4) !important;
}

/* ── Section divider ──────────────────────────────────────── */
hr { border-color: #1a2540 !important; margin: 1.5rem 0 !important; }

/* ── Tab styling ──────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: #0d1421; border-radius: 10px; padding: 4px; gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px; font-weight: 500; color: #64748b !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1e3a5f !important; color: #60a5fa !important;
}

/* ── Spinner ──────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #60a5fa !important; }

/* ── Success/info/warning ─────────────────────────────────── */
.stSuccess { background: #022c22 !important; border-color: #10b981 !important; }
.stWarning { background: #2d1500 !important; border-color: #f59e0b !important; }

/* ── Footer ───────────────────────────────────────────────── */
.ew-footer {
    text-align: center; color: #1e2d3d;
    font-size: 0.78rem; padding: 2rem 0 1rem;
    border-top: 1px solid #1a2540; margin-top: 3rem;
}

/* ── Image border ─────────────────────────────────────────── */
[data-testid="stImage"] img { border-radius: 10px; }

/* ── Sidebar Glossary & Specs ─────────────────────────────── */
.glossary-card {
    background: #0d1421;
    border: 1px solid #1a2540;
    border-radius: 12px;
    padding: 0.85rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease-in-out;
}
.glossary-card:hover {
    border-color: #3b82f6;
    transform: translateY(-1px);
}
.glossary-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.glossary-icon {
    font-size: 1.15rem;
}
.glossary-title {
    font-size: 0.85rem;
    font-weight: 750;
    color: #60a5fa;
    letter-spacing: -0.01em;
}
.glossary-desc {
    font-size: 0.78rem;
    color: #94a3b8;
    line-height: 1.45;
}
.glossary-badge-row {
    display: flex;
    gap: 6px;
    margin-top: 8px;
    flex-wrap: wrap;
}
.glossary-badge {
    font-size: 0.68rem;
    padding: 2px 7px;
    border-radius: 6px;
    font-weight: 600;
}
.badge-healthy {
    background: rgba(16, 185, 129, 0.08);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.badge-faulty {
    background: rgba(239, 68, 68, 0.08);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.2);
}

.spec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 4px;
}
.spec-item {
    background: #0d1421;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 8px 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    transition: all 0.2s ease-in-out;
}
.spec-item:hover {
    border-color: #6366f1;
}
.spec-lbl {
    font-size: 0.62rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}
.spec-val {
    font-size: 0.8rem;
    font-weight: 700;
    color: #a78bfa;
    margin-top: 3px;
}

.deploy-card {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.deploy-item {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #0d1421;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 9px 12px;
    transition: all 0.2s ease-in-out;
}
.deploy-item:hover {
    border-color: #10b981;
}
.deploy-icon {
    font-size: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
}
.deploy-info {
    display: flex;
    flex-direction: column;
}
.deploy-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #cbd5e1;
}
.deploy-desc {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 1px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — Synthetic audio
# ─────────────────────────────────────────────────────────────────────────────
def _make_demo_audio(fault_type: str, duration: float = 5.0, seed: int = 7) -> np.ndarray:
    """
    Generate a synthetic engine audio array for the demo mode.

    Acoustic design
    ---------------
    healthy  — fundamental 70 Hz + 2 harmonics, tiny noise → low ZCR, stable centroid
    worn     — fundamental + bearing harmonics 230/460 Hz + moderate noise + impulses
    critical — multi-band high-frequency signal + heavy noise + frequent impulses
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, duration, int(duration * SR), dtype=np.float32)

    if fault_type == "healthy":
        y = (0.80 * np.sin(2 * np.pi *  70 * t)
           + 0.15 * np.sin(2 * np.pi * 140 * t)
           + 0.05 * np.sin(2 * np.pi * 210 * t)).astype(np.float32)
        y += rng.standard_normal(len(t)).astype(np.float32) * 0.010

    elif fault_type == "worn":
        y = (0.60 * np.sin(2 * np.pi *  70 * t)
           + 0.10 * np.sin(2 * np.pi * 140 * t)
           + 0.30 * np.sin(2 * np.pi * 230 * t)
           + 0.15 * np.sin(2 * np.pi * 460 * t)).astype(np.float32)
        y += rng.standard_normal(len(t)).astype(np.float32) * 0.055
        idx = rng.integers(0, len(t), size=30)
        y[idx] += rng.standard_normal(30).astype(np.float32) * 0.22

    elif fault_type == "critical":
        y = (0.40 * np.sin(2 * np.pi *   70 * t)
           + 0.30 * np.sin(2 * np.pi *  500 * t)
           + 0.20 * np.sin(2 * np.pi * 1000 * t)
           + 0.15 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
        y += rng.standard_normal(len(t)).astype(np.float32) * 0.120
        idx  = rng.integers(0, len(t), size=100)
        sign = rng.choice([-1.0, 1.0], size=100).astype(np.float32)
        y[idx] += sign * 0.70
    else:
        y = rng.standard_normal(int(duration * SR)).astype(np.float32)

    peak = float(np.max(np.abs(y)))
    return (y / peak).astype(np.float32) if peak > 1e-6 else y


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — Model (cached for session lifetime)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_model():
    """
    Load the trained model from disk.
    If models/model.pkl is absent, auto-train on synthetic data.
    Cached across the entire Streamlit session.
    """
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass  # file may be corrupt — fall through to re-train

    # Auto-train
    from src.training.train_runner import train_on_synthetic
    return train_on_synthetic(save_path=MODEL_PATH, n_per_class=300)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — Visualisation
# ─────────────────────────────────────────────────────────────────────────────
_BG   = "#0d1421"
_GRID = "#1a2540"
_TXT  = "#64748b"

def _plot_waveform(y: np.ndarray, sr: int, accent: str = "#60a5fa") -> bytes:
    """Render the waveform and return PNG bytes."""
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig, ax = plt.subplots(figsize=(7, 2.4), facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.plot(times, y, color=accent, linewidth=0.55, alpha=0.9)
    ax.fill_between(times, y, alpha=0.12, color=accent)
    ax.set_xlabel("Time (s)", color=_TXT, fontsize=8)
    ax.set_ylabel("Amplitude", color=_TXT, fontsize=8)
    ax.set_title("Waveform", color="#cbd5e1", fontsize=9, pad=6, fontweight="600")
    ax.tick_params(colors=_TXT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.4, alpha=0.7)
    ax.set_xlim(0, times[-1])
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _plot_mfcc(y: np.ndarray, sr: int) -> bytes:
    """Render the MFCC heatmap and return PNG bytes."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    fig, ax = plt.subplots(figsize=(7, 2.4), facecolor=_BG)
    ax.set_facecolor(_BG)
    img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax, cmap="magma")
    cbar = plt.colorbar(img, ax=ax, format="%+.0f")
    cbar.ax.tick_params(colors=_TXT, labelsize=7)
    cbar.ax.yaxis.label.set_color(_TXT)
    ax.set_title("MFCC Heatmap", color="#cbd5e1", fontsize=9, pad=6, fontweight="600")
    ax.set_xlabel("Time (s)", color=_TXT, fontsize=8)
    ax.set_ylabel("MFCC Coefficient", color=_TXT, fontsize=8)
    ax.tick_params(colors=_TXT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(_GRID)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — PDF Report
# ─────────────────────────────────────────────────────────────────────────────
def _generate_pdf(report: dict, wfm_png: bytes, mfcc_png: bytes) -> bytes:
    """
    Build a professional, multi-page PDF diagnostic report.

    Pages
    -----
    1. Cover — title, timestamp, status badge
    2. Methodology — how the system works (pipeline + feature glossary)
    3. Analysis Result — classification details + feature values table
    4. Visualisations — waveform + MFCC heatmap (embedded images)
    5. Recommendations — component assessment + recommended action + taxonomy
    """
    from fpdf import FPDF

    status = report.get("status", "UNKNOWN").lower()
    sr_rgb = _STATUS_RGB.get(status, (107, 114, 128))

    def clean_text(text):
        if not isinstance(text, str):
            text = str(text)
        # Safely map common Unicode characters that are unsupported by the default Helvetica font
        replacements = {
            "\u2013": "-",       # en-dash
            "\u2014": "-",       # em-dash
            "\u2192": "->",      # arrow
            "\u2022": "*",       # bullet
            "\u00b7": "-",       # middle dot
            "\u2264": "<=",      # less than or equal
            "\u26a0": "[WARNING]", # warning sign emoji
            "\ufe0f": "",        # variant selector
        }
        for orig, rep in replacements.items():
            text = text.replace(orig, rep)
        # Encode as Latin-1 with replacement to prevent FPDFUnicodeEncodingException on any other unsupported character
        return text.encode("latin-1", errors="replace").decode("latin-1")

    class PDF(FPDF):
        def cell(self, *args, **kwargs):
            new_args = list(args)
            if len(new_args) > 2:
                new_args[2] = clean_text(new_args[2])
            if "txt" in kwargs:
                kwargs["txt"] = clean_text(kwargs["txt"])
            elif "text" in kwargs:
                kwargs["text"] = clean_text(kwargs["text"])
            return super().cell(*new_args, **kwargs)

        def multi_cell(self, *args, **kwargs):
            new_args = list(args)
            if len(new_args) > 2:
                new_args[2] = clean_text(new_args[2])
            if "txt" in kwargs:
                kwargs["txt"] = clean_text(kwargs["txt"])
            elif "text" in kwargs:
                kwargs["text"] = clean_text(kwargs["text"])
            return super().multi_cell(*new_args, **kwargs)

        def header(self):
            if self.page_no() > 1:
                self.set_font("Helvetica", "B", 7)
                self.set_text_color(100, 120, 150)
                self.cell(0, 6, "ENGINE WHISPERER  |  AURALYTICS  |  DIAGNOSTIC REPORT", align="L")
                self.set_draw_color(30, 50, 80)
                self.set_line_width(0.3)
                self.line(10, 16, 200, 16)
                self.ln(5)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(100, 120, 150)
            ts = report.get("timestamp", "")
            self.cell(
                0, 8,
                f"Page {self.page_no()}  |  Generated: {ts}  |  Auralytics Engine Whisperer v1.0",
                align="C",
            )

    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(14, 14, 14)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def section(title: str, rgb=(96, 165, 250)):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*rgb)
        pdf.cell(0, 7, title, ln=True)
        pdf.set_draw_color(*rgb)
        pdf.set_line_width(0.35)
        pdf.line(14, pdf.get_y(), 196, pdf.get_y())
        pdf.ln(4)

    def body(text: str, rgb=(220, 230, 245), size=9.5):
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(*rgb)
        pdf.multi_cell(0, 5.5, text)
        pdf.ln(2)

    def kv(key: str, value: str, key_rgb=(148, 163, 184), val_rgb=(220, 230, 245)):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*key_rgb)
        pdf.cell(48, 6, key + ":", ln=False)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*val_rgb)
        pdf.multi_cell(0, 6, value)
        pdf.set_x(pdf.l_margin)

    def embed_image(png_bytes: bytes, caption: str, width: int = 178):
        buf = io.BytesIO(png_bytes)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 5, caption, ln=True)
        pdf.image(buf, x=14, w=width)
        pdf.ln(3)

    # ─── Page 1: Cover ────────────────────────────────────────────────────────
    pdf.add_page()
    # Dark background
    pdf.set_fill_color(6, 11, 24)
    pdf.rect(0, 0, 210, 297, "F")

    # Accent top bar
    r, g, b = sr_rgb
    pdf.set_fill_color(r, g, b)
    pdf.rect(0, 0, 210, 6, "F")

    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_text_color(96, 165, 250)
    pdf.cell(0, 13, "ENGINE WHISPERER", align="C", ln=True)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(167, 139, 250)
    pdf.cell(0, 7, "Auralytics — Sound-Based Engine Fault Diagnosis System", align="C", ln=True)

    pdf.ln(8)
    pdf.set_draw_color(*sr_rgb)
    pdf.set_line_width(0.7)
    pdf.line(35, pdf.get_y(), 175, pdf.get_y())
    pdf.ln(12)

    # Status badge
    pdf.set_font("Helvetica", "B", 40)
    pdf.set_text_color(*sr_rgb)
    icon_map = {"healthy": "[ OK ]", "worn": "[ WARN ]", "critical": "[ CRIT ]"}
    pdf.cell(0, 18, icon_map.get(status, "[ ? ]"), align="C", ln=True)

    pdf.set_font("Helvetica", "B", 28)
    pdf.cell(0, 12, report.get("status", "UNKNOWN"), align="C", ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(200, 210, 230)
    pdf.cell(0, 7, f"Confidence: {report.get('confidence', 0):.1f}%   |   Severity: {report.get('severity', '—')}", align="C", ln=True)
    pdf.cell(0, 7, f"Component: {report.get('component', '—')}", align="C", ln=True)

    pdf.ln(14)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 100, 130)
    pdf.cell(0, 6, f"Report generated: {report.get('timestamp', '—')}", align="C", ln=True)
    pdf.cell(0, 6, "Auralytics Engine Whisperer v1.0  |  Sound-Based Machine Diagnostics", align="C", ln=True)

    # Bottom bar
    pdf.set_y(-20)
    pdf.set_fill_color(r, g, b)
    pdf.rect(0, 287, 210, 10, "F")

    # ─── Page 2: Methodology ──────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(6, 11, 24)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.ln(2)

    section("1. How It Works — System Methodology")
    body(
        "Engine Whisperer diagnoses engine health by treating audio the same way speech-recognition "
        "systems treat human speech: raw waveform → compact numerical fingerprint → pattern classification.\n\n"
        "The core insight: a healthy engine produces a smooth, rhythmic pressure pattern because all its "
        "moving parts are synchronised and balanced. A failing component — a worn bearing, a cracked gear "
        "— introduces irregular vibration at specific frequencies that breaks that smoothness. These "
        "irregularities are consistent enough across many examples that a machine-learning classifier can "
        "learn to recognise them."
    )

    section("2. Processing Pipeline")
    stages = [
        ("Stage 1 — Load",      "Audio file or live mic array is loaded at 22 050 Hz sample rate (mono)."),
        ("Stage 2 — Denoise",   "Spectral noise reduction removes background hiss and room hum so the "
                                 "signal reflects the engine, not the recording environment."),
        ("Stage 3 — Normalise", "Peak-normalise amplitude to [-1, 1]. A loud recording and a quiet "
                                 "recording of the same fault must produce identical feature values."),
        ("Stage 4 — Frame",     "Slice into 1-second windows with 50% overlap (9 frames per 5-second "
                                 "clip). Overlapping windows give the model multiple looks at each fault."),
        ("Stage 5 — Extract",   "Compute MFCC (13 coeff), ZCR (1 value), Spectral Centroid (1 value) "
                                 "per frame → combine into a 15-element feature vector."),
        ("Stage 6 — Classify",  "Feed each frame's vector to the trained Random Forest (200 trees). "
                                 "Take the majority-vote label across all frames."),
        ("Stage 7 — Report",    "Map predicted class → component → recommended technician action + timestamp."),
    ]
    for stage, desc in stages:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(96, 165, 250)
        pdf.cell(50, 6, stage, ln=False)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(200, 210, 230)
        pdf.multi_cell(0, 6, desc)
        pdf.ln(1)

    pdf.ln(4)
    section("3. Feature Glossary")
    features_info = [
        ("MFCC (13 values)",
         "Mel-Frequency Cepstral Coefficients — the frequency 'texture' fingerprint. "
         "Computed via FFT → Mel-scale warping → DCT compression. A healthy engine shows "
         "smooth, uniform MFCC bars; a faulty engine shows irregular, spiky ones."),
        ("ZCR (1 value)",
         "Zero Crossing Rate — counts how many times the waveform crosses the zero-amplitude "
         "line per sample. Low = rhythmic (healthy). High = erratic / knocking (faulty)."),
        ("Spectral Centroid (1 value)",
         "The 'centre of mass' of the frequency spectrum (Hz). A healthy engine's energy "
         "stays concentrated at low frequencies. Bearing wear shifts energy upward, "
         "increasing this value progressively with fault severity."),
    ]
    for fname, fdesc in features_info:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(167, 139, 250)
        pdf.cell(0, 6, fname, ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(180, 190, 210)
        pdf.multi_cell(0, 5.5, fdesc)
        pdf.ln(3)

    # ─── Page 3: Analysis Result ──────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(6, 11, 24)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.ln(2)

    section("4. Analysis Result", rgb=sr_rgb)

    # Status box
    y0 = pdf.get_y()
    pdf.set_fill_color(r // 6, g // 6, b // 6)
    pdf.set_draw_color(r, g, b)
    pdf.set_line_width(0.6)
    pdf.rect(14, y0, 182, 30, "FD")
    pdf.set_y(y0 + 4)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 9, f"STATUS: {report.get('status', '?')}", align="C", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(220, 225, 240)
    pdf.cell(
        0, 7,
        f"Confidence: {report.get('confidence', 0):.1f}%   |   Severity: {report.get('severity', '?')}",
        align="C", ln=True
    )
    pdf.ln(10)

    section("5. Feature Values", rgb=(96, 165, 250))
    col_w = [70, 55, 57]
    hdrs  = ["Feature", "Value", "Unit"]
    rows_pdf = [
        ["MFCC Mean",          str(report.get("mfcc_mean", "N/A")),         "dB"],
        ["Zero Crossing Rate", str(report.get("zcr", "N/A")),               "crossings/sample"],
        ["Spectral Centroid",  str(report.get("spectral_centroid", "N/A")), "Hz"],
        ["Classification",     report.get("status", "?"),                   "label"],
        ["Model Confidence",   f"{report.get('confidence', 0):.1f}",        "%"],
        ["Severity Level",     report.get("severity", "?"),                  "—"],
    ]

    # Table header
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(20, 35, 65)
    pdf.set_text_color(148, 163, 184)
    for i, h in enumerate(hdrs):
        pdf.cell(col_w[i], 6.5, h, border=1, fill=True)
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 8)
    for ri, row in enumerate(rows_pdf):
        fill_c = (13, 22, 38) if ri % 2 == 0 else (17, 28, 50)
        pdf.set_fill_color(*fill_c)
        pdf.set_text_color(210, 220, 235)
        for i, cell in enumerate(row):
            pdf.cell(col_w[i], 6.5, str(cell), border=1, fill=True)
        pdf.ln()

    pdf.ln(5)
    section("6. Diagnostic Description", rgb=(96, 165, 250))
    body(report.get("description", "No description available."))

    # ─── Page 4: Visualisations ───────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(6, 11, 24)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.ln(2)

    section("7. Signal Visualisations")
    body(
        "The plots below show the acoustic fingerprint of the analysed audio clip. "
        "These are exactly what the Random Forest classifier 'sees' before making its prediction."
    )

    embed_image(wfm_png, "Figure 1 — Waveform: amplitude over time")
    embed_image(mfcc_png, "Figure 2 — MFCC Heatmap: frequency texture over time")

    body(
        "MFCC Interpretation:\n"
        "  • Healthy engine  — Smooth, uniform horizontal bands across all time frames.\n"
        "  • Worn engine     — Mild irregularity; slight brightening in upper coefficient bands.\n"
        "  • Critical engine — Chaotic, spiky pattern; strong energy in upper bands throughout.\n\n"
        "The Random Forest learned these patterns from 900+ synthetic training examples "
        "(300 per class) and applies a majority vote across 9 overlapping 1-second frames."
    )

    # ─── Page 5: Recommendations ──────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(6, 11, 24)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.ln(2)

    section("8. Component Assessment & Recommended Action", rgb=sr_rgb)
    kv("Status",    report.get("status", "?"),     val_rgb=sr_rgb)
    kv("Severity",  report.get("severity", "?"),   val_rgb=sr_rgb)
    kv("Component", report.get("component", "?"), val_rgb=(200, 210, 230))
    pdf.ln(4)

    # Action box
    y0 = pdf.get_y()
    pdf.set_fill_color(r // 7, g // 7, b // 7)
    pdf.set_draw_color(r, g, b)
    pdf.set_line_width(0.5)
    action_text = "RECOMMENDED ACTION:\n" + report.get("action", "—")
    # Measure height
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(r, g, b)
    pdf.multi_cell(182, 6, action_text, border=1, fill=True)
    pdf.ln(6)

    section("9. Fault Classification Taxonomy", rgb=(96, 165, 250))
    taxonomy = [
        ("HEALTHY",  (16, 185, 129),   "LOW",    "Smooth MFCC · low ZCR · stable centroid",     "Continue normal monitoring"),
        ("WORN",     (245, 158, 11),   "MEDIUM", "Mild MFCC irregularity · elevated centroid",   "Schedule inspection ≤ 14 days"),
        ("CRITICAL", (239, 68, 68),    "HIGH",   "Chaotic MFCC · high ZCR · sharp centroid rise","Stop operation, inspect now"),
    ]
    t_cols = [30, 25, 68, 59]
    t_hdrs = ["Status", "Severity", "Acoustic Signature", "Action"]
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_fill_color(20, 35, 65)
    pdf.set_text_color(148, 163, 184)
    for i, h in enumerate(t_hdrs):
        pdf.cell(t_cols[i], 6, h, border=1, fill=True)
    pdf.ln()
    for s_label, s_rgb, sev, sig, act in taxonomy:
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(*s_rgb)
        pdf.cell(t_cols[0], 6, s_label, border=1)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(200, 210, 230)
        data = [sev, sig, act]
        for i, cell in enumerate(data, start=1):
            pdf.cell(t_cols[i], 6, cell, border=1)
        pdf.ln()

    pdf.ln(8)
    section("10. Disclaimer", rgb=(80, 100, 130))
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(80, 100, 130)
    pdf.multi_cell(
        0, 5,
        "This report is generated by an automated machine-learning system trained on synthetic "
        "acoustic data. It is intended as a decision-support tool for qualified technicians and "
        "should not replace hands-on mechanical inspection. Auralytics assumes no liability for "
        "decisions made solely on the basis of this automated report. Always verify critical "
        "findings with a certified technician before taking operational action."
    )

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 0.8rem;'>
        <div style='font-size:2.8rem;'>🔧</div>
        <div style='font-size:1.05rem;font-weight:800;color:#60a5fa;letter-spacing:-0.5px;'>Engine Whisperer</div>
        <div style='font-size:0.72rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;'>Auralytics v1.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 🧠 How It Works")
    with st.expander("Pipeline Overview", expanded=True):
        steps = [
            "**Load** — WAV/MP3/FLAC or mic array at 22 050 Hz",
            "**Denoise** — spectral noise reduction",
            "**Normalise** — peak amplitude scaling to [-1, 1]",
            "**Frame** — 1-second windows, 50% overlap",
            "**Extract** — MFCC (13) + ZCR + Spectral Centroid",
            "**Classify** — Random Forest, 200 trees, majority vote",
            "**Report** — component + action + timestamp + PDF",
        ]
        for i, s in enumerate(steps, 1):
            st.markdown(f"""
            <div class="step">
                <div class="step-n">{i}</div>
                <div class="step-t">{s}</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("📖 Feature Glossary"):
        st.markdown("""
        <div class="glossary-card">
            <div class="glossary-header">
                <span class="glossary-icon">📊</span>
                <span class="glossary-title">MFCCs</span>
            </div>
            <div class="glossary-desc">Frequency "texture" fingerprint (13 coefficients). Captures overall sound envelope and quality.</div>
            <div class="glossary-badge-row">
                <span class="glossary-badge badge-healthy">Healthy: Smooth</span>
                <span class="glossary-badge badge-faulty">Faulty: Spiky</span>
            </div>
        </div>
        <div class="glossary-card">
            <div class="glossary-header">
                <span class="glossary-icon">〰️</span>
                <span class="glossary-title">Zero Crossing Rate</span>
            </div>
            <div class="glossary-desc">Rate of transitions across the zero amplitude axis. Measures signal chaotic oscillation.</div>
            <div class="glossary-badge-row">
                <span class="glossary-badge badge-healthy">Low: Rhythmic</span>
                <span class="glossary-badge badge-faulty">High: Erratic Knocking</span>
            </div>
        </div>
        <div class="glossary-card">
            <div class="glossary-header">
                <span class="glossary-icon">📈</span>
                <span class="glossary-title">Spectral Centroid</span>
            </div>
            <div class="glossary-desc">"Center of mass" of sound frequencies (pitch). Increases as energy shifts to higher bands.</div>
            <div class="glossary-badge-row">
                <span class="glossary-badge badge-healthy">Stable: Low Freq</span>
                <span class="glossary-badge badge-faulty">Elevated: Friction</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🤖 Model Details"):
        st.markdown("""
        <div class="spec-grid">
            <div class="spec-item">
                <span class="spec-lbl">Algorithm</span>
                <span class="spec-val" style="color: #60a5fa; font-size: 0.72rem;">Random Forest</span>
            </div>
            <div class="spec-item">
                <span class="spec-lbl">Estimators</span>
                <span class="spec-val">200 Trees</span>
            </div>
            <div class="spec-item">
                <span class="spec-lbl">Feature Size</span>
                <span class="spec-val">15 Metrics</span>
            </div>
            <div class="spec-item">
                <span class="spec-lbl">Target Classes</span>
                <span class="spec-val">3 Labels</span>
            </div>
            <div class="spec-item">
                <span class="spec-lbl">Data Split</span>
                <span class="spec-val">80/20 Train</span>
            </div>
            <div class="spec-item">
                <span class="spec-lbl">Min Accuracy</span>
                <span class="spec-val" style="color: #10b981;">&ge; 85%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🌐 Deployment"):
        st.markdown("""
        <div class="deploy-card">
            <div class="deploy-item">
                <span class="deploy-icon">☁️</span>
                <div class="deploy-info">
                    <div class="deploy-title">Streamlit Cloud</div>
                    <div class="deploy-desc">Hosted on share.streamlit.io</div>
                </div>
            </div>
            <div class="deploy-item">
                <span class="deploy-icon">⚡</span>
                <div class="deploy-info">
                    <div class="deploy-title">On-Demand Training</div>
                    <div class="deploy-desc">Auto-trains on first launch</div>
                </div>
            </div>
            <div class="deploy-item">
                <span class="deploy-icon">🔊</span>
                <div class="deploy-info">
                    <div class="deploy-title">File Types</div>
                    <div class="deploy-desc">WAV, MP3, FLAC, OGG</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.caption("Inspired by Caterpillar TIS/ADSD diagnostic workflows.")
    st.caption("© 2024 Auralytics · MIT Licence")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ew-header">
    <div class="ew-title">🔧 Engine Whisperer</div>
    <div class="ew-sub">Auralytics · ML-Powered Engine Fault Diagnosis · MFCC + Random Forest</div>
</div>
""", unsafe_allow_html=True)

# Model loading
with st.spinner("🔄 Loading diagnostic model — auto-training if needed…"):
    model = _get_model()

st.success("✅ Diagnostic model ready — Random Forest (200 trees)", icon="🤖")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Input Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_demo, tab_upload = st.tabs(["🎯  Demo Mode", "📁  Upload Audio"])

# ── Tab: Demo Mode ────────────────────────────────────────────────────────────
with tab_demo:
    st.markdown("### 🎯 Synthetic Engine Scenario")
    st.markdown(
        "Select an engine condition below. A synthetic audio signal is generated "
        "mathematically to mimic that fault state, then run through the **exact same "
        "pipeline** used on real audio — no actual recording required."
    )
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        choice = st.selectbox(
            "Engine scenario",
            list(_DEMO_OPTIONS.keys()),
            key="demo_choice",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_demo = st.button("▶  Run Analysis", key="btn_demo", use_container_width=True)

    if run_demo:
        fault = _DEMO_OPTIONS[choice]
        if fault is None:
            fault = np.random.choice(["healthy", "worn", "critical"])
        st.session_state.update({
            "audio":     _make_demo_audio(fault, seed=np.random.randint(1, 999)),
            "src_label": fault,
            "ready":     True,
        })
        st.rerun()

# ── Tab: Upload Audio ─────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("### 📁 Upload an Audio File")
    st.markdown(
        "Upload a WAV, MP3, FLAC, or OGG recording of a running engine or motor "
        "(3–10 seconds recommended). The system applies identical preprocessing and "
        "feature extraction to what was used during training."
    )
    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["wav", "mp3", "flac", "ogg"],
        key="file_uploader",
    )
    if uploaded is not None:
        with st.spinner("Loading audio…"):
            try:
                buf = io.BytesIO(uploaded.read())
                y_up, _ = librosa.load(buf, sr=SR, mono=True)
                st.session_state.update({
                    "audio":     y_up.astype(np.float32),
                    "src_label": "uploaded",
                    "ready":     False,
                })
                st.success(f"✅ Loaded: {uploaded.name}  ({len(y_up)/SR:.1f} s)")
            except Exception as e:
                st.error(f"Could not load audio: {e}")

    if (st.session_state.get("src_label") == "uploaded"
            and "audio" in st.session_state
            and not st.session_state.get("ready", False)):
        if st.button("▶  Analyze Upload", key="btn_upload"):
            st.session_state["ready"] = True
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Analysis & Results
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("ready") and "audio" in st.session_state:
    audio_arr = np.asarray(st.session_state["audio"], dtype=np.float32)
    src_label = st.session_state.get("src_label", "unknown")

    st.divider()
    st.markdown("## 🔍 Diagnostic Analysis")

    # ── Run inference ────────────────────────────────────────────────────────
    with st.spinner("Extracting acoustic features and running classifier…"):
        time.sleep(0.3)
        result = classify_live(model, audio_arr, sr=SR)

    label      = result.get("label", "unknown")
    confidence = result.get("confidence", 0.0)
    features   = result.get("features")
    report     = generate_report(label, confidence, features)
    colors     = _COLORS.get(label, _COLORS["unknown"])
    icon       = _ICONS.get(label, "❓")

    # ── Visualisations ───────────────────────────────────────────────────────
    col_wfm, col_mfcc = st.columns(2)
    with col_wfm:
        st.markdown("#### 〰 Waveform")
        wfm_png  = _plot_waveform(audio_arr, SR, accent=colors["accent"])
        st.image(wfm_png, use_container_width=True)
    with col_mfcc:
        st.markdown("#### 🎨 MFCC Heatmap")
        mfcc_png = _plot_mfcc(audio_arr, SR)
        st.image(mfcc_png, use_container_width=True)

    st.divider()

    # ── Status Card ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="status-card" style="
        background: linear-gradient(135deg, {colors['bg']} 0%, #060b18 100%);
        border: 2px solid {colors['accent']};
        box-shadow: 0 0 50px {colors['glow']};
    ">
        <span class="status-icon">{icon}</span>
        <div class="status-lbl" style="color:{colors['accent']};">{report['status']}</div>
        <div class="status-desc" style="color:{colors['light']}99;">{report['description']}</div>
        <span class="status-badge" style="background:{colors['badge']};color:{colors['light']};">
            SEVERITY: {report['severity']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bar ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="conf-wrap">
        <div class="conf-header">
            <span class="conf-label">Model Confidence</span>
            <span class="conf-value" style="color:{colors['accent']};">{confidence:.1f}%</span>
        </div>
        <div class="conf-bg">
            <div class="conf-fill" style="width:{min(confidence,100):.0f}%;
                background:linear-gradient(90deg,{colors['accent']},{colors['accent']}88);"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature metrics ───────────────────────────────────────────────────────
    if features is not None:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val">{report.get('mfcc_mean', 'N/A')}</div>
                <div class="metric-lbl">MFCC Mean (dB)</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{report.get('zcr', 'N/A')}</div>
                <div class="metric-lbl">Zero Crossing Rate</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{report.get('spectral_centroid', 'N/A')}</div>
                <div class="metric-lbl">Spectral Centroid (Hz)</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{len(result.get('frame_labels', []))}</div>
                <div class="metric-lbl">Frames Analysed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Component + Action ────────────────────────────────────────────────────
    col_cmp, col_act = st.columns(2)
    with col_cmp:
        st.markdown("#### 🔩 Flagged Component")
        st.markdown(f"""
        <div class="info-card" style="border-color:{colors['accent']}55;">
            <div class="info-card-title">Component</div>
            <div class="info-card-content" style="color:{colors['accent']};font-weight:600;">
                ⚙️  {report['component']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_act:
        st.markdown("#### 📋 Recommended Action")
        sev_bg     = {"LOW": "#022c22", "MEDIUM": "#2d1500", "HIGH": "#2d0000"}.get(report["severity"], "#111827")
        sev_border = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}.get(report["severity"], "#6b7280")
        st.markdown(f"""
        <div class="action-box" style="background:{sev_bg};border:1px solid {sev_border}55;color:#e2e8f0;">
            {report['action']}
        </div>
        """, unsafe_allow_html=True)

    # ── Frame-level breakdown ─────────────────────────────────────────────────
    with st.expander("🔬 Frame-Level Analysis (Advanced View)"):
        frame_labels = result.get("frame_labels", [])
        frame_confs  = result.get("frame_confidences", [])
        if frame_labels:
            st.markdown(f"**{len(frame_labels)} frames** analysed · Majority vote → **{label.upper()}**")
            n_cols = min(len(frame_labels), 9)
            cols = st.columns(n_cols)
            for i, (fl, fc) in enumerate(zip(frame_labels, frame_confs)):
                fc_colors = _COLORS.get(fl, _COLORS["unknown"])
                with cols[i % n_cols]:
                    st.markdown(f"""
                    <div style="text-align:center;background:#0d1421;
                         border:1px solid {fc_colors['accent']}55;border-radius:10px;
                         padding:10px 4px;margin:2px 0;">
                        <div style="font-size:1.1rem;">{_ICONS.get(fl,'?')}</div>
                        <div style="font-size:0.68rem;color:{fc_colors['accent']};font-weight:600;">{fl[:3].upper()}</div>
                        <div style="font-size:0.62rem;color:#4b5563;">{fc:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.divider()

    # ── Full Report + PDF ──────────────────────────────────────────────────────
    st.markdown("## 📄 Diagnostic Service Report")
    col_rep, col_dl = st.columns([3, 1])

    with col_rep:
        with st.expander("📋 View Full Report Text", expanded=True):
            st.code(f"""
═══════════════════════════════════════════════════════════
  ENGINE WHISPERER — AURALYTICS DIAGNOSTIC REPORT
═══════════════════════════════════════════════════════════
  Timestamp   : {report['timestamp']}
  Status      : {report['status']}
  Severity    : {report['severity']}
  Confidence  : {report['confidence']}%
───────────────────────────────────────────────────────────
  COMPONENT   : {report['component']}
───────────────────────────────────────────────────────────
  ACTION      : {report['action']}
───────────────────────────────────────────────────────────
  DESCRIPTION : {report['description']}
───────────────────────────────────────────────────────────
  FEATURES
    MFCC Mean          : {report.get('mfcc_mean', 'N/A')} dB
    Zero Crossing Rate : {report.get('zcr', 'N/A')}
    Spectral Centroid  : {report.get('spectral_centroid', 'N/A')} Hz
═══════════════════════════════════════════════════════════
  Generated by Engine Whisperer v1.0 | Auralytics
═══════════════════════════════════════════════════════════
""", language=None)

    with col_dl:
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.spinner("Generating PDF…"):
            pdf_bytes = _generate_pdf(report, wfm_png, mfcc_png)
        fname = f"engine_report_{report['status'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label="⬇️  Download PDF Report",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
            key="dl_pdf",
        )

    # ── Reset ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  Run Another Analysis", key="btn_reset"):
        for key in ["audio", "src_label", "ready"]:
            st.session_state.pop(key, None)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ew-footer">
    🔧 Engine Whisperer · Auralytics v1.0 · Sound-Based Engine Fault Diagnosis<br>
    librosa · scikit-learn · Streamlit · MFCC + ZCR + Spectral Centroid · Random Forest<br>
    <span style='color:#0f172a;'>
        Methodology mirrors Caterpillar TIS/ADSD diagnostic workflows
    </span>
</div>
""", unsafe_allow_html=True)
