"""
Reporting module for Engine Whisperer.

Translates raw classifier output (predicted label + confidence + feature values)
into a structured service report that a technician can act on immediately.

The report combines three signals:
  • Status      — directly the predicted class
  • Component   — inferred from which acoustic pattern dominated the prediction
  • Action      — a simple, unambiguous next step for the technician
"""

from datetime import datetime
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Fault taxonomy — the lookup tables that encode domain knowledge
# ─────────────────────────────────────────────────────────────────────────────
_COMPONENT: dict = {
    "healthy":  "All components nominal",
    "worn":     "Bearing — early-stage wear detected",
    "critical": "Bearing / Gear — severe fault signature",
}

_ACTION: dict = {
    "healthy": (
        "No action required. Continue normal monitoring schedule. "
        "Re-record in 30 days for trend tracking."
    ),
    "worn": (
        "Schedule a bearing inspection within 14 days. "
        "Check bearing clearance tolerance and lubrication levels. "
        "Re-record audio in 7 days to monitor fault progression."
    ),
    "critical": (
        "⚠️  STOP OPERATION IMMEDIATELY. "
        "Risk of catastrophic component failure is HIGH. "
        "Perform a full bearing and gear-train inspection before restarting. "
        "Do not operate until cleared by a qualified technician."
    ),
}

_SEVERITY: dict = {
    "healthy":  "LOW",
    "worn":     "MEDIUM",
    "critical": "HIGH",
}

_DESCRIPTION: dict = {
    "healthy": (
        "All frequency patterns are within normal operating range. "
        "The MFCC fingerprint is smooth and rhythmically consistent. "
        "ZCR is low, indicating a regular engine rhythm. "
        "Spectral centroid is stable — no energy shift toward higher frequencies. "
        "No abnormal vibration signatures detected."
    ),
    "worn": (
        "Early-stage bearing wear signatures detected. "
        "The MFCC shows mild irregularity in mid-frequency bands. "
        "Spectral centroid is moderately elevated, indicating energy is beginning "
        "to shift toward bearing defect frequencies. "
        "ZCR is slightly elevated compared to a healthy baseline."
    ),
    "critical": (
        "Severe fault signatures detected. "
        "Chaotic MFCC pattern with high Zero Crossing Rate indicates erratic, "
        "impulsive vibration consistent with a seized bearing or cracked gear teeth. "
        "Spectral centroid is sharply elevated — strong energy concentration at "
        "high frequencies. Immediate intervention is required."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────
def map_fault_to_component(label: str) -> str:
    """Return the component name for a predicted fault label."""
    return _COMPONENT.get(label, "Unknown component — manual inspection required")


def map_fault_to_action(label: str) -> str:
    """Return the recommended technician action for a predicted fault label."""
    return _ACTION.get(label, "Consult a certified technician immediately.")


def map_fault_to_severity(label: str) -> str:
    """Return the severity level string for a predicted fault label."""
    return _SEVERITY.get(label, "UNKNOWN")


def map_fault_to_description(label: str) -> str:
    """Return the plain-English diagnostic description for a predicted label."""
    return _DESCRIPTION.get(label, "No description available.")


def generate_report(
    label:      str,
    confidence: float,
    features:   "np.ndarray | None" = None,
) -> dict:
    """
    Build a complete structured service report dictionary.

    Parameters
    ----------
    label      : 'healthy' | 'worn' | 'critical'
    confidence : model confidence in [0, 100]
    features   : optional 1-D NumPy array of mean feature values
                 (shape: FEATURE_SIZE = 15)

    Returns
    -------
    dict with the following keys:
      timestamp, date, time, status, severity, component, action,
      description, confidence  (+ mfcc_mean, zcr, spectral_centroid if features given)
    """
    now = datetime.now()
    report: dict = {
        "timestamp":   now.strftime("%Y-%m-%d %H:%M:%S"),
        "date":        now.strftime("%d %B %Y"),
        "time":        now.strftime("%H:%M:%S"),
        "status":      label.upper(),
        "severity":    map_fault_to_severity(label),
        "component":   map_fault_to_component(label),
        "action":      map_fault_to_action(label),
        "description": map_fault_to_description(label),
        "confidence":  round(float(confidence), 1),
    }

    if features is not None:
        features = np.asarray(features, dtype=np.float64)
        n_mfcc = len(features) - 2          # last 2 are ZCR and centroid
        report.update({
            "mfcc_mean":         round(float(np.mean(features[:n_mfcc])), 4),
            "zcr":               round(float(features[-2]), 6),
            "spectral_centroid": round(float(features[-1]), 2),
        })

    return report
