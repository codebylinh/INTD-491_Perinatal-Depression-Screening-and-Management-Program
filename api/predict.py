"""
Postpartum Depression Risk Prediction API
Vercel Serverless Function (Python)
Model: F+S+T Pipeline (cls_pipe_FST + reg_pipe_FST)
"""
from http.server import BaseHTTPRequestHandler
import json
import joblib
import numpy as np
import os
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_load_error = None

try:
    cls_model = joblib.load(os.path.join(BASE_DIR, "cls_pipe_FST.pkl"))
    reg_model = joblib.load(os.path.join(BASE_DIR, "reg_pipe_FST.pkl"))
    with open(os.path.join(BASE_DIR, "model_config.json")) as f:
        config = json.load(f)
    THRESHOLD = config["threshold"]
except Exception as e:
    _load_error = f"{type(e).__name__}: {e}"
    cls_model = reg_model = config = None
    THRESHOLD = None

# ═══════════════════════════════════════════════════════════════
# Model Performance Metrics (from F+S+T evaluation, slide 13/14)
# ═══════════════════════════════════════════════════════════════
MODEL_METRICS = {
    "classification": {
        "recall": 0.842,
        "precision": 0.356,
        "f1": 0.500,
        "roc_auc": 0.728,
        "pr_auc": 0.351,
        "cv_auc": 0.727,
        "cv_auc_sd": 0.030,
        "threshold_used": round(THRESHOLD, 4)
    },
    "regression": {
        "mae": 2.269,
        "rmse": 3.182,
        "r2": 0.254,
        "mae_high_risk": 4.553,
        "note": "Regression output is a research tool only. MAE of 4.5 on severe cases is too high for individual clinical decisions."
    }
}

# ═══════════════════════════════════════════════════════════════
# Prediction Logic
# ═══════════════════════════════════════════════════════════════
def predict(input_data):
    df = pd.DataFrame([input_data])

    # Classification: probability of PPD
    prob = float(cls_model.predict_proba(df)[0][1])
    high_risk = prob >= THRESHOLD

    # Regression: severity score (only if high risk)
    severity_score = float(reg_model.predict(df)[0]) if high_risk else None

    # ── Risk Tier ──────────────────────────────────────────────
    if prob < 0.10:
        risk_level = "low"
        trajectory = "Low Stable"
        color = "green"
        risk_label = "Low Risk"
        actions = [
            "Continue routine antenatal care as scheduled.",
            "Re-screen at next trimester visit using EPDS.",
            "Provide general maternal mental health resources.",
            "Encourage open communication with care provider about mood changes."
        ]

    elif prob < 0.405:
        risk_level = "moderate"
        trajectory = "Moderate / Rising"
        color = "orange"
        risk_label = "Moderate Risk"
        actions = [
            "Schedule a follow-up appointment at the third trimester visit.",
            "Monitor EPDS score trend across remaining trimesters.",
            "Consider referral to a maternal mental health support program.",
            "Assess for anxiety (GAD-7) and stressful life events (SLE).",
            "Discuss social support systems and coping strategies with patient."
        ]

    else:
        risk_level = "high"
        trajectory = "High Persistent"
        color = "red"
        risk_label = "High Risk"
        actions = [
            "Trigger mandatory clinical review — do not rely on model flag alone.",
            "Arrange immediate referral to a mental health professional.",
            "Prioritise postpartum follow-up within 2 weeks of birth.",
            "Assess ACE score and stressful life event (SLE) burden for escalation planning.",
            "Consider closer monitoring throughout remaining pregnancy.",
            "Notify care team and document risk flag in patient record."
        ]

    # ── Build Response ─────────────────────────────────────────
    return {
        # Core prediction
        "probability": round(prob, 4),
        "risk_score_percent": round(prob * 100, 1),
        "high_risk": high_risk,
        "threshold": round(THRESHOLD, 4),

        # Risk classification
        "risk_level": risk_level,
        "risk_label": risk_label,
        "trajectory": trajectory,
        "color": color,

        # Severity (regression — research use only)
        "severity_score": round(severity_score, 2) if severity_score is not None else None,
        "severity_note": (
            "Predicted postpartum EPDS score. For research reference only — "
            "not recommended for individual clinical decisions (MAE ≈ 4.5 on severe cases)."
            if severity_score is not None else None
        ),

        # Recommended clinical actions
        "recommended_actions": actions,

        # Model metadata
        "model": "F+S+T",
        "model_metrics": MODEL_METRICS,

        # Clinical disclaimer
        "disclaimer": (
            "This tool is a screening aid only and does not constitute a clinical diagnosis. "
            "All risk flags must be reviewed and acted upon by a qualified healthcare provider. "
            "Recall = 0.842: for every 10 true PPD cases, ~17 false positives are generated. "
            "Acceptable for population screening — not for individual diagnosis."
        )
    }


# ═══════════════════════════════════════════════════════════════
# HTTP Handler
# ═══════════════════════════════════════════════════════════════
class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            input_data = json.loads(body)

            result = predict(input_data)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            error_response = {
                "error": str(e),
                "type": type(e).__name__
            }
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())

    def do_GET(self):
        health = {
            "status": "ok",
            "model": "F+S+T Pipeline",
            "threshold": THRESHOLD,
            "load_error": _load_error,  
            "description": "Postpartum Depression Risk Prediction API",
            "model_metrics": MODEL_METRICS,
            "endpoints": {
                "GET /api/predict": "Health check + model info",
                "POST /api/predict": "Run risk prediction (send JSON body with patient features)"
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(health).encode())

    def log_message(self, format, *args):
        pass  # Suppress default request logging in Vercel
