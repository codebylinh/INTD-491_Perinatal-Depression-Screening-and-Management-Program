"""
Postpartum Depression Risk Prediction API
Vercel Serverless Function (Python)

Model: F+S+T Pipeline (cls_pipe_FST + reg_pipe_FST)
Uses joblib to load trained sklearn pipelines directly.
"""

from http.server import BaseHTTPRequestHandler
import json
import joblib
import numpy as np
import os

# ═══════════════════════════════════════════════════════════════
# Load Models (F+S+T pipelines)
# ═══════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cls_model = joblib.load(os.path.join(BASE_DIR, "cls_pipe_FST.pkl"))
reg_model = joblib.load(os.path.join(BASE_DIR, "reg_pipe_FST.pkl"))

with open(os.path.join(BASE_DIR, "model_config.json")) as f:
    config = json.load(f)

THRESHOLD = config["threshold"]


def predict(input_data):
    """
    Run input through the F+S+T sklearn pipelines.
    cls_pipe_FST → risk probability (classification)
    reg_pipe_FST → severity score (regression, only if high risk)
    """

    # Convert input dict to a single-row DataFrame
    import pandas as pd
    df = pd.DataFrame([input_data])

    # Classification: probability of PPD
    prob = float(cls_model.predict_proba(df)[0][1])
    high_risk = prob >= THRESHOLD

    # Regression: severity score (EPDS predicted score)
    severity_score = float(reg_model.predict(df)[0]) if high_risk else None

    return {
        "probability": round(prob, 4),
        "risk_score_percent": round(prob * 100, 1),
        "high_risk": high_risk,
        "threshold": round(THRESHOLD, 4),
        "severity_score": round(severity_score, 2) if severity_score is not None else None,
        "risk_level": (
            "low" if prob < 0.10
            else "moderate" if prob < 0.30
            else "high"
        ),
        "model": "F+S+T"  # version tag so you can confirm it's live
    }


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
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "ok",
            "model": "F+S+T Pipeline",
            "threshold": THRESHOLD,
            "description": "Postpartum Depression Risk Prediction API"
        }).encode())
