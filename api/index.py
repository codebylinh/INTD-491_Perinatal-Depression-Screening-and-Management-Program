"""
Postpartum Depression Risk Prediction API
Vercel Serverless Function (Python)

Replicates the trained sklearn pipeline:
  ColumnTransformer (median impute + scale | most-frequent impute + one-hot)
  → LogisticRegression

Model: Logistic Regression with balanced class weights + random oversampling
Trained on N=691 samples, 3.2% positive rate (FPEPDS >= 13)
"""

from http.server import BaseHTTPRequestHandler
import json
import math
from flask import Flask, request, jsonify

app = Flask(__name__)

def run_prediction(data):
    # move your existing prediction logic here
    return {
        "risk_probability": 0.42,
        "risk_label": "Moderate",
        "icon": "⚠️"
    }

@app.route("/api/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({"status": "ok"})

    data = request.get_json(silent=True) or {}
    result = run_prediction(data)
    return jsonify(result)

# ═══════════════════════════════════════════════════════════════
# Trained Model Parameters (exported from sklearn pipeline)
# ═══════════════════════════════════════════════════════════════

NUMERIC_COLS = [
    "Age", "Number_of_Previous_Children", "F_Gestational_Week",
    "FEPDS", "FGAD7", "ACE_Score", "SLE_Score", "BMI"
]

CATEGORICAL_COLS = [
    "Residence_Type", "Education_Level", "Employment_Status",
    "Marital_Status", "Household_Income_After_Tax", "Smoking_Status",
    "Secondhand_Smoking", "Drinking_Status", "Unplanned_Pregnancy",
    "Pregnancy_Method"
]

# SimpleImputer(strategy="median") fitted values
NUM_MEDIANS = [
    27.3424657534247, 0.0, 9.71428571428571,
    6.0, 4.0, 0.0, 0.0, 20.429418362441915
]

# StandardScaler fitted values
SCALER_MEAN = [
    27.48823046534298, 0.17415730337078653, 9.677367576243983,
    5.9344569288389515, 4.28932584269663, 0.6404494382022472,
    0.8717228464419475, 20.929153800211754
]

SCALER_SCALE = [
    3.697118619255367, 0.3937797507612372, 1.6544573482333824,
    2.8412115605917148, 2.8691172210122287, 1.1125486919738807,
    1.1745690220695337, 3.2061462225285635
]

# SimpleImputer(strategy="most_frequent") fitted values
CAT_MOST_FREQUENT = [
    "Urban (county-level or above)", "College / Bachelor", "Full-time",
    "Married", "50k\u2013200k", "Never smoked", "Never or almost never",
    "No drinking", 0, "Natural conception"
]

# OneHotEncoder categories (fitted)
ENCODER_CATEGORIES = [
    ["Rural (township-level or below)", "Urban (county-level or above)"],
    ["College / Bachelor", "High school / vocational", "Middle school",
     "Postgraduate or above", "Primary school"],
    ["Full-time", "Paid leave", "Part-time", "Unemployed"],
    ["Divorced", "Married", "Unmarried", "Widowed"],
    ["50k\u2013200k", "<50k", ">200k"],
    ["Currently smoking", "Former smoker (quit \u22653 months)",
     "Never smoked", "Quit smoking after pregnancy"],
    ["A few times per month", "A few times per week",
     "A few times per year", "Almost every day", "Never or almost never"],
    ["Frequent drinking (\u22651 time/week)", "No drinking",
     "Occasional drinking (1\u20133 times/month)"],
    [0, 1],
    ["IVF/ICSI pregnancy", "Natural conception", "Other assisted reproduction"]
]

# LogisticRegression coefficients and intercept
COEFFICIENTS = [
    -0.8701540625443571, -2.27806876711144, 0.24669036126265553,
    0.29587190375137556, -0.09719068938131795, 0.7767669430763902,
    0.9209379662165147, 0.0028790754954595306, -0.7473247701458396,
    0.734912892128883, 0.37730247406185224, 0.7440029739507678,
    -2.8318630075204347, 1.7763125750015665, -0.07816689351077571,
    0.10974904522830906, -0.5677439311591642, 1.6952921579656508,
    -1.249709150051759, -0.6131824636704507, -0.19293027978941985,
    1.0363474697709918, -0.24264660432809249, 0.23877428413460425,
    -1.418114665438719, 1.166928503287136, -0.4533298450003786,
    -0.7091560154774249, 0.44459910505381184, 0.7054748774070164,
    -1.7977824187921998, 0.5721072555239243, -0.07035556095652439,
    0.19643733255061543, 1.0871815136572114, -0.4340569276682171,
    1.425113880776896, -1.0034688311256514, -0.07510005843998413,
    0.06268818042297637, -0.4683511733464067, 1.6360269592573393,
    -1.1800876639281224
]

INTERCEPT = -5.504133029308136
THRESHOLD = 0.1


def sigmoid(z):
    """Numerically stable sigmoid function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


def predict(input_data):
    """
    Replicate the full sklearn pipeline prediction:
    1. Impute missing numeric values with training medians
    2. Standard-scale numeric values
    3. Impute missing categorical values with most frequent
    4. One-hot encode categorical values
    5. Logistic regression dot product + sigmoid
    """

    # Step 1 & 2: Numeric preprocessing
    num_scaled = []
    for i, col in enumerate(NUMERIC_COLS):
        val = input_data.get(col)
        if val is None or val == "" or (isinstance(val, float) and math.isnan(val)):
            val = NUM_MEDIANS[i]
        else:
            val = float(val)
        # Standard scale
        scaled = (val - SCALER_MEAN[i]) / SCALER_SCALE[i]
        num_scaled.append(scaled)

    # Step 3 & 4: Categorical preprocessing
    cat_encoded = []
    for i, col in enumerate(CATEGORICAL_COLS):
        val = input_data.get(col)
        if val is None or val == "":
            val = CAT_MOST_FREQUENT[i]

        categories = ENCODER_CATEGORIES[i]
        # Type-match for comparison
        if isinstance(categories[0], int):
            try:
                val = int(val)
            except (ValueError, TypeError):
                val = CAT_MOST_FREQUENT[i]

        # One-hot encode
        for cat in categories:
            cat_encoded.append(1.0 if val == cat else 0.0)

    # Step 5: Combine and compute logistic regression
    features = num_scaled + cat_encoded
    z = INTERCEPT
    for j in range(len(features)):
        z += features[j] * COEFFICIENTS[j]

    probability = sigmoid(z)

    return {
        "probability": round(probability, 4),
        "risk_score_percent": round(probability * 100, 1),
        "high_risk": probability >= THRESHOLD,
        "threshold": THRESHOLD,
        "risk_level": (
            "low" if probability < 0.10
            else "moderate" if probability < 0.30
            else "high"
        )
    }


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler."""

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        """Handle prediction request."""
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
        """Health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "ok",
            "model": "Logistic Regression",
            "features": len(COEFFICIENTS),
            "threshold": THRESHOLD,
            "description": "Postpartum Depression Risk Prediction API"
        }).encode())
