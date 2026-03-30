# Postpartum Depression Risk Screener

A cloud-deployed web application that predicts postpartum depression risk using a logistic regression model trained on first-trimester clinical data.

**Live URL:** _(your Vercel URL will go here after deployment)_

---

## Project Overview

This project deploys a machine learning model as a web application on **Vercel** (cloud platform). The frontend collects patient information through a form, sends it to a **Python serverless API**, and displays the predicted risk level.

### Model Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression (balanced class weights) |
| Training Data | N = 691 (with random oversampling for class balance) |
| Target | FPEPDS ≥ 13 (postpartum depression classification) |
| Features | 18 inputs (demographics, EPDS, GAD-7, ACE, SLE, BMI) |
| Threshold | 0.10 (optimized for high recall/sensitivity) |
| ROC-AUC | 0.676 |
| Recall | 0.750 |

---

## Project Structure

```
ppd-predictor/
├── api/
│   └── predict.py              ← Python serverless function (prediction API)
├── public/
│   └── index.html              ← Frontend web application
├── requirements.txt            ← API dependencies (for Vercel deployment)
├── requirements-training.txt   ← Training pipeline dependencies (for local/Colab)
├── vercel.json                 ← Vercel deployment configuration
├── .gitignore                  ← Git ignore rules
└── README.md                   ← This file
```

---

## Prerequisites

- **Python 3.9+** installed on your machine
- **Git** installed ([download](https://git-scm.com/downloads))
- **Node.js 18+** installed ([download](https://nodejs.org/)) — needed for Vercel CLI
- A **GitHub** account ([sign up](https://github.com/))
- A **Vercel** account ([sign up free](https://vercel.com/signup))

---

## Step 1: Set Up Local Environment

### 1a. Clone or download this project

```bash
# If using Git:
git clone https://github.com/YOUR_USERNAME/ppd-predictor.git
cd ppd-predictor

# Or just navigate to the unzipped folder:
cd ppd-predictor
```

### 1b. Create a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate it:
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 1c. Install dependencies

```bash
# For the deployment API (minimal — uses only Python stdlib):
pip install -r requirements.txt

# For model training (if you need to retrain):
pip install -r requirements-training.txt
```

### 1d. Test locally

You can verify the API works locally by running a quick test:

```bash
python -c "
from api.predict import predict
result = predict({
    'Age': 28, 'FEPDS': 12, 'FGAD7': 8,
    'ACE_Score': 3, 'SLE_Score': 2, 'BMI': 22.5,
    'Residence_Type': 'Urban (county-level or above)',
    'Education_Level': 'College / Bachelor',
    'Employment_Status': 'Full-time',
    'Marital_Status': 'Married',
    'Household_Income_After_Tax': '50k–200k',
    'Smoking_Status': 'Never smoked',
    'Secondhand_Smoking': 'Never or almost never',
    'Drinking_Status': 'No drinking',
    'Unplanned_Pregnancy': 0,
    'Pregnancy_Method': 'Natural conception',
    'Number_of_Previous_Children': 0,
    'F_Gestational_Week': 10
})
print(result)
"
```

---

## Step 2: Push to GitHub

```bash
# Initialize Git repository
git init
git add .
git commit -m "Initial commit: PPD risk prediction web app"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/ppd-predictor.git
git branch -M main
git push -u origin main
```

---

## Step 3: Deploy to Vercel

### Option A: Vercel CLI (command line)

```bash
# Install Vercel CLI globally
npm install -g vercel

# Deploy (follow the prompts)
vercel

# For production deployment:
vercel --prod
```

### Option B: GitHub Integration (recommended)

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click **"Import Git Repository"**
3. Select your `ppd-predictor` repository
4. Vercel auto-detects the configuration from `vercel.json`
5. Click **Deploy**
6. Your app will be live at `https://ppd-predictor-XXXXX.vercel.app`

Any future `git push` to the `main` branch will automatically redeploy.

---

## Step 4: Verify Deployment

1. Open your Vercel URL in a browser
2. Test the API health check: visit `https://YOUR-URL.vercel.app/api/predict`
3. Fill in the form and click **"Run Risk Assessment"**
4. Verify the prediction result appears

---

## How It Works

### Architecture

```
┌──────────────┐     POST /api/predict     ┌───────────────────┐
│              │  ───────────────────────►  │                   │
│   Frontend   │     JSON payload          │  Python Serverless │
│  (index.html)│  ◄───────────────────────  │  Function          │
│              │     JSON response         │  (predict.py)      │
└──────────────┘                           └───────────────────┘
     Browser                                   Vercel Cloud
```

1. **Frontend** (`public/index.html`): Collects 18 input features via a web form
2. **API** (`api/predict.py`): Python serverless function that replicates the trained sklearn pipeline
3. **Prediction**: Logistic regression with the same preprocessing (imputation → scaling → one-hot encoding → dot product → sigmoid)

### Why embed model parameters instead of using joblib?

- **No heavy dependencies**: The API uses only Python standard library (`json`, `math`), so cold starts are instant
- **Smaller deployment**: No need to ship `scikit-learn` (~30MB), `numpy` (~20MB), etc.
- **Same results**: The embedded parameters produce identical predictions to the original sklearn pipeline

---

## Disclaimer

This tool is for **research purposes only** (INTD 491). It is not a clinical diagnostic instrument. Predictions are based on a single study cohort and should not replace professional clinical evaluation. Always consult a healthcare provider for diagnosis and treatment of postpartum depression.

---

## Libraries and Dependencies

### Deployment (Vercel serverless function)
- Python 3.9+ standard library only (no external packages)

### Model Training (local/Colab)
- numpy 1.26.4
- pandas 2.2.1
- scikit-learn 1.4.2
- pyreadstat 1.2.7
- matplotlib 3.8.4
- seaborn 0.13.2
- imbalanced-learn 0.12.2
- scipy 1.13.0
- joblib 1.4.0

See `requirements-training.txt` for the full list.
