from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import joblib


import sklearn
class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        info = {
            "python_version": sys.version,
            "base_dir": BASE_DIR,
            "files_present": os.listdir(BASE_DIR),
        }
        
        info["sklearn_version_runtime"] = sklearn.__version__

        # Try importing each package separately
        for pkg in ["numpy", "pandas", "sklearn", "joblib"]:
            try:
                __import__(pkg)
                info[pkg] = "OK"
            except Exception as e:
                info[pkg] = f"FAILED: {e}"

        # Try actually loading the models
        for fname in ["cls_pipe_FST.pkl", "reg_pipe_FST.pkl"]:
            try:
                path = os.path.join(BASE_DIR, fname)
                model = joblib.load(path)
                info[f"load_{fname}"] = "OK"
            except Exception as e:
                info[f"load_{fname}"] = f"FAILED: {type(e).__name__}: {e}"

        # Try loading model_config.json
        try:
            with open(os.path.join(BASE_DIR, "model_config.json")) as f:
                config = json.load(f)
            info["load_model_config.json"] = f"OK — threshold={config.get('threshold')}"
        except Exception as e:
            info["load_model_config.json"] = f"FAILED: {type(e).__name__}: {e}"

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(info, indent=2).encode())

    def log_message(self, format, *args):
        pass
