from http.server import BaseHTTPRequestHandler
import json
import os
import sys

class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        info = {
            "python_version": sys.version,
            "base_dir": os.path.dirname(os.path.abspath(__file__)),
            "files_present": os.listdir(os.path.dirname(os.path.abspath(__file__))),
        }

        # Try importing each package separately
        for pkg in ["numpy", "pandas", "sklearn", "joblib"]:
            try:
                __import__(pkg)
                info[pkg] = "OK"
            except Exception as e:
                info[pkg] = f"FAILED: {e}"

        # Try loading models
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        import joblib
    for fname in ["cls_pipe_FST.pkl", "reg_pipe_FST.pkl"]:
    try:
        path = os.path.join(BASE_DIR, fname)
        model = joblib.load(path)
        info[f"load_{fname}"] = "OK"
    except Exception as e:
        info[f"load_{fname}"] = f"FAILED: {type(e).__name__}: {e}"

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(info, indent=2).encode())

    def log_message(self, format, *args):
        pass
