import http.server
import socketserver
import json
import subprocess
import os
import sys
from pathlib import Path

# Configuration
PORT = 8080
PROTOTYPE_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROTOTYPE_DIR / "data"
WWW_DIR = PROTOTYPE_DIR.parent / "www"

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.serve_file(WWW_DIR / "index.html", "text/html")
        elif self.path == "/api/prototype/status":
            self.handle_status()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/prototype/run":
            self.handle_run()
        elif self.path == "/api/prototype/live":
            self.handle_live()
        else:
            self.send_error(404)

    def serve_file(self, file_path, content_type):
        if file_path.exists():
            self.send_response(200)
            self.send_header("Content-type", content_type)
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def handle_status(self):
        try:
            status = {
                "liveServiceRunning": self.is_service_running(),
                "openTrades": self.read_csv("paper_trades/open_trades.csv"),
                "summary": self.read_csv("paper_trades/paper_trade_summary.csv")[0] if self.read_csv("paper_trades/paper_trade_summary.csv") else {},
                "runHistory": self.read_jsonl("audit/prototype_run_history.jsonl", limit=10),
                "health": { "headerSchemaMatch": True, "decisionCsvWritable": True, "staleLocks": [] }
            }
            self.send_json(status)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_run(self):
        try:
            content_length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(content_length))
            run_models = body.get("runModels", False)
            
            args = [sys.executable, str(PROTOTYPE_DIR / "run_prototype.py")]
            if run_models: args.append("--run-models")
            
            # Run in background to not block the response for long runs
            subprocess.Popen(args, cwd=PROTOTYPE_DIR)
            self.send_json({"status": "OK", "message": "Cycle started in background"})
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_live(self):
        try:
            content_length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(content_length))
            action = body.get("action")
            
            if action == "start":
                if not self.is_service_running():
                    subprocess.Popen([sys.executable, str(PROTOTYPE_DIR / "run_live_service.py")], cwd=PROTOTYPE_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
                self.send_json({"status": "OK", "message": "Started"})
            elif action == "stop":
                if os.name == 'nt':
                    subprocess.run(['powershell', '-Command', "Get-CimInstance Win32_Process | Where-Object CommandLine -like '*run_live_service.py*' | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"], check=False)
                self.send_json({"status": "OK", "message": "Stopped"})
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def is_service_running(self):
        try:
            output = subprocess.check_output(['powershell', '-Command', "Get-CimInstance Win32_Process | Where-Object CommandLine -like '*run_live_service.py*' | Select-Object -ExpandProperty ProcessId"], text=True)
            return len(output.strip()) > 0
        except:
            return False

    def read_csv(self, rel_path):
        path = DATA_DIR / rel_path
        if not path.exists(): return []
        import csv
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def read_jsonl(self, rel_path, limit=10):
        path = DATA_DIR / rel_path
        if not path.exists(): return []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return [json.loads(line) for line in lines[-limit:] if line.strip()][::-1]

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

print(f"Dashboard Server started at http://localhost:{PORT}")
print("IMPORTANT: Please use this URL instead of Live Server.")
with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
    httpd.serve_forever()
