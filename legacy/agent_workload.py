import json
import os
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OUT_DIR = os.environ.get("OUT_DIR", "/work")
STATE_PATH = os.path.join(OUT_DIR, "state.json")
LOG_PATH = os.path.join(OUT_DIR, "tick.log")
HOST = os.environ.get("HTTP_HOST", "0.0.0.0")
PORT = int(os.environ.get("HTTP_PORT", "18080"))

os.makedirs(OUT_DIR, exist_ok=True)

def load_state():
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"counter": 0}

def save_state(st):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_PATH)

MEM_MB = int(os.environ.get("MEM_MB", "128"))
blob = bytearray(MEM_MB * 1024 * 1024)
for i in range(0, len(blob), 4096):
    blob[i] = i % 256

st = load_state()
lock = threading.Lock()
runtime_id = str(uuid.uuid4())
started_at = time.time()
http_seq = 0


def snapshot():
    with lock:
        return {
            "runtime_id": runtime_id,
            "pid": os.getpid(),
            "started_at": started_at,
            "counter": st.get("counter", 0),
            "http_seq": http_seq,
            "mem_mb": MEM_MB,
            "port": PORT,
        }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code, payload):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global http_seq
        if self.path not in ("/", "/healthz"):
            self._send_json(404, {"error": "not_found"})
            return
        with lock:
            http_seq += 1
            payload = {
                "runtime_id": runtime_id,
                "pid": os.getpid(),
                "started_at": started_at,
                "counter": st.get("counter", 0),
                "http_seq": http_seq,
                "mem_mb": MEM_MB,
                "port": PORT,
            }
        self._send_json(200, payload)

    def log_message(self, fmt, *args):
        return


def tick_loop():
    while True:
        with lock:
            st["counter"] = st.get("counter", 0) + 1
            idx = random.randint(0, len(blob) - 1)
            blob[idx] = (blob[idx] + 1) % 256
            now = time.time()
            c = st["counter"]

        with open(LOG_PATH, "a") as f:
            f.write(f"{now:.6f} counter={c}\n")
            f.flush()
            os.fsync(f.fileno())

        with lock:
            save_state(st)

        time.sleep(0.05)


thr = threading.Thread(target=tick_loop, daemon=True)
thr.start()

server = ThreadingHTTPServer((HOST, PORT), Handler)
with open(os.path.join(OUT_DIR, "http_boot.json"), "w") as f:
    json.dump(snapshot(), f)
server.serve_forever()
