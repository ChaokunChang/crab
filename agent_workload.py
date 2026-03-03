import os, time, json, random

OUT_DIR = os.environ.get("OUT_DIR", "/work")
STATE_PATH = os.path.join(OUT_DIR, "state.json")
LOG_PATH = os.path.join(OUT_DIR, "tick.log")

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

# allocate some memory to look like an agent runtime
MEM_MB = int(os.environ.get("MEM_MB", "128"))
blob = bytearray(MEM_MB * 1024 * 1024)
for i in range(0, len(blob), 4096):
    blob[i] = i % 256

st = load_state()

while True:
    st["counter"] += 1

    # simulate “thinking” + some random access
    idx = random.randint(0, len(blob)-1)
    blob[idx] = (blob[idx] + 1) % 256

    # simulate tool output / scratchpad writes
    with open(LOG_PATH, "a") as f:
        f.write(f"{time.time():.6f} counter={st['counter']}\n")
        f.flush()
        os.fsync(f.fileno())

    save_state(st)

    # keep it stable and not too hot
    time.sleep(0.05)