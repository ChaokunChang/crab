from __future__ import annotations

import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class SimulatedLLMHandler(BaseHTTPRequestHandler):
    rng = random.Random(1337)
    actions = [
        {
            "action_id": "stateless-doc-search",
            "category": "stateless",
            "tool": "doc_search",
            "payload": {"query": "checkpoint policy"},
        },
        {
            "action_id": "stateful-memory-note",
            "category": "stateful",
            "tool": "memory_note",
            "payload": {"note": "remember sandbox session"},
        },
        {
            "action_id": "side-effect-write-artifact",
            "category": "side_effectful",
            "tool": "write_artifact",
            "payload": {"filename": "tool_artifact.txt", "content": "artifact updated"},
        },
    ]

    def do_POST(self) -> None:
        if self.path != "/next_action":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            _ = self.rfile.read(length)
        action = self.rng.choice(self.actions)
        body = json.dumps(action).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self.send_error(404)
            return
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        _ = (format, args)
        return


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SimulatedLLMHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
