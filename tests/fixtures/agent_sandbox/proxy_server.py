from __future__ import annotations

import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ProxyHandler(BaseHTTPRequestHandler):
    llm_url = ""

    def do_POST(self) -> None:
        if self.path != "/next_action":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length) if length else b"{}"
        req = urllib.request.Request(
            self.llm_url + "/next_action",
            data=payload,
            headers={"Content-Type": "application/json", "X-Agent-Sandbox": "true"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            body = resp.read()
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
    parser.add_argument("--llm-url", required=True)
    args = parser.parse_args()

    ProxyHandler.llm_url = args.llm_url.rstrip("/")
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
