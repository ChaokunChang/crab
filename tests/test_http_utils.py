from __future__ import annotations

import socket
import threading
import unittest

from agent_cr.http_utils import ThreadLocalHttpClient


class ThreadLocalHttpClientTests(unittest.TestCase):
    def test_close_interrupts_in_flight_response_read(self) -> None:
        server_ready = threading.Event()
        request_received = threading.Event()
        release_server = threading.Event()
        port_holder: list[int] = []
        server_errors: list[BaseException] = []

        def serve_one_stalled_response() -> None:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port_holder.append(int(listener.getsockname()[1]))
            server_ready.set()
            try:
                connection, _ = listener.accept()
            except BaseException as exc:
                server_errors.append(exc)
                listener.close()
                return
            listener.close()
            with connection:
                connection.settimeout(5.0)
                try:
                    payload = b""
                    while b"\r\n\r\n" not in payload:
                        chunk = connection.recv(4096)
                        if not chunk:
                            return
                        payload += chunk
                    header_blob, body = payload.split(b"\r\n\r\n", 1)
                    content_length = 0
                    for line in header_blob.split(b"\r\n"):
                        if line.lower().startswith(b"content-length:"):
                            content_length = int(line.split(b":", 1)[1].strip())
                            break
                    while len(body) < content_length:
                        chunk = connection.recv(4096)
                        if not chunk:
                            return
                        body += chunk
                    request_received.set()
                    release_server.wait(timeout=5.0)
                except BaseException as exc:
                    server_errors.append(exc)

        server_thread = threading.Thread(target=serve_one_stalled_response)
        server_thread.start()
        self.assertTrue(server_ready.wait(timeout=2.0))
        self.assertTrue(port_holder)

        client = ThreadLocalHttpClient(f"http://127.0.0.1:{port_holder[0]}", timeout_seconds=30.0)
        client_errors: list[BaseException] = []

        def issue_request() -> None:
            try:
                client.post_json("/stall", {"ok": True})
            except BaseException as exc:
                client_errors.append(exc)

        client_thread = threading.Thread(target=issue_request)
        client_thread.start()
        self.assertTrue(request_received.wait(timeout=2.0))

        client.close()
        client_thread.join(timeout=2.0)
        interrupted = not client_thread.is_alive()
        release_server.set()
        client_thread.join(timeout=5.0)
        server_thread.join(timeout=5.0)

        self.assertTrue(interrupted, "close() did not interrupt the blocked response read")
        self.assertTrue(client_errors)
        self.assertFalse(server_errors)


if __name__ == "__main__":
    unittest.main()
