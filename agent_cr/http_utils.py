from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import http.client
from http.server import HTTPServer
from threading import Lock, local
from typing import Any
from urllib.parse import urlsplit

from .json_codec import JsonCodec, get_json_codec


DEFAULT_HTTP_SERVER_MAX_WORKERS = 32


class HttpStatusError(RuntimeError):
    def __init__(self, *, status_code: int, path: str, body: bytes) -> None:
        self.status_code = int(status_code)
        self.path = path
        self.body = body
        message = f"http request failed status={self.status_code} path={self.path}"
        super().__init__(message)


class PooledHTTPServer(HTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        *,
        max_workers: int | None = None,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        workers = DEFAULT_HTTP_SERVER_MAX_WORKERS if max_workers is None else max(1, int(max_workers))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent-cr-http")

    def process_request(self, request, client_address) -> None:
        self._executor.submit(self._process_request_worker, request, client_address)

    def _process_request_worker(self, request, client_address) -> None:
        try:
            self.finish_request(request, client_address)
            self.shutdown_request(request)
        except Exception:
            self.handle_error(request, client_address)
            self.shutdown_request(request)

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)


class ThreadLocalHttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        serializer: str = "auto",
    ) -> None:
        parsed = urlsplit(base_url.rstrip("/"))
        if parsed.scheme != "http":
            raise ValueError(f"only http base URLs are supported, got {base_url!r}")
        self._host = parsed.hostname or "127.0.0.1"
        self._port = int(parsed.port or 80)
        self._base_path = parsed.path.rstrip("/")
        self._timeout_seconds = float(timeout_seconds)
        self._codec = get_json_codec(serializer)
        self._local = local()
        self._connections_lock = Lock()
        self._connections: set[http.client.HTTPConnection] = set()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        normalized_path = self._request_path(path)
        last_exc: Exception | None = None
        for _ in range(2):
            connection = self._connection()
            try:
                connection.request(method, normalized_path, body=body, headers=headers or {})
                response = connection.getresponse()
                try:
                    payload = response.read()
                    response_headers = [(str(key), str(value)) for key, value in response.getheaders()]
                    should_close = str(response.getheader("Connection", "")).strip().lower() == "close"
                    status_code = int(response.status)
                finally:
                    response.close()
                if should_close:
                    self._reset_thread_connection()
                return status_code, response_headers, payload
            except (BrokenPipeError, ConnectionResetError, http.client.HTTPException, OSError) as exc:
                last_exc = exc
                self._reset_thread_connection()
        assert last_exc is not None
        raise last_exc

    def get_json(self, path: str) -> dict[str, Any]:
        status_code, _, payload = self.request("GET", path)
        if status_code >= 400:
            raise HttpStatusError(status_code=status_code, path=path, body=payload)
        result = self._codec.loads(payload)
        if not isinstance(result, dict):
            raise ValueError(f"expected JSON object response for GET {path}, got {type(result).__name__}")
        return dict(result)

    def post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        merged_headers = {"Content-Type": "application/json"}
        if headers:
            merged_headers.update(headers)
        status_code, _, body = self.request(
            "POST",
            path,
            body=self._codec.dumps_bytes(payload),
            headers=merged_headers,
        )
        if status_code >= 400:
            raise HttpStatusError(status_code=status_code, path=path, body=body)
        result = self._codec.loads(body)
        if not isinstance(result, dict):
            raise ValueError(f"expected JSON object response for POST {path}, got {type(result).__name__}")
        return dict(result)

    def close(self) -> None:
        with self._connections_lock:
            connections = list(self._connections)
            self._connections.clear()
        for connection in connections:
            try:
                connection.close()
            except Exception:
                continue
        self._local.connection = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return

    def _connection(self) -> http.client.HTTPConnection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = http.client.HTTPConnection(
                self._host,
                self._port,
                timeout=self._timeout_seconds,
            )
            with self._connections_lock:
                self._connections.add(connection)
            self._local.connection = connection
        return connection

    def _reset_thread_connection(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            return
        with self._connections_lock:
            self._connections.discard(connection)
        try:
            connection.close()
        except Exception:
            pass
        self._local.connection = None

    def _request_path(self, path: str) -> str:
        suffix = path if path.startswith("/") else f"/{path}"
        prefix = self._base_path or ""
        return f"{prefix}{suffix}"
