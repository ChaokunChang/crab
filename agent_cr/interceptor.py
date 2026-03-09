from __future__ import annotations

from .contracts import RequestInterceptorHook, TelemetrySink
from .models import RequestContext


class CompositeRequestInterceptorHook(RequestInterceptorHook):
    def __init__(self, hooks: list[RequestInterceptorHook] | None = None):
        self._hooks = list(hooks or [])

    def add_hook(self, hook: RequestInterceptorHook) -> None:
        self._hooks.append(hook)

    def on_request_start(self, context: RequestContext) -> None:
        for hook in self._hooks:
            hook.on_request_start(context)

    def on_request_end(self, context: RequestContext) -> None:
        for hook in self._hooks:
            hook.on_request_end(context)


class TelemetryRequestInterceptorHook(RequestInterceptorHook):
    def __init__(self, telemetry: TelemetrySink):
        self._telemetry = telemetry

    def on_request_start(self, context: RequestContext) -> None:
        self._telemetry.emit_event(
            "request.start",
            {
                "request_id": context.request_id,
                "sandbox_id": str(context.sandbox_id),
            },
        )

    def on_request_end(self, context: RequestContext) -> None:
        self._telemetry.emit_event(
            "request.end",
            {
                "request_id": context.request_id,
                "sandbox_id": str(context.sandbox_id),
            },
        )
