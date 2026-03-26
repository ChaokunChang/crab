from .analyzer import TelemetryAnalysis, analyze_telemetry_file

__all__ = [
    "TelemetryAnalysis",
    "analyze_telemetry_file",
    "generate_report_bundle",
]


def generate_report_bundle(*args, **kwargs):
    from .report import generate_report_bundle as _generate_report_bundle

    return _generate_report_bundle(*args, **kwargs)
