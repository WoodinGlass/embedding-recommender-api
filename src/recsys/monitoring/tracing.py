"""OpenTelemetry tracing.

Full OTEL setup lands in M4 (Observability). Until then this module is a
no-op that keeps the interface in place, so routers and the app factory do
not need to change when instrumentation is added.
"""

from __future__ import annotations

from recsys.config.settings import Settings
from recsys.monitoring.logging import get_logger

log = get_logger(__name__)


def configure_tracing(settings: Settings) -> None:
    """No-op until M4. See README milestones."""
    log.info(
        "tracing.skipped",
        reason="implementation_lands_in_M4",
        endpoint=settings.otel_exporter_otlp_endpoint,
    )
