# syntax=docker/dockerfile:1.7
# =============================================================================
# Multi-stage build:
#   builder — installs the package into an isolated virtualenv
#   runtime — copies only the virtualenv, runs as a non-root user
#
# Only the extras actually needed at runtime are installed. Embeddings
# (ONNX, sentence-transformers) will be added when the retrieval path lands
# in M2/M3.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /build

# Build into a virtualenv we can copy into the runtime stage verbatim.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only what the build needs. Hatchling validates `readme` and
# `license = { file = ... }` from pyproject.toml, so both README.md and
# LICENSE must be present at build time. src/ holds the package itself.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip && \
    pip install ".[api,observability]"

# -----------------------------------------------------------------------------
# Stage 2: runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# Non-root user with a stable UID so bind mounts behave predictably. uid 1000
# is the conventional first non-system user; the --system flag is omitted
# because it warns when the uid is above SYS_UID_MAX.
RUN groupadd --gid 1000 recsys && \
    useradd --uid 1000 --gid recsys \
            --create-home --shell /usr/sbin/nologin recsys

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=prod

# Copy the prebuilt virtualenv. No compiler, no source tree, no pip cache.
COPY --from=builder /opt/venv /opt/venv

USER recsys
WORKDIR /home/recsys

EXPOSE 8000

# Liveness-only healthcheck. Readiness (which will check the DB and index from
# M3) is a separate concern and is exercised by the orchestrator, not Docker.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import sys,urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["python", "-m", "recsys"]
