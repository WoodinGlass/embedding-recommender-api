# Tests

Three tiers, separated by directory. The tier marker is applied automatically
by `tests/conftest.py` — you do not need to decorate every test by hand.

| Directory | Marker | Runs when | Budget |
|---|---|---|---|
| `tests/unit/` | `unit` | Always | Milliseconds |
| `tests/integration/` | `integration` | Env vars set (see below) | Seconds |
| `tests/load/` | `load` | Manual / M4 | Minutes |

## Running

```bash
make test            # unit + integration (integration skips if env not set)
make test-unit       # unit only
make test-integration
make coverage
```

## Enabling integration tests

Integration tests skip cleanly unless the env vars below are set. In CI they
are provided by service containers (PostgreSQL with pgvector, Redis).

| Env var | Used by |
|---|---|
| `RECSYS_TEST_DATABASE_URL` | PostgreSQL fixtures |
| `RECSYS_TEST_REDIS_URL` | Redis fixtures |

## Rules

1. A unit test must never import from a module that talks to a network,
   database, or filesystem outside `tmp_path`.
2. If a test needs a real service, it lives under `tests/integration/`.
3. `make check-markers` (also run in CI) fails the build if a unit test
   references an integration or load marker.
