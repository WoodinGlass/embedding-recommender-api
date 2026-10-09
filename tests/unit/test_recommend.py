"""Unit tests for the recommend handler (M3.6.6b.2).

Two classes:

- ``TestHandlerWiring`` mocks ``sync_pipeline`` and asserts the
  handler maps a result to the response schema and a failure to the
  HTTP status the reason implies. No DB, no encoder, no thread.
- ``TestHandlerWithFakeCollaborators`` keeps the real handler and
  the real ``sync_pipeline`` but replaces the pool, the encoder,
  and the backend the pipeline builds with fakes. This is what
  catches a signature drift between the handler and the pipeline
  that a mocked pipeline would hide.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recsys.api import pipeline as pipeline_module
from recsys.api.app import create_app
from recsys.api.pipeline import PipelineFailure, PipelineResult
from recsys.config.enums import AppEnv
from recsys.config.settings import Settings
from recsys.fallback import FallbackResult

#: The API key every TestClient in this module sends.
_TEST_API_KEY = "test-api-key"


# ---------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------- #
def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "app_env": AppEnv.DEV,
        "request_timeout_seconds": 1.0,
        # The router requires a credential; the clients below send
        # this key on every request.
        "api_keys": _TEST_API_KEY,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class _FakeCursor:
    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        self._rows = rows

    def execute(self, sql: str, params: Any = None) -> None:
        pass

    def fetchall(self) -> list[tuple[str, str, str]]:
        return list(self._rows)

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeConn:
    def __init__(self, rows: list[tuple[str, str, str]] | None = None) -> None:
        # ``rows=[]`` is a legitimate value ("catalog has no such
        # seed"); ``or`` would turn it into the default and hide the
        # failure the test is written to catch.
        self._rows = [("i_seed", "t", "d")] if rows is None else rows

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._rows)


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    @contextlib.contextmanager
    def connection(self) -> Iterator[_FakeConn]:
        yield self._conn


class _FakeEncoder:
    model_version = "test-model+abc12345"

    def encode(self, texts: Any) -> Any:
        return np.ones((len(texts), 4), dtype=np.float32)


class _FakeBackend:
    active_index_version = "idx-tests0001"

    def search(self, *, vector: Any, k: int, filters: Any = None) -> Any:
        return [("i_a", 0.9), ("i_b", 0.5)][:k]

    def search_with_vectors(self, **kwargs: Any) -> Any:
        raise AssertionError("MMR is not enabled in these tests")


# ---------------------------------------------------------------- #
# wiring
# ---------------------------------------------------------------- #
class TestHandlerWiring:
    """The handler maps a result to a response, a client error to a
    4xx, and a dependency error to a 5xx."""

    def _client_with_mock(self, monkeypatch: pytest.MonkeyPatch, result: Any) -> TestClient:
        def _fake(*args: Any, **kwargs: Any) -> Any:
            return result

        monkeypatch.setattr(pipeline_module, "sync_pipeline", _fake)
        # The router imports `sync_pipeline` by name; patch there too.
        from recsys.api.routers import recommend as rec_router

        monkeypatch.setattr(rec_router, "sync_pipeline", _fake)

        # The app factory stores the pool and the encoder in
        # closure variables and assigns them to app.state inside
        # the lifespan. Setting app.state after create_app does
        # not survive that: the lifespan overwrites it by design.
        # Patch the *builders* instead — they are called once,
        # before the lifespan runs.
        import recsys.api.app as app_module

        monkeypatch.setattr(app_module, "_build_db_pool", lambda _settings: _FakePool(_FakeConn()))
        monkeypatch.setattr(app_module, "_build_encoder", lambda _settings, _log: _FakeEncoder())

        # Default: the fallback chain produces nothing, so a
        # PipelineFailure still maps to 503. Tests that want a
        # fallback response override _run_fallback (see
        # TestHandlerFallback).
        from recsys.api.routers import recommend as rec_router

        async def _no_fallback(**_kwargs: Any) -> None:
            return None

        monkeypatch.setattr(rec_router, "_run_fallback", _no_fallback)

        app = create_app(_settings())
        # Every request authenticates; the test bodies do not spell
        # out the header because the endpoint under test is not the
        # auth dependency (see tests/unit/test_auth_dependency.py).
        return TestClient(app, headers={"X-API-Key": _TEST_API_KEY})

    def test_happy_path_returns_200_with_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._client_with_mock(
            monkeypatch,
            PipelineResult(
                items=(("i_a", 0.9), ("i_b", 0.5)),
                index_version="idx-tests0001",
                model_version="test-model+abc12345",
            ),
        )
        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": ["i_seed"], "k": 2},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["meta"]["source"] == "ann"
        assert body["meta"]["index_version"] == "idx-tests0001"
        assert body["meta"]["model_version"] == "test-model+abc12345"
        assert [it["item_id"] for it in body["items"]] == ["i_a", "i_b"]
        assert [it["rank"] for it in body["items"]] == [1, 2]

    def test_empty_seeds_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._client_with_mock(monkeypatch, PipelineFailure("empty_seeds"))
        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": [], "k": 5},
            )
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "bad_request"

    def test_all_seeds_missing_returns_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._client_with_mock(monkeypatch, PipelineFailure("all_seeds_missing"))
        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": ["i_x"], "k": 5},
            )
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "not_found"

    @pytest.mark.parametrize(
        "reason",
        ["encoder_unavailable", "no_active_index", "ann_error"],
    )
    def test_dependency_failure_returns_503(
        self, monkeypatch: pytest.MonkeyPatch, reason: str
    ) -> None:
        client = self._client_with_mock(
            monkeypatch,
            PipelineFailure(reason=reason),  # type: ignore[arg-type]
        )
        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": ["i_seed"], "k": 5},
            )
        assert r.status_code == 503
        assert r.json()["detail"]["code"] == "unavailable"

    def test_empty_retrieval_returns_200_with_no_items(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = self._client_with_mock(
            monkeypatch,
            PipelineResult(
                items=(),
                index_version="idx-tests0001",
                model_version="test-model+abc12345",
            ),
        )
        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": ["i_seed"], "k": 5},
            )
        assert r.status_code == 200
        assert r.json()["items"] == []


# ---------------------------------------------------------------- #
# end-to-end with a real pipeline
# ---------------------------------------------------------------- #
class TestHandlerWithFakeCollaborators:
    """The handler runs the real ``sync_pipeline``; the pool, the
    encoder, and the backend the pipeline builds are fakes."""

    def _client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        seed_rows: list[tuple[str, str, str]] | None = None,
    ) -> TestClient:
        # Patch the backend factory the pipeline imports lazily.
        from recsys.retrieval import pgvector as pgvector_module

        def _fake_from_registry(
            cls: Any,
            connection: Any,
            *,
            hnsw_ef_search: int,
            max_scan_tuples: Any = None,
        ) -> Any:
            return _FakeBackend()

        monkeypatch.setattr(
            pgvector_module.PgvectorBackend,
            "from_registry",
            classmethod(_fake_from_registry),
        )

        # See the note in TestHandlerWiring: the builder is the
        # right seam, app.state is not.
        import recsys.api.app as app_module

        monkeypatch.setattr(
            app_module,
            "_build_db_pool",
            lambda _settings: _FakePool(_FakeConn(rows=seed_rows)),
        )
        monkeypatch.setattr(app_module, "_build_encoder", lambda _settings, _log: _FakeEncoder())

        # Explicit: the fallback chain is not what these tests
        # exercise. Keeping it deterministic (return None) means
        # these tests do not depend on whether the popularity
        # snapshot file is present on disk.
        from recsys.api.routers import recommend as rec_router

        async def _no_fallback(**_kwargs: Any) -> None:
            return None

        monkeypatch.setattr(rec_router, "_run_fallback", _no_fallback)

        app = create_app(_settings())
        return TestClient(app, headers={"X-API-Key": _TEST_API_KEY})

    def test_happy_path_uses_real_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._client(monkeypatch)
        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": ["i_seed"], "k": 2},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["meta"]["source"] == "ann"
        assert [it["item_id"] for it in body["items"]] == ["i_a", "i_b"]

    def test_empty_seeds_short_circuits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._client(monkeypatch)
        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": [], "k": 2},
            )
        assert r.status_code == 400

    def test_all_seeds_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._client(monkeypatch, seed_rows=[])
        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": ["i_gone"], "k": 2},
            )
        assert r.status_code == 404


# ---------------------------------------------------------------- #
# RecommendMeta accepts null versions (fallback path)
# ---------------------------------------------------------------- #
class TestRecommendMetaSchema:
    """The schema allows ``None`` for model_version and index_version.

    The ANN path always fills them; the fallback path (ADR-0020)
    does not use a model or an index, and the field is nullable
    rather than a magic string. These tests pin the contract so a
    future change that tightens it back to ``str`` fails loudly.
    """

    def test_meta_accepts_null_versions(self) -> None:
        from recsys.api.schemas.recommend import RecommendMeta

        meta = RecommendMeta(
            source="fallback_ann",
            model_version=None,
            index_version=None,
        )
        assert meta.model_version is None
        assert meta.index_version is None

    def test_meta_accepts_filled_versions(self) -> None:
        from recsys.api.schemas.recommend import RecommendMeta

        meta = RecommendMeta(
            source="ann",
            model_version="minilm-onnx-v1+a3f9e021",
            index_version="idx-a3f9e021",
        )
        assert meta.model_version is not None
        assert meta.index_version is not None

    def test_meta_rejects_missing_version_fields(self) -> None:
        # Both fields are required (nullable, but not optional): a
        # caller must say "no version" explicitly rather than omit
        # the field and let the default silently decide.
        import pytest
        from pydantic import ValidationError

        from recsys.api.schemas.recommend import RecommendMeta

        with pytest.raises(ValidationError):
            RecommendMeta(source="ann")  # type: ignore[call-arg]


# ---------------------------------------------------------------- #
# fallback chain (ADR-0020 tiers 3 and 4)
# ---------------------------------------------------------------- #
class TestHandlerFallback:
    """The handler maps a non-None _run_fallback to 200 with the
    tier's source and null versions, and maps None to 503."""

    def _client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        pipeline_result: PipelineResult | PipelineFailure,
        fallback_result: FallbackResult | None,
    ) -> TestClient:
        from recsys.api import pipeline as pipeline_module
        from recsys.api.routers import recommend as rec_router

        def _fake_pipeline(*_args: Any, **_kwargs: Any) -> Any:
            return pipeline_result

        monkeypatch.setattr(pipeline_module, "sync_pipeline", _fake_pipeline)
        monkeypatch.setattr(rec_router, "sync_pipeline", _fake_pipeline)

        async def _fake_fallback(**_kwargs: Any) -> FallbackResult | None:
            return fallback_result

        monkeypatch.setattr(rec_router, "_run_fallback", _fake_fallback)

        import recsys.api.app as app_module

        monkeypatch.setattr(app_module, "_build_db_pool", lambda _settings: _FakePool(_FakeConn()))
        monkeypatch.setattr(app_module, "_build_encoder", lambda _settings, _log: _FakeEncoder())
        app = create_app(_settings())
        return TestClient(app, headers={"X-API-Key": _TEST_API_KEY})

    def test_fallback_ann_returns_200_with_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._client(
            monkeypatch,
            pipeline_result=PipelineFailure("ann_error"),
            fallback_result=FallbackResult(
                source="fallback_ann",
                items=(("i_pop1", 0.9), ("i_pop2", 0.5)),
            ),
        )
        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": ["i_seed"], "k": 2},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["meta"]["source"] == "fallback_ann"
        assert body["meta"]["model_version"] is None
        assert body["meta"]["index_version"] is None
        assert [it["item_id"] for it in body["items"]] == ["i_pop1", "i_pop2"]
        assert [it["rank"] for it in body["items"]] == [1, 2]

    def test_fallback_cached_returns_200_with_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._client(
            monkeypatch,
            pipeline_result=PipelineFailure("ann_error"),
            fallback_result=FallbackResult(
                source="fallback_cached",
                items=(("i_mem1", 0.8),),
            ),
        )
        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": ["i_seed"], "k": 5},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["meta"]["source"] == "fallback_cached"
        assert body["meta"]["model_version"] is None
        assert body["meta"]["index_version"] is None

    def test_both_tiers_empty_returns_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._client(
            monkeypatch,
            pipeline_result=PipelineFailure("ann_error"),
            fallback_result=None,
        )
        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": ["i_seed"], "k": 5},
            )
        assert r.status_code == 503
        assert r.json()["detail"]["code"] == "unavailable"

    def test_client_error_does_not_trigger_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 400/404 must not be turned into a popular list."""
        from recsys.api.routers import recommend as rec_router

        called: list[bool] = []

        async def _should_not_be_called(**_kwargs: Any) -> FallbackResult | None:
            called.append(True)
            return None

        # Override the default set inside _client by patching before
        # the fixture runs. Rebuild the client with the pipeline
        # returning a client error.
        client = self._client(
            monkeypatch,
            pipeline_result=PipelineFailure("empty_seeds"),
            fallback_result=None,
        )
        # The helper above set _run_fallback; replace it with one
        # that records whether it is called.
        monkeypatch.setattr(rec_router, "_run_fallback", _should_not_be_called)

        with client:
            r = client.post(
                "/v1/recommend",
                json={"user_id": "u_1", "seed_item_ids": [], "k": 5},
            )
        assert r.status_code == 400
        assert called == [], "fallback must not run for a client error"
