"""Unit tests for the embedding artifact primitives.

These tests exercise the paths, locking, atomic writes, config hashing,
and run-id allocation that ``pipeline.py`` relies on. They do not touch an
encoder; pyarrow is used directly for the Parquet tests (installed via the
``[dev-lite]`` extra).
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest
from numpy.typing import NDArray

from recsys.embeddings import artifacts as art

pytestmark = pytest.mark.unit


# ============================================================================
# helpers
# ============================================================================
def _root(tmp_path: pathlib.Path) -> art.EmbeddingsRoot:
    r = art.EmbeddingsRoot(path=tmp_path / "embeddings")
    r.path.mkdir(parents=True, exist_ok=True)
    return r


def _fake_embedding(dim: int, seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _write_parquet(
    path: pathlib.Path,
    *,
    ids: list[str],
    dim: int = 4,
    version: str = "v1",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with art.ParquetBatchWriter(path, embedding_dim=dim) as w:
        w.write_batch(
            item_ids=ids,
            embeddings=np.stack([_fake_embedding(dim, i) for i in range(len(ids))]),
            content_hashes=[f"h{i:016x}" for i in range(len(ids))],
            preprocessing_version=version,
        )


# ============================================================================
# EmbeddingsRoot / RunPaths
# ============================================================================
def test_embeddings_root_layout(tmp_path: pathlib.Path) -> None:
    r = _root(tmp_path)
    assert r.runs_dir == r.path / "runs"
    assert r.current_file == r.path / "current"
    assert r.lock_file == r.path / ".lock"


def test_run_paths_shape(tmp_path: pathlib.Path) -> None:
    r = _root(tmp_path)
    rp = art.RunPaths(root=r, run_id="2026-10-07T11-30-00Z__m+v1")
    assert rp.dir == r.runs_dir / "2026-10-07T11-30-00Z__m+v1"
    assert rp.parquet.name == "embeddings.parquet"
    assert rp.manifest.name == "manifest.json"
    assert rp.state.name == "state.json"


@pytest.mark.parametrize(
    "bad",
    ["", "..", "../escape", "a/b", "a\\b", ".hidden", "./x"],
)
def test_run_paths_rejects_unsafe_run_id(tmp_path: pathlib.Path, bad: str) -> None:
    r = _root(tmp_path)
    with pytest.raises(ValueError, match="unsafe run_id"):
        art.RunPaths(root=r, run_id=bad)


# ============================================================================
# file_lock
# ============================================================================
def test_file_lock_acquire_and_release(tmp_path: pathlib.Path) -> None:
    lock = tmp_path / "l"
    with art.file_lock(lock, timeout=1.0):
        pass
    # Immediately reacquirable.
    with art.file_lock(lock, timeout=1.0):
        pass


def test_file_lock_serialises_threads(tmp_path: pathlib.Path) -> None:
    lock = tmp_path / "l"
    events: list[str] = []
    started = threading.Event()
    release_first = threading.Event()

    def first() -> None:
        with art.file_lock(lock, timeout=5.0):
            events.append("first-enter")
            started.set()
            release_first.wait(timeout=5.0)
            events.append("first-exit")

    def second() -> None:
        started.wait(timeout=5.0)
        with art.file_lock(lock, timeout=5.0):
            events.append("second-enter")

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    # Let the first thread acquire, then give the second a moment to block.
    time.sleep(0.3)
    # At this point, the second thread must not have entered.
    assert "second-enter" not in events
    release_first.set()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert events == ["first-enter", "first-exit", "second-enter"]


def test_file_lock_times_out(tmp_path: pathlib.Path) -> None:
    lock = tmp_path / "l"
    first = art.file_lock(lock, timeout=1.0)
    first.__enter__()
    try:
        with (
            pytest.raises(art.LockTimeoutError, match="could not acquire"),
            art.file_lock(lock, timeout=0.3),
        ):
            pass
    finally:
        first.__exit__(None, None, None)


def test_file_lock_writes_diagnostics(tmp_path: pathlib.Path) -> None:
    lock = tmp_path / "l"
    with art.file_lock(lock, timeout=1.0):
        payload = json.loads(lock.read_text(encoding="utf-8"))
        assert "pid" in payload
        assert "host" in payload
        assert payload["acquired_at"].endswith("Z")


# ============================================================================
# atomic writes
# ============================================================================
def test_atomic_write_text_creates_file(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "sub" / "file.txt"
    art.atomic_write_text(p, "hello\n")
    assert p.read_text(encoding="utf-8") == "hello\n"


def test_atomic_write_text_overwrites(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "f.txt"
    art.atomic_write_text(p, "old")
    art.atomic_write_text(p, "new")
    assert p.read_text(encoding="utf-8") == "new"


def test_atomic_write_json_is_canonical(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "d.json"
    art.atomic_write_json(p, {"b": 2, "a": 1})
    text = p.read_text(encoding="utf-8")
    # sorted keys, two-space indent, trailing newline
    assert text == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_atomic_write_leaves_no_temp_on_success(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "x.txt"
    art.atomic_write_text(p, "a")
    leftovers = [f for f in tmp_path.iterdir() if f.name.startswith(".tmp-")]
    assert leftovers == []


# ============================================================================
# config_hash
# ============================================================================
# Minimal set of content-affecting parameters. Each test below calls
# `art.config_hash` with explicit keyword arguments so mypy can check each
# field against the signature. Unpacking a `dict[str, object]` would defeat
# that check.
def _base_config() -> dict[str, str | int | bool]:
    return {
        "preprocessing_version": "v1",
        "onnx_artifact_sha256": "a" * 64,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "max_seq_length": 256,
        "embedding_dim": 384,
    }


def _h(**over: str | int | bool) -> str:
    base = _base_config()
    base.update(over)
    return art.config_hash(**base)  # type: ignore[arg-type]


def test_config_hash_format_and_determinism() -> None:
    h1 = art.config_hash(
        preprocessing_version="v1",
        onnx_artifact_sha256="a" * 64,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length=256,
        embedding_dim=384,
    )
    h2 = art.config_hash(
        preprocessing_version="v1",
        onnx_artifact_sha256="a" * 64,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length=256,
        embedding_dim=384,
    )
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert len(h1) == len("sha256:") + 64


@pytest.mark.parametrize(
    ("key", "new"),
    [
        ("preprocessing_version", "v2"),
        ("onnx_artifact_sha256", "b" * 64),
        ("model_name", "other/model"),
        ("max_seq_length", 512),
        ("embedding_dim", 768),
        ("pooling", "cls"),
        ("normalize", False),
    ],
)
def test_config_hash_is_sensitive_to_content_params(key: str, new: str | int | bool) -> None:
    base = _h()
    changed = _h(**{key: new})
    assert changed != base, f"config_hash did not change when {key} changed"


def test_config_hash_is_canonical_regardless_of_argument_order() -> None:
    # Same keyword arguments, written in two different textual orders.
    h1 = art.config_hash(
        preprocessing_version="v1",
        onnx_artifact_sha256="a" * 64,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length=256,
        embedding_dim=384,
    )
    h2 = art.config_hash(
        embedding_dim=384,
        max_seq_length=256,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        onnx_artifact_sha256="a" * 64,
        preprocessing_version="v1",
    )
    assert h1 == h2


# ============================================================================
# make_run_id
# ============================================================================
def test_make_run_id_format(tmp_path: pathlib.Path) -> None:
    r = _root(tmp_path)
    rid = art.make_run_id(
        r,
        when=datetime(2026, 10, 7, 11, 30, 0, tzinfo=UTC),
        model_version="m+v1",
    )
    assert rid == "2026-10-07T11-30-00Z__m+v1"


def test_make_run_id_collision_suffix(tmp_path: pathlib.Path) -> None:
    r = _root(tmp_path)
    when = datetime(2026, 10, 7, 11, 30, 0, tzinfo=UTC)
    first = art.make_run_id(r, when=when, model_version="m+v1")
    (r.runs_dir / first).mkdir(parents=True)
    second = art.make_run_id(r, when=when, model_version="m+v1")
    assert second == "2026-10-07T11-30-00Z-01__m+v1"


def test_make_run_id_rejects_naive_datetime(tmp_path: pathlib.Path) -> None:
    r = _root(tmp_path)
    with pytest.raises(ValueError, match="timezone-aware"):
        art.make_run_id(r, when=datetime(2026, 10, 7, 11, 30), model_version="m")


def test_make_run_id_accepts_non_utc_timezone(tmp_path: pathlib.Path) -> None:
    r = _root(tmp_path)
    tz = timezone(timedelta(hours=7))
    rid = art.make_run_id(
        r,
        when=datetime(2026, 10, 7, 18, 30, 0, tzinfo=tz),  # 11:30 UTC
        model_version="m+v1",
    )
    assert rid == "2026-10-07T11-30-00Z__m+v1"


# ============================================================================
# current / manifest / resolve_current
# ============================================================================
def test_read_current_returns_none_when_missing(tmp_path: pathlib.Path) -> None:
    r = _root(tmp_path)
    assert art.read_current(r) is None


def test_read_current_returns_run_id(tmp_path: pathlib.Path) -> None:
    r = _root(tmp_path)
    art.atomic_write_text(r.current_file, "2026-10-07T11-30-00Z__m+v1\n")
    assert art.read_current(r) == "2026-10-07T11-30-00Z__m+v1"


def test_read_current_returns_none_when_blank(tmp_path: pathlib.Path) -> None:
    r = _root(tmp_path)
    art.atomic_write_text(r.current_file, "\n")
    assert art.read_current(r) is None


def test_read_manifest_raises_on_missing(tmp_path: pathlib.Path) -> None:
    with pytest.raises(art.OutputError, match="could not read manifest"):
        art.read_manifest(tmp_path / "nope.json")


def test_read_manifest_raises_on_bad_json(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "m.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(art.OutputError, match="could not read manifest"):
        art.read_manifest(p)


def test_read_manifest_raises_on_non_object(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "m.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(art.OutputError, match="not a JSON object"):
        art.read_manifest(p)


def test_resolve_current_returns_none_when_no_pointer(
    tmp_path: pathlib.Path,
) -> None:
    r = _root(tmp_path)
    assert art.resolve_current(r) is None


def test_resolve_current_raises_when_manifest_missing(
    tmp_path: pathlib.Path,
) -> None:
    r = _root(tmp_path)
    art.atomic_write_text(r.current_file, "2026-10-07T11-30-00Z__m+v1\n")
    with pytest.raises(art.OutputError, match=r"but .* is missing"):
        art.resolve_current(r)


def test_resolve_current_happy_path(tmp_path: pathlib.Path) -> None:
    r = _root(tmp_path)
    run_id = "2026-10-07T11-30-00Z__m+v1"
    rp = art.RunPaths(root=r, run_id=run_id)
    rp.dir.mkdir(parents=True, exist_ok=True)
    art.atomic_write_json(rp.manifest, {"schema_version": 1, "run_id": run_id})
    art.atomic_write_text(r.current_file, run_id + "\n")

    cur = art.resolve_current(r)
    assert cur is not None
    assert cur.run_id == run_id
    assert cur.manifest["schema_version"] == 1


# ============================================================================
# read_state
# ============================================================================
def _valid_state_doc(**over: object) -> dict[str, object]:
    doc: dict[str, object] = {
        "schema_version": 1,
        "model_version": "m+v1",
        "preprocessing_version": "v1",
        "config_hash": "sha256:deadbeef",
        "hash_algorithm": "sha256",
        "catalog_snapshot": "sha256:0000000000000000",
        "items": {"i_0001": "aaaa"},
    }
    doc.update(over)
    return doc


def _write_state(path: pathlib.Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


_STATE_KW = {
    "expected_model_version": "m+v1",
    "expected_preprocessing_version": "v1",
    "expected_config_hash": "sha256:deadbeef",
}


def test_read_state_missing(tmp_path: pathlib.Path) -> None:
    sr = art.read_state(tmp_path / "s.json", **_STATE_KW)
    assert not sr.valid
    assert sr.reason == "missing"
    assert sr.state is None


def test_read_state_malformed(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "s.json"
    p.write_text("{not json", encoding="utf-8")
    sr = art.read_state(p, **_STATE_KW)
    assert not sr.valid
    assert sr.reason == "malformed"


def test_read_state_wrong_schema_version(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "s.json"
    _write_state(p, _valid_state_doc(schema_version=2))
    sr = art.read_state(p, **_STATE_KW)
    assert not sr.valid
    assert sr.reason == "schema_version"


def test_read_state_wrong_model_version(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "s.json"
    _write_state(p, _valid_state_doc(model_version="m+v2"))
    sr = art.read_state(p, **_STATE_KW)
    assert not sr.valid
    assert sr.reason == "model_version"


def test_read_state_wrong_preprocessing_version(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "s.json"
    _write_state(p, _valid_state_doc(preprocessing_version="v2"))
    sr = art.read_state(p, **_STATE_KW)
    assert not sr.valid
    assert sr.reason == "preprocessing_version"


def test_read_state_wrong_config_hash(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "s.json"
    _write_state(p, _valid_state_doc(config_hash="sha256:other"))
    sr = art.read_state(p, **_STATE_KW)
    assert not sr.valid
    assert sr.reason == "config_hash"


def test_read_state_missing_items(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "s.json"
    _write_state(p, _valid_state_doc(items=None))
    sr = art.read_state(p, **_STATE_KW)
    assert not sr.valid
    assert sr.reason == "malformed"


def test_read_state_valid(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "s.json"
    _write_state(p, _valid_state_doc())
    sr = art.read_state(p, **_STATE_KW)
    assert sr.valid
    assert sr.reason is None
    assert sr.state is not None
    assert sr.state["model_version"] == "m+v1"


def test_read_state_items_extracts_mapping(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "s.json"
    _write_state(p, _valid_state_doc(items={"i_0001": "aaaa", "i_0002": "bbbb"}))
    sr = art.read_state(p, **_STATE_KW)
    assert sr.state is not None
    items = art.read_state_items(sr.state)
    assert items == {"i_0001": "aaaa", "i_0002": "bbbb"}


# ============================================================================
# validate_embeddings
# ============================================================================
def _good_embeddings(n: int = 3, dim: int = 4) -> NDArray[np.float32]:
    rows = [_fake_embedding(dim, i) for i in range(n)]
    return np.stack(rows)


def test_validate_ok() -> None:
    art.validate_embeddings(
        embeddings=_good_embeddings(),
        item_ids=["a", "b", "c"],
        expected_dim=4,
    )


def test_validate_empty_is_ok() -> None:
    art.validate_embeddings(
        embeddings=np.zeros((0, 4), dtype=np.float32),
        item_ids=[],
        expected_dim=4,
    )


def test_validate_rejects_wrong_dtype() -> None:
    # Deliberately wrong dtype; the signature expects float32 so we widen.
    arr = np.asarray(_good_embeddings(), dtype=np.float64)
    with pytest.raises(art.OutputError, match="dtype must be float32"):
        art.validate_embeddings(
            embeddings=arr,  # type: ignore[arg-type]
            item_ids=["a", "b", "c"],
            expected_dim=4,
        )


def test_validate_rejects_1d() -> None:
    arr = _good_embeddings()[0]
    with pytest.raises(art.OutputError, match="must be 2-D"):
        art.validate_embeddings(embeddings=arr, item_ids=["a"], expected_dim=4)


def test_validate_rejects_row_count_mismatch() -> None:
    with pytest.raises(art.OutputError, match="row count mismatch"):
        art.validate_embeddings(
            embeddings=_good_embeddings(n=3), item_ids=["a", "b"], expected_dim=4
        )


def test_validate_rejects_dim_mismatch() -> None:
    with pytest.raises(art.OutputError, match="dim mismatch"):
        art.validate_embeddings(
            embeddings=_good_embeddings(dim=8), item_ids=["a", "b", "c"], expected_dim=4
        )


def test_validate_rejects_nan() -> None:
    arr = _good_embeddings()
    arr[1, 0] = np.nan
    with pytest.raises(art.OutputError, match="non-finite"):
        art.validate_embeddings(embeddings=arr, item_ids=["a", "b", "c"], expected_dim=4)


def test_validate_rejects_inf() -> None:
    arr = _good_embeddings()
    arr[0, 0] = np.inf
    with pytest.raises(art.OutputError, match="non-finite"):
        art.validate_embeddings(embeddings=arr, item_ids=["a", "b", "c"], expected_dim=4)


def test_validate_rejects_duplicate_ids() -> None:
    with pytest.raises(art.OutputError, match="not unique"):
        art.validate_embeddings(
            embeddings=_good_embeddings(), item_ids=["a", "a", "c"], expected_dim=4
        )


def test_validate_rejects_denormalized() -> None:
    arr = _good_embeddings()
    arr[0] = arr[0] * 2.0
    with pytest.raises(art.OutputError, match="not L2-normalized"):
        art.validate_embeddings(embeddings=arr, item_ids=["a", "b", "c"], expected_dim=4)


# ============================================================================
# content_hash_map_from_items
# ============================================================================
def test_content_hash_map_from_items_accepts_typed_dict() -> None:
    # Any Mapping-compatible structure works; we use plain dicts here.
    items = [{"item_id": "i_1", "title": "A"}, {"item_id": "i_2", "title": "B"}]
    out = art.content_hash_map_from_items(items, content_hash_fn=lambda it: "h-" + it["item_id"])
    assert out == {"i_1": "h-i_1", "i_2": "h-i_2"}


# ============================================================================
# build_manifest / build_state
# ============================================================================
def test_build_manifest_shape() -> None:
    m = art.build_manifest(
        run_id="r1",
        created_at="2026-10-07T11:30:00Z",
        mode="incremental",
        model_version="m+v1",
        preprocessing_version="v1",
        config_hash_value="sha256:abc",
        catalog_snapshot="sha256:0000000000000000",
        onnx_artifact_sha256="a" * 64,
        parquet_relative_path="runs/r1/embeddings.parquet",
        parquet_sha256="f" * 64,
        parquet_rows=10,
        parquet_encoded_rows=2,
        parquet_dim=384,
        parquet_dtype="float32",
        state_relative_path="runs/r1/state.json",
        state_sha256="e" * 64,
        environment={"python_version": "3.11.0"},
    )
    assert m["schema_version"] == 1
    assert m["parquet"]["rows"] == 10
    assert m["parquet"]["encoded_rows"] == 2
    assert m["state"]["sha256"] == "e" * 64
    # Paths are relative, never absolute.
    assert not m["parquet"]["path"].startswith("/")
    assert not m["state"]["path"].startswith("/")


def test_build_state_shape() -> None:
    s = art.build_state(
        model_version="m+v1",
        preprocessing_version="v1",
        config_hash_value="sha256:abc",
        catalog_snapshot="sha256:0000000000000000",
        items={"i_1": "h1", "i_2": "h2"},
    )
    assert s["schema_version"] == 1
    assert s["hash_algorithm"] == "sha256"
    assert s["items"] == {"i_1": "h1", "i_2": "h2"}


# ============================================================================
# commit_run
# ============================================================================
def test_commit_run_writes_manifest_state_and_pointer(
    tmp_path: pathlib.Path,
) -> None:
    r = _root(tmp_path)
    run_id = "2026-10-07T11-30-00Z__m+v1"
    rp = art.RunPaths(root=r, run_id=run_id)
    rp.dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(rp.parquet, ids=["i_0001", "i_0002"])

    manifest = {"schema_version": 1, "run_id": run_id}
    state = {"schema_version": 1, "items": {"i_0001": "h0", "i_0002": "h1"}}
    art.commit_run(root=r, run_paths=rp, manifest=manifest, state_doc=state)

    assert rp.manifest.is_file()
    assert rp.state.is_file()
    assert art.read_current(r) == run_id
    assert json.loads(rp.manifest.read_text())["run_id"] == run_id


def test_commit_run_fails_without_parquet(tmp_path: pathlib.Path) -> None:
    r = _root(tmp_path)
    rp = art.RunPaths(root=r, run_id="2026-10-07T11-30-00Z__m+v1")
    rp.dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(art.OutputError, match="parquet not found"):
        art.commit_run(root=r, run_paths=rp, manifest={}, state_doc={})


def test_commit_run_does_not_move_pointer_on_manifest_failure(
    tmp_path: pathlib.Path,
) -> None:
    """A failure between Parquet and pointer must not corrupt ``current``."""
    r = _root(tmp_path)
    # Pre-existing active run.
    prev_id = "2026-10-07T10-00-00Z__m+v1"
    prev = art.RunPaths(root=r, run_id=prev_id)
    prev.dir.mkdir(parents=True, exist_ok=True)
    art.atomic_write_text(r.current_file, prev_id + "\n")

    # New run: parquet present, but manifest is deliberately un-serialisable.
    new_id = "2026-10-07T11-00-00Z__m+v1"
    new = art.RunPaths(root=r, run_id=new_id)
    new.dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(new.parquet, ids=["i_0001"])

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        art.commit_run(
            root=r,
            run_paths=new,
            manifest={"x": Unserialisable()},
            state_doc={},
        )
    # The old pointer is unchanged.
    assert art.read_current(r) == prev_id


# ============================================================================
# ParquetBatchWriter + iter_previous_batches
# ============================================================================
def test_parquet_roundtrip(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "e.parquet"
    ids = ["i_0001", "i_0002", "i_0003"]
    _write_parquet(p, ids=ids, dim=4)

    batches = list(art.iter_previous_batches(p, batch_size=2))
    all_ids = [iid for b in batches for iid in b.item_ids]
    assert all_ids == ids

    # Embeddings are preserved row-for-row.
    expected = np.stack([_fake_embedding(4, i) for i in range(3)])
    got = np.concatenate([b.embeddings for b in batches], axis=0)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_parquet_writer_ignores_empty_batches(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "e.parquet"
    with art.ParquetBatchWriter(p, embedding_dim=4) as w:
        w.write_batch(
            item_ids=[],
            embeddings=np.zeros((0, 4), dtype=np.float32),
            content_hashes=[],
            preprocessing_version="v1",
        )
    assert p.is_file()
    assert list(art.iter_previous_batches(p)) == []


def test_parquet_writer_rejects_length_mismatch(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "e.parquet"
    with (
        art.ParquetBatchWriter(p, embedding_dim=4) as w,
        pytest.raises(art.OutputError, match="length mismatch"),
    ):
        w.write_batch(
            item_ids=["a", "b"],
            embeddings=np.zeros((1, 4), dtype=np.float32),
            content_hashes=["h1", "h2"],
            preprocessing_version="v1",
        )


def test_parquet_writer_rejects_dim_mismatch(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "e.parquet"
    with (
        art.ParquetBatchWriter(p, embedding_dim=4) as w,
        pytest.raises(art.OutputError, match="dim"),
    ):
        w.write_batch(
            item_ids=["a"],
            embeddings=np.zeros((1, 8), dtype=np.float32),
            content_hashes=["h1"],
            preprocessing_version="v1",
        )


def test_iter_previous_batches_rejects_bad_batch_size(
    tmp_path: pathlib.Path,
) -> None:
    p = tmp_path / "e.parquet"
    _write_parquet(p, ids=["i_0001"])
    with pytest.raises(ValueError, match="batch_size"):
        list(art.iter_previous_batches(p, batch_size=0))


def test_parquet_columns_match_contract(tmp_path: pathlib.Path) -> None:
    import pyarrow.parquet as pq

    p = tmp_path / "e.parquet"
    _write_parquet(p, ids=["i_0001", "i_0002"])
    table = pq.read_table(str(p))
    assert table.column_names == [
        "item_id",
        "embedding",
        "content_hash",
        "preprocessing_version",
    ]
