"""Build an index from an embedding run.

Reads the active run (per ``artifacts/embeddings/current``), computes the
``index_version`` (ADR-0007), and writes rows into ``index_registry``,
``item``, and ``embedding`` (ADR-0006). The result is a ``building`` row
that ``promote_index.py`` will later activate.

The module is split into a pure part (:func:`resolve_build_inputs`,
:func:`read_catalog_items`) that reads disk and config without a database,
and a DB-touching part (:func:`build_index`) that takes a connection. That
split keeps the disk-reading contract testable in Colab, where a pgvector
service is not available.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Literal

from recsys.embeddings.artifacts import iter_previous_batches
from recsys.embeddings.preprocess import CatalogItem, item_content_hash
from recsys.monitoring.logging import get_logger
from recsys.retrieval.identity import (
    IndexIdentityInputs,
    resolve_index_version,
)

log = get_logger(__name__)

DEFAULT_BATCH_SIZE = 1_000
DEFAULT_GOLDEN_SET_VERSION = "v1"


# ============================================================================
# pure: disk and config
# ============================================================================
@dataclass(frozen=True)
class BuildInputs:
    """Everything needed to describe a build, read from disk and config."""

    run_id: str
    run_dir: pathlib.Path
    parquet_path: pathlib.Path
    model_version: str
    catalog_snapshot: str
    preprocessing_version: str
    onnx_artifact_sha256: str
    parquet_rows: int


class BuildInputError(Exception):
    """Raised for problems reading the run directory or the catalog."""


def resolve_build_inputs(
    embeddings_root: pathlib.Path,
    *,
    run_id: str | None = None,
) -> BuildInputs:
    """Read the active run (or a named one) and return its :class:`BuildInputs`.

    If ``run_id`` is ``None``, reads ``embeddings_root/current``. If given,
    reads ``embeddings_root/runs/<run_id>/manifest.json`` directly, which
    is useful for rebuilding a specific run.
    """
    if run_id is None:
        current_file = embeddings_root / "current"
        if not current_file.is_file():
            raise BuildInputError(
                f"no active run: {current_file} does not exist. "
                f"Run the embedding pipeline first "
                f"(python -m recsys.embeddings.pipeline --mode=batch ...)"
            )
        run_id = current_file.read_text(encoding="utf-8").strip()
        if not run_id:
            raise BuildInputError(f"current pointer is empty: {current_file}")

    run_dir = embeddings_root / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BuildInputError(f"manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BuildInputError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BuildInputError(f"manifest is not a JSON object: {manifest_path}")

    for field in (
        "model_version",
        "catalog_snapshot",
        "preprocessing_version",
        "onnx_artifact_sha256",
        "parquet",
    ):
        if field not in manifest:
            raise BuildInputError(f"manifest missing field {field!r}: {manifest_path}")

    parquet_field = manifest["parquet"]
    if not isinstance(parquet_field, dict) or "path" not in parquet_field:
        raise BuildInputError(f"manifest.parquet missing 'path': {manifest_path}")

    parquet_path = embeddings_root / parquet_field["path"]
    if not parquet_path.is_file():
        raise BuildInputError(f"parquet not found: {parquet_path}")

    return BuildInputs(
        run_id=run_id,
        run_dir=run_dir,
        parquet_path=parquet_path,
        model_version=str(manifest["model_version"]),
        catalog_snapshot=str(manifest["catalog_snapshot"]),
        preprocessing_version=str(manifest["preprocessing_version"]),
        onnx_artifact_sha256=str(manifest["onnx_artifact_sha256"]),
        parquet_rows=int(parquet_field.get("rows", 0)),
    )


def read_catalog_items(catalog_path: pathlib.Path) -> dict[str, CatalogItem]:
    """Load a JSONL catalog into ``{item_id: CatalogItem}``.

    Raises :class:`BuildInputError` on malformed input. Duplicate
    ``item_id`` is an error; the pipeline has the same rule.
    """
    if not catalog_path.is_file():
        raise BuildInputError(f"catalog not found: {catalog_path}")
    items: dict[str, CatalogItem] = {}
    required = {"item_id", "title", "description", "category", "brand"}
    for lineno, raw in enumerate(catalog_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BuildInputError(f"line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise BuildInputError(f"line {lineno}: not a JSON object")
        missing = required - set(obj.keys())
        if missing:
            raise BuildInputError(f"line {lineno}: missing fields {sorted(missing)}")
        iid = obj["item_id"]
        if not isinstance(iid, str) or not iid:
            raise BuildInputError(f"line {lineno}: item_id must be a non-empty string")
        if iid in items:
            raise BuildInputError(f"line {lineno}: duplicate item_id {iid!r}")
        items[iid] = CatalogItem(
            item_id=iid,
            title=str(obj["title"]),
            description=str(obj["description"]),
            category=str(obj["category"]),
            brand=str(obj["brand"]),
        )
    if not items:
        raise BuildInputError(f"catalog {catalog_path} contains no items")
    return items


# ============================================================================
# pure: vector literal (duplicated from pgvector.py; kept private here to
# avoid a circular import — pgvector.py could import from build.py but not
# the other way around, so we re-declare a minimal version).
# ============================================================================
def _vector_literal(values: list[float]) -> str:
    """Render a float list as a pgvector literal ``[a,b,c]``."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


# ============================================================================
# db: registry lookup + build
# ============================================================================
@dataclass(frozen=True)
class BuildResult:
    index_version: str
    status: Literal["created", "duplicate"]
    row_count: int
    duration_ms: int


def _registry_lookup(connection: Any) -> Any:
    """Return a callable ``version -> IndexIdentityInputs | None``.

    The callable queries ``index_registry`` for the seven columns that
    enter the hash (ADR-0007). It is used by
    :func:`recsys.retrieval.identity.resolve_index_version`.
    """

    def lookup(index_version: str) -> IndexIdentityInputs | None:
        with connection.cursor() as cur:
            cur.execute(
                """
                SELECT model_version, catalog_snapshot, preprocessing_version,
                       metric, hnsw_m, hnsw_ef_construction, pgvector_version
                FROM index_registry
                WHERE index_version = %s
                """,
                (index_version,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return IndexIdentityInputs(
            model_version=str(row[0]),
            catalog_snapshot=str(row[1]),
            preprocessing_version=str(row[2]),
            metric=str(row[3]),
            hnsw_m=int(row[4]),
            hnsw_ef_construction=int(row[5]),
            pgvector_version=str(row[6]),
        )

    return lookup


def _read_pgvector_version(connection: Any) -> str:
    with connection.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(
            "pgvector extension is not installed in this database; "
            "run `alembic upgrade head` (migration 0001 creates it)."
        )
    return str(row[0])


def build_index(
    connection: Any,
    *,
    inputs: BuildInputs,
    catalog_items: dict[str, CatalogItem],
    metric: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
    golden_set_version: str = DEFAULT_GOLDEN_SET_VERSION,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> BuildResult:
    """Create a ``building`` row for the index and populate it.

    Idempotent per ADR-0007: if the resolved ``index_version`` already
    exists with identical inputs, returns ``status="duplicate"`` without
    touching the database. If a previous run left a ``building`` row for
    this version (a crash), it is deleted first; the FK cascade removes
    its embeddings.
    """
    t0 = time.perf_counter()

    pgvector_version = _read_pgvector_version(connection)
    identity_inputs = IndexIdentityInputs(
        model_version=inputs.model_version,
        catalog_snapshot=inputs.catalog_snapshot,
        preprocessing_version=inputs.preprocessing_version,
        metric=metric,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        pgvector_version=pgvector_version,
    )
    index_version, status = resolve_index_version(
        identity_inputs, lookup=_registry_lookup(connection)
    )

    if status == "duplicate":
        log.info("index.build.duplicate", index_version=index_version)
        return BuildResult(
            index_version=index_version,
            status="duplicate",
            row_count=0,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )

    # Clean up any orphaned 'building' row from a previous crash. Cascade
    # deletes the embeddings that were already inserted.
    with connection.cursor() as cur:
        cur.execute(
            "DELETE FROM index_registry WHERE index_version = %s AND status = 'building'",
            (index_version,),
        )
    connection.commit()

    # Insert the registry row in 'building' state.
    with connection.cursor() as cur:
        cur.execute(
            """
            INSERT INTO index_registry (
                index_version, model_version, catalog_snapshot,
                preprocessing_version, metric,
                hnsw_m, hnsw_ef_construction, hnsw_ef_search,
                pgvector_version, golden_set_version,
                row_count, status
            ) VALUES (
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s,
                0, 'building'
            )
            """,
            (
                index_version,
                identity_inputs.model_version,
                identity_inputs.catalog_snapshot,
                identity_inputs.preprocessing_version,
                identity_inputs.metric,
                identity_inputs.hnsw_m,
                identity_inputs.hnsw_ef_construction,
                hnsw_ef_search,
                identity_inputs.pgvector_version,
                golden_set_version,
            ),
        )
    connection.commit()

    # Upsert items. Content hash is computed the same way the pipeline
    # does it, so the value stored here matches what the Parquet records.
    item_rows = [
        (
            it["item_id"],
            it["title"],
            it["description"],
            it["category"],
            it["brand"],
            item_content_hash(it),
        )
        for it in catalog_items.values()
    ]
    with connection.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO item (
                item_id, title, description, category, brand, content_hash
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (item_id) DO UPDATE SET
                title        = EXCLUDED.title,
                description  = EXCLUDED.description,
                category     = EXCLUDED.category,
                brand        = EXCLUDED.brand,
                content_hash = EXCLUDED.content_hash,
                updated_at   = now()
            """,
            item_rows,
        )
    connection.commit()

    # Stream the Parquet and insert embeddings in batches.
    total = 0
    for batch in iter_previous_batches(inputs.parquet_path, batch_size=batch_size):
        rows = [
            (
                batch.item_ids[i],
                index_version,
                identity_inputs.model_version,
                _vector_literal(batch.embeddings[i].tolist()),
            )
            for i in range(len(batch.item_ids))
        ]
        with connection.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO embedding (item_id, index_version, model_version, vector)
                VALUES (%s, %s, %s, %s::vector)
                """,
                rows,
            )
        connection.commit()
        total += len(rows)

    # Finalize the registry row.
    with connection.cursor() as cur:
        cur.execute(
            "UPDATE index_registry SET row_count = %s WHERE index_version = %s",
            (total, index_version),
        )
    connection.commit()

    duration_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "index.build.done",
        index_version=index_version,
        model_version=identity_inputs.model_version,
        catalog_snapshot=identity_inputs.catalog_snapshot,
        row_count=total,
        duration_ms=duration_ms,
    )
    return BuildResult(
        index_version=index_version,
        status="created",
        row_count=total,
        duration_ms=duration_ms,
    )
