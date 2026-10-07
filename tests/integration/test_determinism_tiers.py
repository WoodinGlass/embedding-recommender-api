"""Three-tier determinism contract, tested explicitly.

The contract is defined in ``docs/embedding-pipeline.md`` § 5:

- **Strict tier** — SHA256 of the Parquet matches between two consecutive
  runs in the same pinned environment. CI only: ONNX Runtime and BLAS may
  produce bit-level differences across thread counts. Enabled in CI by
  setting ``RECSYS_STRICT_DETERMINISM=1``; skipped locally with an
  explanatory reason.

- **Semantic tier** — Top-k (k=10) neighbours are identical between two
  runs, ties broken by item index ascending. Runs everywhere. This is the
  user-visible contract: retrieval must not change between runs
  regardless of bit-level bytes.

- **Tolerance tier** — Per-row cosine similarity between two runs is
  >= 0.9999. Runs everywhere. Catches a class of drift that is
  semantically meaningful (a real change in embedding content), not
  bit-level noise.

Each test below runs the full pipeline twice over the sample catalog and
compares the resulting Parquet. Marked ``encoder`` so CI can run them in
the dedicated encoder-parity job.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray

from recsys.embeddings import artifacts as art
from recsys.embeddings import pipeline as pl
from recsys.embeddings.encoder import OnnxEncoder
from recsys.embeddings.onnx_export import export

pytestmark = [pytest.mark.integration, pytest.mark.encoder]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SLUG = "sentence-transformers__all-MiniLM-L6-v2"
LABEL = "minilm-onnx-v1"
ONNX_DIR = REPO_ROOT / "artifacts" / "onnx" / MODEL_SLUG
CATALOG = REPO_ROOT / "data" / "sample" / "catalog.jsonl"

STRICT_TIER_ENV = "RECSYS_STRICT_DETERMINISM"
SEMANTIC_K = 10
TOLERANCE_MIN_COSINE = 0.9999


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def onnx_dir() -> pathlib.Path:
    if not (ONNX_DIR / "model.onnx").is_file():
        export(MODEL_NAME, ONNX_DIR)
    return ONNX_DIR


@pytest.fixture(scope="module")
def encoder(onnx_dir: pathlib.Path) -> OnnxEncoder:
    return OnnxEncoder(onnx_dir, label=LABEL)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _run_once(
    out_dir: pathlib.Path,
    encoder: OnnxEncoder,
) -> pl.RunSummary:
    return pl.run(
        catalog_path=CATALOG,
        out_dir=out_dir,
        encoder=encoder,
        mode="batch",
        batch_size=64,
        onnx_artifact_sha256=encoder.sha256,
        lock_timeout=10.0,
        progress=False,
    )


def _parquet_path(out_dir: pathlib.Path, run_id: str) -> pathlib.Path:
    return art.RunPaths(root=art.EmbeddingsRoot(path=out_dir), run_id=run_id).parquet


def _load_ids_and_embeddings(
    path: pathlib.Path,
) -> tuple[list[str], NDArray[np.float32]]:
    table = pq.read_table(str(path))
    ids = table.column("item_id").to_pylist()
    emb = np.asarray(
        table.column("embedding").to_pylist(),
        dtype=np.float32,
    )
    return ids, emb


def _topk_indices(sim_row: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    """Top-k indices by (-sim, index): ties broken by index ascending."""
    order: NDArray[np.int64] = np.lexsort((np.arange(sim_row.shape[0]), -sim_row))[:k]
    return order


# --------------------------------------------------------------------------- #
# Strict tier
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get(STRICT_TIER_ENV) != "1",
    reason=(
        "strict (byte-identical) tier is CI-only; set "
        f"{STRICT_TIER_ENV}=1 in an environment with pinned thread counts "
        "to run it"
    ),
)
def test_strict_tier_two_batch_runs_produce_identical_bytes(
    tmp_path: pathlib.Path, encoder: OnnxEncoder
) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    s1 = _run_once(out1, encoder)
    s2 = _run_once(out2, encoder)
    assert s1.checksum is not None
    assert s2.checksum is not None
    assert s1.checksum == s2.checksum, (
        "STRICT TIER FAILED: two batch runs produced different Parquet bytes\n"
        f"  run1 = {s1.checksum}\n"
        f"  run2 = {s2.checksum}"
    )


# --------------------------------------------------------------------------- #
# Semantic tier
# --------------------------------------------------------------------------- #
def test_semantic_tier_topk_identical(tmp_path: pathlib.Path, encoder: OnnxEncoder) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    s1 = _run_once(out1, encoder)
    s2 = _run_once(out2, encoder)

    ids1, emb1 = _load_ids_and_embeddings(_parquet_path(out1, s1.run_id))
    ids2, emb2 = _load_ids_and_embeddings(_parquet_path(out2, s2.run_id))
    assert ids1 == ids2, "item_id order differs between two runs"

    sims1 = emb1 @ emb1.T
    sims2 = emb2 @ emb2.T
    for i in range(len(ids1)):
        top1 = _topk_indices(sims1[i], SEMANTIC_K)
        top2 = _topk_indices(sims2[i], SEMANTIC_K)
        np.testing.assert_array_equal(
            top1,
            top2,
            err_msg=f"SEMANTIC TIER FAILED: top-{SEMANTIC_K} differs for {ids1[i]!r}",
        )


# --------------------------------------------------------------------------- #
# Tolerance tier
# --------------------------------------------------------------------------- #
def test_tolerance_tier_per_row_cosine(tmp_path: pathlib.Path, encoder: OnnxEncoder) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    s1 = _run_once(out1, encoder)
    s2 = _run_once(out2, encoder)

    _, emb1 = _load_ids_and_embeddings(_parquet_path(out1, s1.run_id))
    _, emb2 = _load_ids_and_embeddings(_parquet_path(out2, s2.run_id))
    assert emb1.shape == emb2.shape

    per_row = (emb1 * emb2).sum(axis=1)
    min_sim = float(per_row.min())
    assert min_sim >= TOLERANCE_MIN_COSINE, (
        f"TOLERANCE TIER FAILED: min cosine similarity {min_sim:.6f} < {TOLERANCE_MIN_COSINE}"
    )
