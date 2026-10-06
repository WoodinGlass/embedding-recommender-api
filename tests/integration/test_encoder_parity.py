"""ONNX encoder parity and determinism — integration tier.

These tests require both the ``export`` and ``inference`` extras (sentence-
transformers and onnxruntime). They download a small model on first run;
GitHub Actions caches the HuggingFace hub directory between runs.

Assertions follow ``docs/embedding-pipeline.md`` § 5:

- **Tolerance**: per-row cosine similarity between ONNX and the reference
  model is above 0.999 (the ONNX export is *not* expected to be bit-identical
  to torch — the graph includes pooling and normalization ops that torch
  evaluates at float32 with slightly different kernel order).
- **Semantic**: for a fixed probe set (the sample catalog), the top-10
  neighbours by cosine distance are identical between two encodes of the
  same texts, with ties broken by item index ascending.
- **Artifact integrity**: the SHA256 sidecar matches the artifact bytes.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from numpy.typing import NDArray

from recsys.embeddings.encoder import OnnxEncoder, ReferenceEncoder, sha256_file
from recsys.embeddings.onnx_export import export
from recsys.embeddings.preprocess import CatalogItem, preprocess_item

pytestmark = [pytest.mark.integration, pytest.mark.encoder]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SLUG = "sentence-transformers__all-MiniLM-L6-v2"
LABEL = "minilm-onnx-v1"
ONNX_DIR = REPO_ROOT / "artifacts" / "onnx" / MODEL_SLUG
CATALOG_PATH = REPO_ROOT / "data" / "sample" / "catalog.jsonl"


def _load_catalog() -> list[CatalogItem]:
    lines = CATALOG_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _ensure_artifact() -> pathlib.Path:
    if not (ONNX_DIR / "model.onnx").is_file():
        export(MODEL_NAME, ONNX_DIR)
    return ONNX_DIR


@pytest.fixture(scope="module")
def onnx_encoder() -> OnnxEncoder:
    return OnnxEncoder(_ensure_artifact(), label=LABEL)


@pytest.fixture(scope="module")
def reference_encoder() -> ReferenceEncoder:
    return ReferenceEncoder(MODEL_NAME)


# --------------------------------------------------------------------------- #
# artifact integrity
# --------------------------------------------------------------------------- #
def test_sha_sidecar_matches_artifact() -> None:
    d = _ensure_artifact()
    onnx_path = d / "model.onnx"
    recorded = (d / "model.onnx.sha256").read_text(encoding="utf-8").strip()
    assert recorded == sha256_file(onnx_path)


def test_onnx_encoder_refuses_mismatched_sidecar(tmp_path: pathlib.Path) -> None:
    d = _ensure_artifact()
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for name in ("model.onnx", "tokenizer.json", "config.json"):
        (tampered / name).write_bytes((d / name).read_bytes())
    # Deliberately wrong SHA sidecar.
    (tampered / "model.onnx.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        OnnxEncoder(tampered, label=LABEL)


# --------------------------------------------------------------------------- #
# basic properties
# --------------------------------------------------------------------------- #
def test_model_version_shape(onnx_encoder: OnnxEncoder) -> None:
    mv = onnx_encoder.model_version
    assert mv.startswith(LABEL + "+")
    assert len(mv) == len(LABEL) + 1 + 8


def test_embedding_dim_matches_reference(
    onnx_encoder: OnnxEncoder, reference_encoder: ReferenceEncoder
) -> None:
    assert onnx_encoder.embedding_dim == reference_encoder.embedding_dim


def test_output_is_l2_normalized(onnx_encoder: OnnxEncoder) -> None:
    out = onnx_encoder.encode(["hello world", "second sentence"])
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


def test_empty_batch(onnx_encoder: OnnxEncoder) -> None:
    out = onnx_encoder.encode([])
    assert out.shape == (0, onnx_encoder.embedding_dim)


# --------------------------------------------------------------------------- #
# parity: ONNX vs reference
# --------------------------------------------------------------------------- #
_PARITY_TEXTS = [
    "a guide to the ring road in iceland",
    "blue note vinyl from the 1960s",
    "a lightweight daily running shoe",
    "how to pull a perfect espresso shot",
    "a beginner mechanical keyboard kit",
    "post-bop jazz with odd time signatures",
    "high-altitude trail running in wet conditions",
    "a dual-boiler espresso machine with pid",
]


def test_onnx_matches_reference_within_tolerance(
    onnx_encoder: OnnxEncoder, reference_encoder: ReferenceEncoder
) -> None:
    a = onnx_encoder.encode(_PARITY_TEXTS)
    b = reference_encoder.encode(_PARITY_TEXTS)
    assert a.shape == b.shape
    # Cosine similarity per row. Both sides are L2-normalized, so the dot
    # product is the cosine similarity.
    per_row = (a * b).sum(axis=1)
    min_sim = float(per_row.min())
    assert min_sim >= 0.999, f"min cosine similarity {min_sim:.6f} < 0.999"


# --------------------------------------------------------------------------- #
# semantic determinism: top-10 neighbours identical across two runs
# --------------------------------------------------------------------------- #
def _topk_stable(sim_row: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    """Top-k indices by (-sim, index), i.e. ties broken by index ascending."""
    result: NDArray[np.int64] = np.lexsort((np.arange(sim_row.shape[0]), -sim_row))[:k]
    return result


def test_topk_neighbours_are_identical_between_two_encodes(
    onnx_encoder: OnnxEncoder,
) -> None:
    catalog = _load_catalog()
    texts = [preprocess_item(it) for it in catalog]
    k = 10

    emb1 = onnx_encoder.encode(texts)
    emb2 = onnx_encoder.encode(texts)
    assert emb1.shape == emb2.shape

    sims1 = emb1 @ emb1.T
    sims2 = emb2 @ emb2.T

    for i in range(len(texts)):
        top1 = _topk_stable(sims1[i], k)
        top2 = _topk_stable(sims2[i], k)
        np.testing.assert_array_equal(
            top1,
            top2,
            err_msg=f"top-{k} neighbours differ for item index {i}",
        )
