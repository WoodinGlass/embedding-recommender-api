"""Embedding pipeline: preprocessing, encoding, and ONNX export.

Public surface:

- :func:`preprocess_item` / :data:`PREPROCESSING_VERSION` — pure text rules.
- :func:`item_content_hash` / :func:`catalog_snapshot` — identifiers used in
  artifact filenames and manifests (see ``docs/contracts.md`` § 1.3).
- :class:`Encoder` protocol, :class:`OnnxEncoder`, :class:`ReferenceEncoder`.

The design doc is ``docs/embedding-pipeline.md``.
"""

from recsys.embeddings.encoder import (
    Encoder,
    OnnxEncoder,
    ReferenceEncoder,
)
from recsys.embeddings.preprocess import (
    MAX_CHARS,
    PREPROCESSING_VERSION,
    CatalogItem,
    catalog_snapshot,
    item_content_hash,
    preprocess_item,
)

__all__ = [
    "MAX_CHARS",
    "PREPROCESSING_VERSION",
    "CatalogItem",
    "Encoder",
    "OnnxEncoder",
    "ReferenceEncoder",
    "catalog_snapshot",
    "item_content_hash",
    "preprocess_item",
]
