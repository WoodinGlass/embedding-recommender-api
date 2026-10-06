"""Preprocessing rules for catalog items.

The rules are pinned and versioned (``PREPROCESSING_VERSION``). Any change to
the rules bumps the version, which invalidates every ``catalog_snapshot``
computed against the old rules. See ``docs/embedding-pipeline.md`` § 4.

The functions in this module are pure: given the same input, they return the
same output. No I/O, no logging, no configuration. That property is what
makes the determinism contract testable without a model.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import TypedDict

PREPROCESSING_VERSION = "v1"

#: Character-level truncation applied *before* tokenization. This is an
#: upper bound only — the tokenizer applies its own truncation to
#: ``config.json`` max_seq_length at encode time. 512 characters is a safe
#: cap for a 256-token model: worst case, one token per character for
#: unusual scripts; a five-character English word averages ~1.3 tokens.
MAX_CHARS = 512

_WHITESPACE_RE = re.compile(r"\s+")


class CatalogItem(TypedDict):
    """A single item from the catalog. Shape is fixed; see
    ``docs/embedding-pipeline.md`` § 2 for the field list.
    """

    item_id: str
    title: str
    description: str
    category: str
    brand: str


def _normalize_unicode(text: str) -> str:
    """NFKC normalization. Turns visually-equivalent sequences into one form."""
    return unicodedata.normalize("NFKC", text)


def _collapse_whitespace(text: str) -> str:
    """Collapse any run of whitespace into a single ASCII space."""
    return _WHITESPACE_RE.sub(" ", text)


def preprocess_item(item: CatalogItem) -> str:
    """Return the text that will be embedded for ``item``.

    Rules, in order (pinned by :data:`PREPROCESSING_VERSION`):

    1. Unicode NFKC normalization on ``title`` and ``description``.
    2. Whitespace collapse.
    3. Concatenate with ``" | "``; if ``description`` is empty after
       stripping, use only ``title`` (no trailing separator).
    4. Truncate to :data:`MAX_CHARS` characters.
    5. Strip leading and trailing whitespace.
    """
    title = _collapse_whitespace(_normalize_unicode(item["title"])).strip()
    description = _collapse_whitespace(_normalize_unicode(item["description"])).strip()
    text = f"{title} | {description}" if description else title
    return text[:MAX_CHARS].strip()


def _sha256_hex16(text: str) -> str:
    """First 16 hex characters of the SHA256 of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def item_content_hash(item: CatalogItem) -> str:
    """Return the 16-hex ``content_hash`` for ``item``.

    The hash covers the *preprocessed* text, so a rule change in this module
    changes every item's hash and therefore every ``catalog_snapshot``.
    Stored as a column in the embedding Parquet, so incremental runs decide
    "changed / unchanged" without re-reading the raw catalog.
    """
    return _sha256_hex16(preprocess_item(item))


def catalog_snapshot(items: list[CatalogItem]) -> str:
    """Return the ``catalog_snapshot`` identifier for ``items``.

    Format: ``sha256:<16 hex>``. The hash covers the sorted list of
    ``(item_id, content_hash)`` pairs, so the result depends only on the
    semantic contents of the catalog, not on row order. See
    ``docs/embedding-pipeline.md`` § 3.
    """
    pairs = sorted((it["item_id"], item_content_hash(it)) for it in items)
    joined = "\n".join(f"{iid}:{ch}" for iid, ch in pairs)
    return f"sha256:{_sha256_hex16(joined)}"
