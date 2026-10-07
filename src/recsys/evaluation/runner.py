"""Offline evaluation runner.

Orchestrates the pieces: for each query in the golden set, encode the
query vector from seed embeddings (ADR-0009 § 3: mean + L2 normalize),
call each backend, compute metrics, and assemble a report the gate can
consume (ADR-0010).

The runner does not open connections, does not read configuration, and
does not write files except when the caller asks it to. It takes its
inputs as arguments:

- ``golden_set`` — a validated :class:`GoldenSet`.
- ``systems`` — a mapping from a system name (used as the report key and
  as the threshold file key) to an :class:`IndexBackend`.
- ``embedding_lookup`` — a function from item id to the item's stored
  vector, or ``None`` if the id has no vector. The lookup is what makes
  the query vector possible; it is a parameter rather than a class method
  so tests can supply a dict-backed function and the CLI can supply one
  backed by a :class:`NumpyBackend` or an artifact run.
- ``exact_backend`` — optional. When provided, the runner also computes
  the ANN-fidelity metric (ADR-0009 § 5) for each system, comparing the
  system's top-k to the exact backend's top-k for the same query vector.

The report schema is fixed by ADR-0010 § The report file. It is the
contract between this runner and the gate in :mod:`recsys.evaluation.thresholds`.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from recsys.evaluation.golden_set import GoldenSet
from recsys.evaluation.metrics import (
    ann_recall_vs_exact,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from recsys.retrieval.base import IndexBackend

#: Report schema version. Matches the ``schema_version`` the gate expects.
REPORT_SCHEMA_VERSION = 1

EmbeddingLookup = Callable[[str], NDArray[np.float32] | None]


class EvaluationError(Exception):
    """Raised when evaluation cannot proceed (missing vectors, empty set)."""


# --------------------------------------------------------------------------- #
# query encoding
# --------------------------------------------------------------------------- #
def encode_query(
    seed_item_ids: Sequence[str],
    *,
    embedding_lookup: EmbeddingLookup,
) -> NDArray[np.float32]:
    """Return the L2-normalized mean of the seed embeddings.

    ADR-0009 § 3. The mean is deterministic and parameter-free; the L2
    normalization is required because cosine distance is only a valid
    score on unit vectors. A seed whose embedding is missing is skipped
    with an error only if every seed is missing — a partially-available
    seed set still produces a usable query.

    Raises :class:`EvaluationError` if no seed has an embedding, or if
    the mean is the zero vector (which cannot be normalized).
    """
    if not seed_item_ids:
        raise EvaluationError("cannot encode a query with no seeds")

    vectors: list[NDArray[np.float32]] = []
    for iid in seed_item_ids:
        v = embedding_lookup(iid)
        if v is not None:
            vectors.append(np.asarray(v, dtype=np.float32))

    if not vectors:
        raise EvaluationError(f"none of the seed items {list(seed_item_ids)} has an embedding")

    stacked = np.stack(vectors, axis=0)
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-12:
        raise EvaluationError(f"query vector for seeds {list(seed_item_ids)} is the zero vector")
    result: NDArray[np.float32] = (mean / norm).astype(np.float32)
    return result


# --------------------------------------------------------------------------- #
# per-system evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QueryOutcome:
    """One query evaluated against one system."""

    query_id: str
    retrieved: tuple[str, ...]
    metrics: dict[str, float]
    ann_recall_vs_exact: float | None


def evaluate_system(
    *,
    backend: IndexBackend,
    golden_set: GoldenSet,
    embedding_lookup: EmbeddingLookup,
    k: int,
    exact_backend: IndexBackend | None = None,
    include_per_query: bool = False,
) -> tuple[dict[str, float], list[QueryOutcome]]:
    """Evaluate ``backend`` over every query in ``golden_set``.

    Returns ``(aggregate_metrics, per_query_outcomes)``. The aggregate is
    the arithmetic mean of each metric over the queries. The per-query
    list is empty when ``include_per_query`` is False.

    ``k`` is the number of neighbours the backend is asked for. The
    metrics are named ``recall_at_{k}``, ``ndcg_at_{k}``, and ``mrr``;
    ``ann_recall_vs_exact`` is added only when ``exact_backend`` is given.
    """
    if k <= 0:
        raise EvaluationError(f"k must be positive, got {k}")

    recall_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    fidelity_values: list[float] = []
    outcomes: list[QueryOutcome] = []

    for query in golden_set.queries:
        query_vec = encode_query(query.seed_item_ids, embedding_lookup=embedding_lookup)
        # Over-fetch to allow seed exclusion; see the docstring above.
        # The backend protocol does not carry an exclude list, so the
        # caller retrieves more and filters. This is also what the M3
        # serving layer will do (retrieve more, filter, truncate).
        seeds = set(query.seed_item_ids)
        fetch_k = k + len(seeds)
        pairs = backend.search(vector=query_vec, k=fetch_k)
        filtered = [(iid, s) for iid, s in pairs if iid not in seeds][:k]
        retrieved = [iid for iid, _ in filtered]
        relevant = set(query.relevant_item_ids)

        r = recall_at_k(retrieved, relevant, k)
        n = ndcg_at_k(retrieved, relevant, k)
        m = mrr(retrieved, relevant)
        recall_values.append(r)
        ndcg_values.append(n)
        mrr_values.append(m)

        fidelity: float | None = None
        if exact_backend is not None:
            exact_pairs = exact_backend.search(vector=query_vec, k=fetch_k)
            exact_filtered = [(iid, s) for iid, s in exact_pairs if iid not in seeds][:k]
            exact_ids = [iid for iid, _ in exact_filtered]
            fidelity = ann_recall_vs_exact(retrieved, exact_ids, k)
            fidelity_values.append(fidelity)

        if include_per_query:
            per_query_metrics: dict[str, float] = {
                f"recall_at_{k}": r,
                f"ndcg_at_{k}": n,
                "mrr": m,
            }
            if fidelity is not None:
                per_query_metrics["ann_recall_vs_exact"] = fidelity
            outcomes.append(
                QueryOutcome(
                    query_id=query.query_id,
                    retrieved=tuple(retrieved),
                    metrics=per_query_metrics,
                    ann_recall_vs_exact=fidelity,
                )
            )

    aggregate: dict[str, float] = {
        f"recall_at_{k}": _mean(recall_values),
        f"ndcg_at_{k}": _mean(ndcg_values),
        "mrr": _mean(mrr_values),
    }
    if exact_backend is not None:
        aggregate["ann_recall_vs_exact"] = _mean(fidelity_values)
    return aggregate, outcomes


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


# --------------------------------------------------------------------------- #
# report assembly
# --------------------------------------------------------------------------- #
def build_report(
    *,
    golden_set_version: str,
    commit: str,
    created_at: str,
    systems: Mapping[str, Mapping[str, float]],
    per_query: Mapping[str, Sequence[QueryOutcome]] | None = None,
) -> dict[str, Any]:
    """Assemble the report document the gate consumes.

    ``systems`` is ``{system_name: {metric_name: value}}``. ``per_query``,
    when given, is written alongside the aggregate for diagnostics; it is
    not read by the gate.
    """
    systems_doc: dict[str, Any] = {}
    for name, metrics in systems.items():
        entry: dict[str, Any] = {"metrics": dict(metrics)}
        if per_query is not None and name in per_query:
            entry["per_query"] = [
                {
                    "query_id": q.query_id,
                    "retrieved": list(q.retrieved),
                    "metrics": q.metrics,
                }
                for q in per_query[name]
            ]
        systems_doc[name] = entry

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "golden_set_version": golden_set_version,
        "commit": commit,
        "created_at": created_at,
        "systems": systems_doc,
    }


def write_report(path: pathlib.Path, report: Mapping[str, Any]) -> None:
    """Write the report as JSON, atomically (tempfile + rename).

    The gate reads the file; a partial write from a crash would look like
    a malformed report. Writing atomically means the reader either sees
    the old file or the complete new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "EmbeddingLookup",
    "EvaluationError",
    "QueryOutcome",
    "build_report",
    "encode_query",
    "evaluate_system",
    "write_report",
]
