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
from recsys.retrieval.providers import PopularityProvider, RecencyProvider
from recsys.retrieval.rerank import Candidate, RerankConfig, Reranker, RerankError

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


@dataclass(frozen=True)
class RerankArmResult:
    """Result of evaluating one rerank arm over the golden set.

    ``status`` is one of:

    - ``"ok"`` — the arm ran and every signal provider answered every
      query.
    - ``"degraded"`` — the arm ran, but at least one provider was
      unavailable for the whole golden set (every query ended up with
      that signal neutral). The metrics are still produced, but the
      signal that was supposed to distinguish candidates did not.
      A reader must not treat a degraded arm's numbers as fully
      informed.
    - ``"error"`` — the arm could not run. ``aggregate`` is empty and
      ``error_message`` carries the reason. The caller (the script)
      records this and, when the arm is in the gate section of the
      thresholds file, fails the build. Silent skips are not permitted:
      an arm that could not run is not the same as an arm that passed.

    ``aggregate`` values are ``float | None``. ``None`` means "this
    metric does not apply to this arm": ``ann_recall_vs_exact`` is null
    post-rerank because the reranker deliberately changes the top-k
    (ADR-0016 § Step 3), and the pre-rerank value is reported separately
    in ``ann_recall_vs_exact_pre_rerank`` so the retrieval stage stays
    measurable. The gate skips a null metric; it does not skip a metric
    that is absent from the report.

    ``provider_missing_rate_popularity`` and
    ``provider_missing_rate_recency`` are the fraction of queries whose
    signal came from a provider that raised or timed out. They are the
    number that turns "degraded" into something an operator can act on.
    """

    aggregate: dict[str, float | None]
    outcomes: tuple[QueryOutcome, ...]
    status: str
    error_message: str | None = None


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
# rerank arm
# --------------------------------------------------------------------------- #
def _pick_search_fn(
    backend: IndexBackend,
    *,
    needs_vectors: bool,
) -> Callable[..., Any]:
    """Return the backend method the arm should call.

    ``search`` when the arm does not need the candidate vectors; the
    backend's ``search_with_vectors`` when it does. A backend without
    the optional method cannot serve a vector-needing arm; the caller
    turns that into a ``status="error"`` result, not a crash, so a
    misconfigured arm is visible in the report as an error rather than
    taking down the whole evaluation.
    """
    if not needs_vectors:
        return backend.search
    fn = getattr(backend, "search_with_vectors", None)
    if fn is None:
        raise EvaluationError(
            f"backend {backend.name!r} has no search_with_vectors; MMR needs the candidate vectors"
        )
    return fn  # type: ignore[no-any-return]


def _safe_provider_get(
    provider: Any,
    ids: list[str],
) -> tuple[dict[str, float], bool]:
    """Call ``provider.get(ids)``; return ``({}, False)`` on failure.

    Any exception counts as a failure. The provider contract is "raise"
    (see ``recsys.retrieval.providers``): a timeout, a connection error,
    a batch-too-large error, and a logic error are all the same to the
    re-ranker (ADR-0016 § Failure behavior) — the signal is neutral for
    this query. The caller records the missing rate per signal so a
    provider outage is visible in the report.
    """
    if not ids:
        return {}, True
    try:
        result = provider.get(ids)
    except Exception:
        return {}, False
    return {str(k): float(v) for k, v in dict(result).items()}, True


def evaluate_rerank_arm(
    *,
    backend: IndexBackend,
    golden_set: GoldenSet,
    embedding_lookup: EmbeddingLookup,
    k: int,
    config: RerankConfig,
    reranker: Reranker,
    popularity_provider: PopularityProvider,
    recency_provider: RecencyProvider,
    exact_backend: IndexBackend | None = None,
    include_per_query: bool = False,
) -> RerankArmResult:
    """Evaluate one rerank arm over every query in ``golden_set``.

    The pipeline per query:

    1. Encode the query vector from its seed embeddings.
    2. Retrieve ``candidate_k = k × candidate_multiplier`` candidates.
       ``search_with_vectors`` is used when MMR will run; ``search``
       otherwise. The choice mirrors ``mmr_status``: MMR is off when the
       config disables it, and skipped when ``k < mmr_min_k`` because
       diversity has no room below that threshold (ADR-0016 § Step 3).
    3. Fetch popularity and recency in batch. A provider that raises or
       times out is neutral for that signal on that query; the caller
       records the missing rate.
    4. Build the ``Candidate`` list, then call ``reranker.rerank``. A
       ``RerankError`` means the re-ranker failed on this query; the ADR
       says the retrieval result is returned. The arm continues.
    5. Exclude seeds, truncate to ``k``, compute metrics.

    Two fidelity numbers are reported:

    - ``ann_recall_vs_exact`` — always ``None``. The reranker's output
      deliberately differs from exact kNN (that is what diversification
      is); computing fidelity post-rerank would measure the wrong
      thing.
    - ``ann_recall_vs_exact_pre_rerank`` — the fidelity of the
      retrieval stage, computed the same way ``evaluate_system`` does
      for a retrieval-only system. This is what makes the retrieval
      stage still measurable under an arm that reranks.
    """
    if k <= 0:
        raise EvaluationError(f"k must be positive, got {k}")

    n_queries = len(golden_set.queries)
    if n_queries == 0:
        return RerankArmResult(
            aggregate={},
            outcomes=(),
            status="error",
            error_message="golden set is empty",
        )

    candidate_k = k * config.candidate_multiplier
    needs_vectors = config.enable_mmr and k >= config.mmr_min_k

    try:
        search_fn = _pick_search_fn(backend, needs_vectors=needs_vectors)
    except EvaluationError as exc:
        return RerankArmResult(
            aggregate={},
            outcomes=(),
            status="error",
            error_message=str(exc),
        )

    recall_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    pre_fidelity_values: list[float] = []
    popularity_missing = 0
    recency_missing = 0
    outcomes: list[QueryOutcome] = []

    for query in golden_set.queries:
        query_vec = encode_query(query.seed_item_ids, embedding_lookup=embedding_lookup)
        seeds = set(query.seed_item_ids)

        raw = search_fn(vector=query_vec, k=candidate_k)
        pairs: list[tuple[str, float]]
        vecs: dict[str, NDArray[np.float32]]
        if needs_vectors:
            pairs = [(str(iid), float(score)) for iid, score, _v in raw]
            vecs = {str(iid): v for iid, _s, v in raw}
        else:
            pairs = [(str(iid), float(s)) for iid, s in raw]
            vecs = {}

        # Pre-rerank fidelity, computed exactly like evaluate_system does
        # for a retrieval-only system. It is the arm's window into the
        # retrieval stage under reranking.
        pre_fidelity: float | None = None
        if exact_backend is not None:
            exact_pairs = exact_backend.search(vector=query_vec, k=k + len(seeds))
            exact_filtered = [(str(iid), s) for iid, s in exact_pairs if iid not in seeds][:k]
            exact_ids = [iid for iid, _ in exact_filtered]
            pre_filtered = [(iid, s) for iid, s in pairs if iid not in seeds][:k]
            pre_ids = [iid for iid, _ in pre_filtered]
            pre_fidelity = ann_recall_vs_exact(pre_ids, exact_ids, k)
            pre_fidelity_values.append(pre_fidelity)

        # Signal providers: batch per query. Failure is neutral; the
        # rate is recorded so a provider outage is visible.
        ids = [iid for iid, _ in pairs]
        pop_map, pop_ok = _safe_provider_get(popularity_provider, ids)
        rec_map, rec_ok = _safe_provider_get(recency_provider, ids)
        if not pop_ok:
            popularity_missing += 1
        if not rec_ok:
            recency_missing += 1

        missing: set[str] = set()
        if not pop_ok:
            missing.add("popularity")
        if not rec_ok:
            missing.add("recency")

        candidates = [
            Candidate(
                item_id=iid,
                similarity=score,
                popularity=pop_map.get(iid, 0.0),
                age_days=rec_map.get(iid, 0.0),
                vector=vecs.get(iid),
            )
            for iid, score in pairs
        ]

        try:
            ranked = reranker.rerank(
                candidates=candidates,
                k=k + len(seeds),
                missing_signals=frozenset(missing),
            )
        except RerankError:
            # ADR-0016 § Failure behavior: return the retrieval result.
            ranked = pairs

        filtered = [(iid, s) for iid, s in ranked if iid not in seeds][:k]
        retrieved = [iid for iid, _ in filtered]
        relevant = set(query.relevant_item_ids)

        r = recall_at_k(retrieved, relevant, k)
        n = ndcg_at_k(retrieved, relevant, k)
        m = mrr(retrieved, relevant)
        recall_values.append(r)
        ndcg_values.append(n)
        mrr_values.append(m)

        if include_per_query:
            per_query_metrics: dict[str, float] = {
                f"recall_at_{k}": r,
                f"ndcg_at_{k}": n,
                "mrr": m,
            }
            if pre_fidelity is not None:
                per_query_metrics["ann_recall_vs_exact_pre_rerank"] = pre_fidelity
            outcomes.append(
                QueryOutcome(
                    query_id=query.query_id,
                    retrieved=tuple(retrieved),
                    metrics=per_query_metrics,
                    ann_recall_vs_exact=None,
                )
            )

    aggregate: dict[str, float | None] = {
        f"recall_at_{k}": _mean(recall_values),
        f"ndcg_at_{k}": _mean(ndcg_values),
        "mrr": _mean(mrr_values),
        "ann_recall_vs_exact": None,
        "provider_missing_rate_popularity": popularity_missing / n_queries,
        "provider_missing_rate_recency": recency_missing / n_queries,
    }
    if pre_fidelity_values:
        aggregate["ann_recall_vs_exact_pre_rerank"] = _mean(pre_fidelity_values)

    # "degraded" when a provider was unavailable for the whole set: a
    # signal that should distinguish candidates did not. Partial outages
    # leave the arm "ok" and are visible in provider_missing_rate_*.
    status = "ok"
    if popularity_missing == n_queries or recency_missing == n_queries:
        status = "degraded"

    return RerankArmResult(
        aggregate=aggregate,
        outcomes=tuple(outcomes),
        status=status,
    )


# --------------------------------------------------------------------------- #
# report assembly
# --------------------------------------------------------------------------- #
def build_report(
    *,
    golden_set_version: str,
    commit: str,
    created_at: str,
    systems: Mapping[str, Mapping[str, float | None]],
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
    "RerankArmResult",
    "build_report",
    "encode_query",
    "evaluate_rerank_arm",
    "evaluate_system",
    "write_report",
]
