#!/usr/bin/env python3
"""Run the offline evaluation and the CI gate.

    python scripts/eval.py
    python scripts/eval.py --k 10 --no-pgvector
    python scripts/eval.py --commit $(git rev-parse HEAD)

The script ties the evaluation pieces together:

1. Read the active embedding run (``artifacts/embeddings/current``).
2. Load the golden set (``evaluation/golden_set/v1.jsonl``).
3. Build the systems to evaluate:

   - ``random_baseline``     — random_baseline() from baselines.py
   - ``popularity_synthetic`` — popularity_baseline() with the synthetic provider
   - ``exact_knn``           — NumpyBackend.from_run_directory()
   - ``pgvector_hnsw``       — PgvectorBackend.from_registry(), when a
                               DATABASE_URL is available

4. Evaluate each system over the golden set.
5. Assemble the report and write it to ``evaluation/report.json``.
6. Load the thresholds and evaluate the gate.
7. Exit 0 if the gate passes, 1 if it fails, 2 if the evaluation could
   not run (missing run, missing golden set, ...). The distinction
   matters: ``1`` means "ran and did not meet the bar"; ``2`` means "did
   not run". The two have different remediation.

The report is also written when the gate fails (exit 1), so a failing
build has an artifact to inspect.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recsys.config.hot import HotConfigStore  # noqa: E402
from recsys.evaluation.baselines import (  # noqa: E402
    popularity_baseline,
    random_baseline,
)
from recsys.evaluation.golden_set import (  # noqa: E402
    GoldenSetError,
    load_golden_set,
)
from recsys.evaluation.runner import (  # noqa: E402
    EvaluationError,
    RerankArmResult,
    build_report,
    evaluate_rerank_arm,
    write_report,
)
from recsys.evaluation.thresholds import (  # noqa: E402
    ThresholdError,
    evaluate_gate,
    load_thresholds,
)
from recsys.retrieval.build import (  # noqa: E402
    BuildInputError,
    resolve_build_inputs,
)
from recsys.retrieval.numpy_backend import NumpyBackend  # noqa: E402
from recsys.retrieval.pgvector import PgvectorBackend  # noqa: E402
from recsys.retrieval.providers import (  # noqa: E402
    FrozenRecencyProvider,
    PgRecencyProvider,
    PopularityProvider,
    RecencyProvider,
    SyntheticPopularityProvider,
)
from recsys.retrieval.rerank import RerankConfig, Reranker, WeightedBlendReranker  # noqa: E402

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_COULD_NOT_RUN = 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _env_commit() -> str:
    """Return the commit under test.

    Checks several environment variables GitHub Actions and similar CI
    systems set (``GITHUB_SHA``, ``CI_COMMIT_SHA``, ``COMMIT_SHA``),
    falls back to ``git rev-parse HEAD``, and finally to ``"unknown"``.
    A value is considered valid only if it is a non-empty string after
    stripping. The previous version returned the empty string when
    ``GITHUB_SHA`` was set to ``""`` (which happens in some
    configurations), producing ``"commit": ""`` in the report and
    silently disabling the commit check in the gate.
    """
    for key in ("GITHUB_SHA", "CI_COMMIT_SHA", "COMMIT_SHA"):
        sha = os.environ.get(key, "").strip()
        if sha:
            return sha
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        # S603: the command is a list of trusted literals plus a git binary
        # resolved by shutil.which. No shell, no user input, no untrusted
        # argument. The suppression documents that this specific call site
        # was considered.
        out = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    sha = out.stdout.strip()
    return sha if sha else "unknown"


def _try_open_database() -> Any | None:
    """Open a psycopg connection if possible. Return None otherwise."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    try:
        import psycopg
    except ImportError:
        return None
    try:
        return psycopg.connect(url)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# evaluation of one system
# --------------------------------------------------------------------------- #
def _evaluate_vector_system(
    *,
    backend: Any,
    golden_set: Any,
    embedding_lookup: Any,
    exact_backend: Any,
    k: int,
) -> dict[str, float | None]:
    """Evaluate a search backend and return its aggregate metrics."""
    from recsys.evaluation.runner import evaluate_system

    metrics: dict[str, float | None] = dict(
        evaluate_system(
            backend=backend,
            golden_set=golden_set,
            embedding_lookup=embedding_lookup,
            k=k,
            exact_backend=exact_backend,
            include_per_query=False,
        )[0]
    )
    return metrics


def _evaluate_baseline(
    *,
    baseline_fn: Any,
    item_ids: list[str],
    golden_set: Any,
    k: int,
) -> dict[str, float | None]:
    """Evaluate a non-vector baseline (random, popularity).

    Baselines produce a ranked list per query without a query vector;
    metrics are computed directly from the ranking. ANN fidelity does
    not apply (a baseline is not a search over a vector index) and is
    omitted.
    """
    from recsys.evaluation.metrics import mrr, ndcg_at_k, recall_at_k

    recall_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    for q in golden_set.queries:
        result = baseline_fn(item_ids, k=k, query_id=q.query_id)
        retrieved = list(result.retrieved)
        relevant = set(q.relevant_item_ids)
        recall_values.append(recall_at_k(retrieved, relevant, k))
        ndcg_values.append(ndcg_at_k(retrieved, relevant, k))
        mrr_values.append(mrr(retrieved, relevant))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        f"recall_at_{k}": _mean(recall_values),
        f"ndcg_at_{k}": _mean(ndcg_values),
        "mrr": _mean(mrr_values),
    }


def _evaluate_popularity(
    *,
    item_ids: list[str],
    golden_set: Any,
    k: int,
) -> dict[str, float | None]:
    """Popularity baseline: same shape as _evaluate_baseline but without
    a query_id (popularity is query-independent)."""
    from recsys.evaluation.metrics import mrr, ndcg_at_k, recall_at_k

    recall_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    for q in golden_set.queries:
        result = popularity_baseline(item_ids, k=k)
        retrieved = list(result.retrieved)
        relevant = set(q.relevant_item_ids)
        recall_values.append(recall_at_k(retrieved, relevant, k))
        ndcg_values.append(ndcg_at_k(retrieved, relevant, k))
        mrr_values.append(mrr(retrieved, relevant))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        f"recall_at_{k}": _mean(recall_values),
        f"ndcg_at_{k}": _mean(ndcg_values),
        "mrr": _mean(mrr_values),
    }


# --------------------------------------------------------------------------- #
# rerank arms
# --------------------------------------------------------------------------- #
def _build_rerank_providers(
    conn: Any | None,
) -> tuple[PopularityProvider, RecencyProvider]:
    """Return the (popularity, recency) providers the arms will use.

    Popularity is always :class:`SyntheticPopularityProvider`, the
    placeholder the ADR ships until M5 lands an event-based provider.
    Its value is a hash of the item id, so an evaluation run is
    deterministic across processes.

    Recency is ``PgRecencyProvider`` when a database is reachable, and a
    ``FrozenRecencyProvider`` with ``default=0.0`` otherwise. The frozen
    provider returns the same age for every item; the recency signal is
    then uniform and does not change the ranking (ADR-0016 § Step 1,
    degenerate case). The arm still runs and its metrics are still
    produced; a reader knows recency was neutral because the ages were
    uniform, not because a provider failed.
    """
    popularity = SyntheticPopularityProvider()
    if conn is not None:
        recency: RecencyProvider = PgRecencyProvider(conn, timeout_seconds=2.0)
    else:
        recency = FrozenRecencyProvider({}, default=0.0)
    return popularity, recency


def _evaluate_rerank_arm_system(
    *,
    backend: Any,
    golden_set: Any,
    embedding_lookup: Any,
    exact_backend: Any,
    k: int,
    config: RerankConfig,
    reranker: Reranker,
    popularity_provider: PopularityProvider,
    recency_provider: RecencyProvider,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Evaluate one rerank arm and return ``(aggregate, status_info)``.

    ``status_info`` is written next to ``metrics`` in the report so a
    reader can tell "this arm scored low" from "this arm did not run".
    The gate does not read it; the summary table does.
    """
    result: RerankArmResult = evaluate_rerank_arm(
        backend=backend,
        golden_set=golden_set,
        embedding_lookup=embedding_lookup,
        k=k,
        config=config,
        reranker=reranker,
        popularity_provider=popularity_provider,
        recency_provider=recency_provider,
        exact_backend=exact_backend,
        include_per_query=False,
    )
    status_info: dict[str, Any] = {
        "status": result.status,
        "error_message": result.error_message,
    }
    return dict(result.aggregate), status_info


def _selected_arms(spec: str) -> set[str]:
    """Parse ``--arms`` into a set of arm names.

    ``"all"`` (the default) means every rerank arm. Otherwise a
    comma-separated list. The retrieval-only systems (``random_baseline``,
    ``popularity_synthetic``, ``exact_knn``, ``pgvector_hnsw``) always
    run; they are not "arms".
    """
    if spec.strip().lower() == "all":
        return {"blend", "blend_mmr", "exact_knn_blend"}
    out: set[str] = set()
    for raw in spec.split(","):
        name = raw.strip()
        if name:
            out.add(name)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--embeddings-root",
        type=pathlib.Path,
        default=REPO_ROOT / "artifacts" / "embeddings",
    )
    p.add_argument(
        "--golden-set",
        type=pathlib.Path,
        default=REPO_ROOT / "evaluation" / "golden_set" / "v1.jsonl",
    )
    p.add_argument(
        "--thresholds",
        type=pathlib.Path,
        default=REPO_ROOT / "evaluation" / "thresholds.yaml",
    )
    p.add_argument(
        "--report",
        type=pathlib.Path,
        default=REPO_ROOT / "evaluation" / "report.json",
    )
    p.add_argument(
        "--catalog",
        type=pathlib.Path,
        default=REPO_ROOT / "data" / "sample" / "catalog.jsonl",
        help="catalog file used to enumerate item ids for the baselines",
    )
    p.add_argument(
        "--no-pgvector",
        action="store_true",
        help="skip the pgvector system even if a database is reachable",
    )
    p.add_argument(
        "--arms",
        default="all",
        help=(
            "rerank arms: 'all' (default) or a comma-separated subset of "
            "blend,blend_mmr,exact_knn_blend. Retrieval-only systems always run."
        ),
    )
    p.add_argument(
        "--hot-config",
        type=pathlib.Path,
        default=REPO_ROOT / "config" / "hot.yaml",
        help="hot config file; supplies the reranker weights and MMR settings",
    )
    p.add_argument(
        "--commit",
        default=None,
        help="commit under test (default: $GITHUB_SHA or `git rev-parse HEAD`)",
    )
    p.add_argument(
        "--max-report-age-hours",
        type=float,
        default=24.0,
        help="freshness window for the gate",
    )
    p.add_argument(
        "--no-freshness-check",
        action="store_true",
        help="skip the report freshness check (useful for local runs)",
    )
    return p


def _num(value: Any) -> float:
    """Return ``value`` as a float, or 0.0 when it is None.

    A ``None`` metric is "does not apply" (ADR-0016 § Step 3). The
    summary shows it as 0.0 for column alignment; the report carries the
    real value for the gate.
    """
    return float(value) if isinstance(value, (int, float)) else 0.0


def _print_metrics_summary(
    systems: dict[str, dict[str, float | None]],
    *,
    k: int,
    system_status: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Print a compact table of metrics to stderr.

    The CI log shows the numbers directly, so a reviewer does not need to
    download the report artifact to see what the gate compared against.
    Stderr, not stdout: stdout is reserved for the single-line JSON
    events the caller may parse.

    ``ann_pre`` is the pre-rerank ANN fidelity. It equals
    ``ann_recall_vs_exact`` for a retrieval-only system and is the
    retrieval stage's number for a rerank arm. ``status`` is the arm's
    ``ok`` / ``degraded`` / ``error`` state; empty for a retrieval-only
    system.
    """
    recall_key = f"recall_at_{k}"
    ndcg_key = f"ndcg_at_{k}"
    headers = ("system", recall_key, ndcg_key, "mrr", "ann_pre", "status")
    rows: list[tuple[str, str, str, str, str, str]] = []
    status_map = system_status or {}
    for name in sorted(systems):
        m = systems[name]
        pre = m.get("ann_recall_vs_exact_pre_rerank")
        if pre is None and "ann_recall_vs_exact" in m and m["ann_recall_vs_exact"] is not None:
            pre = m["ann_recall_vs_exact"]
        pre_str = f"{pre:.4f}" if isinstance(pre, (int, float)) else "n/a"
        status = status_map.get(name, {}).get("status", "")
        rows.append(
            (
                name,
                f"{_num(m.get(recall_key)):.4f}",
                f"{_num(m.get(ndcg_key)):.4f}",
                f"{_num(m.get('mrr')):.4f}",
                pre_str,
                status,
            )
        )
    widths = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(6)]
    sep = "  ".join("-" * w for w in widths)
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines = ["", "=== evaluation metrics ===", header_line, sep]
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(6)))
    lines.append("")
    print("\n".join(lines), file=sys.stderr)


def _read_catalog_item_ids(path: pathlib.Path) -> list[str]:
    """Read the item ids from a JSONL catalog."""
    ids: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        ids.append(str(obj["item_id"]))
    return ids


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)

    # ---- Inputs -------------------------------------------------------- #
    try:
        inputs = resolve_build_inputs(args.embeddings_root)
    except BuildInputError as e:
        print(json.dumps({"event": "eval.input_error", "message": str(e)}))
        return EXIT_COULD_NOT_RUN

    try:
        catalog_ids = _read_catalog_item_ids(args.catalog)
    except (OSError, KeyError, json.JSONDecodeError) as e:
        print(json.dumps({"event": "eval.catalog_error", "message": str(e)}))
        return EXIT_COULD_NOT_RUN

    try:
        golden_set = load_golden_set(args.golden_set, known_item_ids=catalog_ids)
    except GoldenSetError as e:
        print(json.dumps({"event": "eval.golden_set_error", "message": str(e)}))
        return EXIT_COULD_NOT_RUN

    # ---- Reranker config (hot config) ---------------------------------- #
    # Weights, MMR settings, and `enable_mmr` come from config/hot.yaml.
    # A missing or malformed file is a runtime error: there is no report
    # to compare, so this is not a gate failure.
    try:
        hot_store = HotConfigStore.from_path(args.hot_config)
        rerank_config = hot_store.get().rerank.to_rerank_config()
    except Exception as e:
        print(json.dumps({"event": "eval.hot_config_error", "message": str(e)}))
        return EXIT_COULD_NOT_RUN

    # Two rerankers: one with MMR off, one on. The arm name is the
    # report key; the config each arm uses decides whether MMR runs.
    config_no_mmr = RerankConfig(
        w_sim=rerank_config.w_sim,
        w_pop=rerank_config.w_pop,
        w_rec=rerank_config.w_rec,
        mmr_lambda=rerank_config.mmr_lambda,
        mmr_window=rerank_config.mmr_window,
        mmr_min_k=rerank_config.mmr_min_k,
        candidate_multiplier=rerank_config.candidate_multiplier,
        recency_half_life_days=rerank_config.recency_half_life_days,
        enable_mmr=False,
    )
    config_with_mmr = RerankConfig(
        w_sim=rerank_config.w_sim,
        w_pop=rerank_config.w_pop,
        w_rec=rerank_config.w_rec,
        mmr_lambda=rerank_config.mmr_lambda,
        mmr_window=rerank_config.mmr_window,
        mmr_min_k=rerank_config.mmr_min_k,
        candidate_multiplier=rerank_config.candidate_multiplier,
        recency_half_life_days=rerank_config.recency_half_life_days,
        enable_mmr=True,
    )
    reranker_no_mmr = WeightedBlendReranker(config_no_mmr)
    reranker_with_mmr = WeightedBlendReranker(config_with_mmr)
    arms_to_run = _selected_arms(args.arms)

    # ---- Systems ------------------------------------------------------- #
    exact_backend = NumpyBackend.from_run_directory(inputs.run_dir)
    embedding_lookup = exact_backend.embedding_for

    systems: dict[str, dict[str, float | None]] = {}
    system_status: dict[str, dict[str, Any]] = {}
    required_systems: list[str] = []

    systems["random_baseline"] = _evaluate_baseline(
        baseline_fn=random_baseline,
        item_ids=catalog_ids,
        golden_set=golden_set,
        k=args.k,
    )
    required_systems.append("random_baseline")

    systems["popularity_synthetic"] = _evaluate_popularity(
        item_ids=catalog_ids,
        golden_set=golden_set,
        k=args.k,
    )
    required_systems.append("popularity_synthetic")

    systems["exact_knn"] = _evaluate_vector_system(
        backend=exact_backend,
        golden_set=golden_set,
        embedding_lookup=embedding_lookup,
        exact_backend=exact_backend,
        k=args.k,
    )
    required_systems.append("exact_knn")

    # Providers are built once and shared by every arm.
    conn = None if args.no_pgvector else _try_open_database()
    popularity_provider, recency_provider = _build_rerank_providers(conn)

    # exact_knn_blend (diagnostic, informational): an upper bound on what
    # the reranker can do without an ANN approximation in the way.
    if "exact_knn_blend" in arms_to_run:
        metrics, status = _evaluate_rerank_arm_system(
            backend=exact_backend,
            golden_set=golden_set,
            embedding_lookup=embedding_lookup,
            exact_backend=exact_backend,
            k=args.k,
            config=config_no_mmr,
            reranker=reranker_no_mmr,
            popularity_provider=popularity_provider,
            recency_provider=recency_provider,
        )
        systems["exact_knn_blend"] = metrics
        system_status["exact_knn_blend"] = status

    # pgvector_hnsw (retrieval only) and the two pgvector arms.
    if conn is not None:
        try:
            pg_backend = PgvectorBackend.from_registry(conn, hnsw_ef_search=100)
            systems["pgvector_hnsw"] = _evaluate_vector_system(
                backend=pg_backend,
                golden_set=golden_set,
                embedding_lookup=embedding_lookup,
                exact_backend=exact_backend,
                k=args.k,
            )
            required_systems.append("pgvector_hnsw")

            if "blend" in arms_to_run:
                metrics, status = _evaluate_rerank_arm_system(
                    backend=pg_backend,
                    golden_set=golden_set,
                    embedding_lookup=embedding_lookup,
                    exact_backend=exact_backend,
                    k=args.k,
                    config=config_no_mmr,
                    reranker=reranker_no_mmr,
                    popularity_provider=popularity_provider,
                    recency_provider=recency_provider,
                )
                systems["pgvector_hnsw_blend"] = metrics
                system_status["pgvector_hnsw_blend"] = status

            if "blend_mmr" in arms_to_run:
                metrics, status = _evaluate_rerank_arm_system(
                    backend=pg_backend,
                    golden_set=golden_set,
                    embedding_lookup=embedding_lookup,
                    exact_backend=exact_backend,
                    k=args.k,
                    config=config_with_mmr,
                    reranker=reranker_with_mmr,
                    popularity_provider=popularity_provider,
                    recency_provider=recency_provider,
                )
                systems["pgvector_hnsw_blend_mmr"] = metrics
                system_status["pgvector_hnsw_blend_mmr"] = status
        except EvaluationError as e:
            print(json.dumps({"event": "eval.pgvector_skipped", "message": str(e)}))
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    # ---- Report -------------------------------------------------------- #
    commit = args.commit if args.commit is not None else _env_commit()
    created_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = build_report(
        golden_set_version=golden_set.version,
        commit=commit,
        created_at=created_at,
        systems=systems,
        system_status=system_status,
    )
    write_report(args.report, report)
    print(
        json.dumps(
            {
                "event": "eval.report_written",
                "path": str(args.report),
                "commit": commit,
                "systems": sorted(systems),
            }
        )
    )

    # ---- Metrics summary (human-readable, on stderr) ------------------- #
    _print_metrics_summary(systems, k=args.k, system_status=system_status)

    # ---- Gate ---------------------------------------------------------- #
    # Exit-code contract:
    #   0  gate passed
    #   1  gate failed (a metric is below threshold, or a version/commit/
    #      freshness check failed)
    #   2  the evaluation could not run (missing inputs, malformed
    #      thresholds, or an arm in the gate section that returned
    #      status="error")
    try:
        thresholds = load_thresholds(args.thresholds)
    except ThresholdError as e:
        print(json.dumps({"event": "eval.thresholds_error", "message": str(e)}))
        return EXIT_COULD_NOT_RUN

    # An arm in the gate section whose status is "error" is a runtime
    # error, not a gate failure: there is no number to compare. An arm in
    # informational whose status is "error" is logged but does not change
    # the exit code.
    for name, info in system_status.items():
        if info.get("status") == "error" and name in thresholds.per_system:
            print(
                json.dumps(
                    {
                        "event": "eval.arm_error",
                        "system": name,
                        "message": info.get("error_message"),
                    }
                )
            )
            return EXIT_COULD_NOT_RUN

    now = None if args.no_freshness_check else dt.datetime.now(dt.UTC)
    result = evaluate_gate(
        thresholds,
        report,
        current_commit=commit,
        now=now,
        max_report_age_hours=args.max_report_age_hours,
        required_systems=required_systems,
    )

    if result.passed:
        print(json.dumps({"event": "eval.gate_passed"}))
        return EXIT_OK

    print(
        json.dumps(
            {
                "event": "eval.gate_failed",
                "reason": result.reason,
                "failures": [
                    {
                        "system": f.system,
                        "metric": f.metric,
                        "value": f.value,
                        "threshold": f.threshold,
                        "kind": f.kind,
                    }
                    for f in result.failures
                ],
            }
        )
    )
    return EXIT_GATE_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
