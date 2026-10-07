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
    build_report,
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

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_COULD_NOT_RUN = 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _env_commit() -> str:
    """Return the commit under test.

    Prefers ``GITHUB_SHA`` (set by GitHub Actions), falls back to
    ``git rev-parse HEAD``, and finally to ``"unknown"``.
    """
    sha = os.environ.get("GITHUB_SHA")
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
    return out.stdout.strip()


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
) -> dict[str, float]:
    """Evaluate a search backend and return its aggregate metrics."""
    from recsys.evaluation.runner import evaluate_system

    metrics, _ = evaluate_system(
        backend=backend,
        golden_set=golden_set,
        embedding_lookup=embedding_lookup,
        k=k,
        exact_backend=exact_backend,
        include_per_query=False,
    )
    return metrics


def _evaluate_baseline(
    *,
    baseline_fn: Any,
    item_ids: list[str],
    golden_set: Any,
    k: int,
) -> dict[str, float]:
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
) -> dict[str, float]:
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

    # ---- Systems ------------------------------------------------------- #
    exact_backend = NumpyBackend.from_run_directory(inputs.run_dir)
    embedding_lookup = exact_backend.embedding_for

    systems: dict[str, dict[str, float]] = {}
    required_systems: list[str] = []

    # Random
    systems["random_baseline"] = _evaluate_baseline(
        baseline_fn=random_baseline,
        item_ids=catalog_ids,
        golden_set=golden_set,
        k=args.k,
    )
    required_systems.append("random_baseline")

    # Popularity
    systems["popularity_synthetic"] = _evaluate_popularity(
        item_ids=catalog_ids,
        golden_set=golden_set,
        k=args.k,
    )
    required_systems.append("popularity_synthetic")

    # Exact kNN
    systems["exact_knn"] = _evaluate_vector_system(
        backend=exact_backend,
        golden_set=golden_set,
        embedding_lookup=embedding_lookup,
        exact_backend=exact_backend,
        k=args.k,
    )
    required_systems.append("exact_knn")

    # pgvector (optional)
    if not args.no_pgvector:
        conn = _try_open_database()
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
            except EvaluationError as e:
                print(json.dumps({"event": "eval.pgvector_skipped", "message": str(e)}))
            finally:
                # Best-effort close: if the connection is already broken,
                # releasing it is the OS's job and there is nothing
                # useful to do here.
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

    # ---- Gate ---------------------------------------------------------- #
    try:
        thresholds = load_thresholds(args.thresholds)
    except ThresholdError as e:
        print(json.dumps({"event": "eval.thresholds_error", "message": str(e)}))
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
