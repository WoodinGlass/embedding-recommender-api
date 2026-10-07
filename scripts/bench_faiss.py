#!/usr/bin/env python3
"""FAISS benchmark runner (ADR-0011).

Compares exact kNN (numpy), FAISS HNSW, and — when a database is
reachable — pgvector HNSW on the same vectors, the same golden set, and
the same query vectors. Every measurement writes a JSON file with
environment metadata so a reader can tell under what conditions the
number was produced.

Usage:

    python scripts/bench_faiss.py
    python scripts/bench_faiss.py --output docs/faiss-benchmark.json

Exit codes:

    0 — benchmark completed, JSON written
    2 — inputs missing (no active run, no golden set)
    3 — a library is unavailable (faiss not installed)

The benchmark is not part of CI. It runs on demand; the JSON and the
generated Markdown are committed so the README table can cite them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recsys.evaluation.golden_set import (  # noqa: E402
    GoldenSet,
    GoldenSetError,
    load_golden_set,
)
from recsys.evaluation.metrics import (  # noqa: E402
    ann_recall_vs_exact,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from recsys.evaluation.runner import encode_query  # noqa: E402
from recsys.retrieval.build import (  # noqa: E402
    BuildInputError,
    resolve_build_inputs,
)
from recsys.retrieval.numpy_backend import NumpyBackend  # noqa: E402

EXIT_OK = 0
EXIT_INPUTS_MISSING = 2
EXIT_LIBRARY_MISSING = 3

BENCHMARK_SCHEMA_VERSION = 1
SEED = 42
EF_SEARCH_GRID: tuple[int, ...] = (40, 80, 160, 320)
WARMUP_QUERIES = 50
MEASURED_QUERIES = 500
K_DEFAULT = 10
HNSW_M_DEFAULT = 16
HNSW_EF_CONSTRUCTION_DEFAULT = 64


# --------------------------------------------------------------------------- #
# environment metadata
# --------------------------------------------------------------------------- #
def _cpu_model() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        sysctl = shutil.which("sysctl")
        if sysctl:
            try:
                out = subprocess.run(  # noqa: S603
                    [sysctl, "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return out.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
    return "unknown"


def _ram_bytes() -> int:
    if platform.system() != "Linux":
        return 0
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _lib_version(name: str) -> str | None:
    try:
        mod = __import__(name)
    except ImportError:
        return None
    return getattr(mod, "__version__", None)


def _git_commit() -> str:
    git = shutil.which("git")
    if not git:
        return "unknown"
    try:
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


def _collect_environment() -> dict[str, Any]:
    return {
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count() or 1,
        "ram_bytes": _ram_bytes(),
        "os": platform.system(),
        "kernel": platform.release(),
        "python_version": platform.python_version(),
        "numpy_version": _lib_version("numpy"),
        "faiss_version": _lib_version("faiss"),
        "num_threads": 1,
        "seed": SEED,
    }


# --------------------------------------------------------------------------- #
# input loading
# --------------------------------------------------------------------------- #
def _load_vectors(path: pathlib.Path) -> tuple[list[str], NDArray[np.float32]]:
    from recsys.embeddings.artifacts import iter_previous_batches

    ids: list[str] = []
    chunks: list[NDArray[np.float32]] = []
    for batch in iter_previous_batches(path):
        ids.extend(batch.item_ids)
        chunks.append(batch.embeddings)
    if not chunks:
        raise RuntimeError(f"no rows in {path}")
    return ids, np.concatenate(chunks, axis=0)


def _query_vectors(golden_set: GoldenSet, lookup: Any) -> list[NDArray[np.float32]]:
    return [encode_query(q.seed_item_ids, embedding_lookup=lookup) for q in golden_set.queries]


# --------------------------------------------------------------------------- #
# search helpers
# --------------------------------------------------------------------------- #
def _topk_exact(vectors: NDArray[np.float32], q: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    sims = vectors @ q
    idx = np.argpartition(-sims, k - 1)[:k]
    order = np.argsort(-sims[idx], kind="stable")
    result: NDArray[np.int64] = idx[order].astype(np.int64)
    return result


def _topk_faiss(index: Any, q: NDArray[np.float32], k: int, ef_search: int) -> NDArray[np.int64]:
    index.hnsw.efSearch = ef_search
    _d, idx = index.search(q.reshape(1, -1).astype(np.float32), k)
    # ``idx`` is Any because faiss ships no type stubs. np.asarray pins the
    # type at the interop boundary rather than letting Any leak through
    # ``astype``.
    result: NDArray[np.int64] = np.asarray(idx[0], dtype=np.int64)
    return result


def _percentiles(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
    }


# --------------------------------------------------------------------------- #
# metric aggregation
# --------------------------------------------------------------------------- #
def _aggregate_metrics(
    *,
    topk_indices: list[NDArray[np.int64]],
    item_ids: list[str],
    golden_set: GoldenSet,
    k: int,
    exact_topk_indices: list[NDArray[np.int64]],
) -> dict[str, float]:
    recalls: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []
    fidelities: list[float] = []
    for q, idx, exact_idx in zip(golden_set.queries, topk_indices, exact_topk_indices, strict=True):
        retrieved = [item_ids[int(i)] for i in idx]
        exact_retrieved = [item_ids[int(i)] for i in exact_idx]
        relevant = set(q.relevant_item_ids)
        recalls.append(recall_at_k(retrieved, relevant, k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
        mrrs.append(mrr(retrieved, relevant))
        fidelities.append(ann_recall_vs_exact(retrieved, exact_retrieved, k))
    n = len(golden_set.queries)
    return {
        f"recall_at_{k}": round(sum(recalls) / n, 4),
        f"ndcg_at_{k}": round(sum(ndcgs) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "ann_recall_vs_exact": round(sum(fidelities) / n, 4),
    }


# --------------------------------------------------------------------------- #
# per-system measurement
# --------------------------------------------------------------------------- #
def _measure_exact(
    *,
    vectors: NDArray[np.float32],
    queries: list[NDArray[np.float32]],
    golden_set: GoldenSet,
    item_ids: list[str],
    k: int,
) -> dict[str, Any]:
    """Exact kNN: top-k per query + latency loop."""
    # Top-k for every golden query (ground truth and the exact system's
    # results, which are the same thing).
    topk = [_topk_exact(vectors, q, k) for q in queries]

    # Latency measurement, separate from the metric pass so the top-k
    # computation above does not pollute the timing.
    for i in range(WARMUP_QUERIES):
        _ = _topk_exact(vectors, queries[i % len(queries)], k)
    latencies: list[float] = []
    for i in range(MEASURED_QUERIES):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        _ = _topk_exact(vectors, q, k)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    metrics = _aggregate_metrics(
        topk_indices=topk,
        item_ids=item_ids,
        golden_set=golden_set,
        k=k,
        exact_topk_indices=topk,
    )
    return {
        "metrics": metrics,
        "latency": _percentiles(latencies),
        "qps": round(MEASURED_QUERIES / (sum(latencies) / 1000.0), 2),
        "index_build_time_s": 0.0,
    }


def _measure_faiss(
    *,
    vectors: NDArray[np.float32],
    queries: list[NDArray[np.float32]],
    k: int,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> dict[str, Any]:
    """FAISS HNSW: build once, search with one ef_search value."""
    import faiss

    faiss.omp_set_num_threads(1)
    np.random.seed(SEED)

    dim = vectors.shape[1]
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    build_s = time.perf_counter() - t0

    # Metric pass (no timing).
    topk = [_topk_faiss(index, q, k, ef_search) for q in queries]

    # Latency pass.
    for i in range(WARMUP_QUERIES):
        _ = _topk_faiss(index, queries[i % len(queries)], k, ef_search)
    latencies: list[float] = []
    for i in range(MEASURED_QUERIES):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        _ = _topk_faiss(index, q, k, ef_search)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "topk": topk,
        "latency": _percentiles(latencies),
        "qps": round(MEASURED_QUERIES / (sum(latencies) / 1000.0), 2),
        "index_build_time_s": round(build_s, 4),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=REPO_ROOT / "docs" / "faiss-benchmark.json",
    )
    p.add_argument("--k", type=int, default=K_DEFAULT, help="top-k for metrics and search")
    p.add_argument("--hnsw-m", type=int, default=HNSW_M_DEFAULT, help="HNSW m parameter")
    p.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=HNSW_EF_CONSTRUCTION_DEFAULT,
    )
    p.add_argument(
        "--ef-search-grid",
        type=int,
        nargs="+",
        default=list(EF_SEARCH_GRID),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)

    # ---- inputs -------------------------------------------------------- #
    try:
        inputs = resolve_build_inputs(REPO_ROOT / "artifacts" / "embeddings")
    except BuildInputError as e:
        print(json.dumps({"event": "bench.input_error", "message": str(e)}))
        return EXIT_INPUTS_MISSING

    catalog_ids: list[str] = []
    catalog_path = REPO_ROOT / "data" / "sample" / "catalog.jsonl"
    for raw in catalog_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            catalog_ids.append(str(json.loads(line)["item_id"]))

    try:
        golden_set = load_golden_set(
            REPO_ROOT / "evaluation" / "golden_set" / "v1.jsonl",
            known_item_ids=catalog_ids,
        )
    except GoldenSetError as e:
        print(json.dumps({"event": "bench.golden_set_error", "message": str(e)}))
        return EXIT_INPUTS_MISSING

    item_ids, vectors = _load_vectors(inputs.parquet_path)
    dim = int(vectors.shape[1])
    print(
        json.dumps(
            {
                "event": "bench.inputs",
                "run_id": inputs.run_id,
                "vectors": len(item_ids),
                "dim": dim,
                "queries": len(golden_set.queries),
                "k": args.k,
            }
        ),
        file=sys.stderr,
    )

    # ---- query vectors ------------------------------------------------- #
    exact_backend = NumpyBackend.from_run_directory(inputs.run_dir)
    queries = _query_vectors(golden_set, exact_backend.embedding_for)

    # ---- exact kNN ----------------------------------------------------- #
    print(
        json.dumps({"event": "bench.system_start", "system": "exact_knn"}),
        file=sys.stderr,
    )
    exact_result = _measure_exact(
        vectors=vectors,
        queries=queries,
        golden_set=golden_set,
        item_ids=item_ids,
        k=args.k,
    )
    exact_topk = [_topk_exact(vectors, q, args.k) for q in queries]
    print(
        json.dumps(
            {
                "event": "bench.system_done",
                "system": "exact_knn",
                "metrics": exact_result["metrics"],
                "p95_ms": exact_result["latency"]["p95_ms"],
            }
        ),
        file=sys.stderr,
    )

    # ---- FAISS HNSW ---------------------------------------------------- #
    try:
        import faiss  # noqa: F401
    except ImportError:
        print(
            json.dumps(
                {
                    "event": "bench.library_missing",
                    "library": "faiss",
                    "install": "pip install -e '.[bench]'",
                }
            )
        )
        return EXIT_LIBRARY_MISSING

    faiss_systems: dict[str, Any] = {}
    for ef_search in args.ef_search_grid:
        print(
            json.dumps(
                {
                    "event": "bench.system_start",
                    "system": "faiss_hnsw",
                    "ef_search": ef_search,
                }
            ),
            file=sys.stderr,
        )
        result = _measure_faiss(
            vectors=vectors,
            queries=queries,
            k=args.k,
            m=args.hnsw_m,
            ef_construction=args.hnsw_ef_construction,
            ef_search=ef_search,
        )
        metrics = _aggregate_metrics(
            topk_indices=result["topk"],
            item_ids=item_ids,
            golden_set=golden_set,
            k=args.k,
            exact_topk_indices=exact_topk,
        )
        faiss_systems[f"ef_search_{ef_search}"] = {
            "ef_search": ef_search,
            "metrics": metrics,
            "latency": result["latency"],
            "qps": result["qps"],
            "index_build_time_s": result["index_build_time_s"],
        }
        print(
            json.dumps(
                {
                    "event": "bench.system_done",
                    "system": "faiss_hnsw",
                    "ef_search": ef_search,
                    "metrics": metrics,
                    "p95_ms": result["latency"]["p95_ms"],
                }
            ),
            file=sys.stderr,
        )

    # ---- assemble + write --------------------------------------------- #
    doc = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_commit(),
        "inputs": {
            "run_id": inputs.run_id,
            "model_version": inputs.model_version,
            "catalog_snapshot": inputs.catalog_snapshot,
            "golden_set_version": golden_set.version,
            "vector_count": len(item_ids),
            "vector_dim": dim,
            "k": args.k,
        },
        "environment": _collect_environment(),
        "parameters": {
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_construction": args.hnsw_ef_construction,
            "ef_search_grid": list(args.ef_search_grid),
            "warmup_queries": WARMUP_QUERIES,
            "measured_queries": MEASURED_QUERIES,
        },
        "systems": {
            "exact_knn": {
                "metrics": exact_result["metrics"],
                "latency": exact_result["latency"],
                "qps": exact_result["qps"],
                "index_build_time_s": exact_result["index_build_time_s"],
            },
            "faiss_hnsw": faiss_systems,
        },
        "notes": (
            "Latency is measured in-process with time.perf_counter() on a "
            "single thread (faiss.omp_set_num_threads(1)), after a warmup "
            "of WARMUP_QUERIES. Latency of exact_knn and faiss_hnsw is "
            "directly comparable; pgvector is not measured here (see "
            "docs/retrieval-and-evaluation.md for why the served path is "
            "measured separately)."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "bench.written", "path": str(args.output)}))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
