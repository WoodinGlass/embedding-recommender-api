# ADR-0012: Backend abstraction, `NumpyBackend` for exact kNN, and pgvector in CI only

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

M2 needs to run retrieval in three environments that cannot all host the
same storage:

1. **CI**, where the workflow already provisions a `pgvector/pgvector:pg16`
   service container. pgvector is available and is the production path.
2. **Google Colab**, where the primary development environment runs. Colab
   has no Docker daemon, so the service container pattern does not apply,
   and installing PostgreSQL + pgvector via `apt` from source is fragile
   and slow. Retrieval must still be exercisable there — otherwise M2
   development degenerates into "push a commit and wait for CI".
3. **Evaluation**, where an exact kNN baseline over the sample catalog is
   needed. Exact kNN is not pgvector's job (its HNSW index is approximate
   by design); it needs a brute-force comparison that reads the same vectors
   and produces the same distance ordering.

Additionally, the M0 scaffold defines an `IndexBackend` protocol in
`src/recsys/retrieval/base.py` and a `_REGISTRY` in
`src/recsys/retrieval/registry.py`, with two placeholder factories
(pgvector, faiss) that raise `NotImplementedError`. M2 fills one of them
in and adds a third. The protocol's current method signature is:

```python
async def search(
    self,
    *,
    vector: list[float],
    k: int,
    filters: dict[str, object] | None = None,
) -> list[tuple[str, float]]: ...
```

The M0 signature was written before the metrics and evaluation harness
existed, and it needs a decision about what "backend" means: is a backend
an asynchronous service client, or is it a strategy that may be
implemented in-process?

## Decision

### One protocol, two implementations in M2, one more in M3

`IndexBackend` remains the interface. M2 adds two implementations:

- **`PgvectorBackend`** — the production path. Reads vectors from the
  `embedding` table over a `psycopg` connection; runs an HNSW query with
  an optional `WHERE` clause (per ADR-0008); returns `(item_id, score)`
  pairs sorted by score descending, ties broken by `item_id` ascending.
- **`NumpyBackend`** — an in-process brute-force implementation. Reads
  the active run's Parquet into a numpy array at construction; computes
  cosine distance for every query against every row; returns the same
  shape as pgvector.

FAISS remains a benchmark-only path (ADR-0011) and is **not** registered
as a backend.

### `NumpyBackend` is not a production path

`NumpyBackend` exists for exactly three reasons:

1. **Colab end-to-end evaluation.** Colab runs the sample catalog (200
   items × 384 dims = ~300 KB of float32). Brute-force over 200 items
   is microseconds per query. The golden set's 20 queries finish in
   milliseconds. There is no scenario in Colab where the ANN machinery
   matters; the point of running evaluation there is to check the
   metrics code and the report format, not the index.
2. **The exact kNN baseline.** The M2 evaluation table and the ANN-fidelity
   metric (ADR-0009) require a brute-force reference. `NumpyBackend`
   provides that reference using the *same* `Backend` interface as the
   production path, so the evaluation harness is not special-cased.
3. **Regression tests.** A small test can construct both a `NumpyBackend`
   and a `PgvectorBackend` over the same vectors and assert that, with
   `ef_search` large enough that HNSW degenerates to exact search, the two
   return identical top-k. This is the test that proves the two
   implementations agree on what "cosine distance" and "top-k" mean.

`NumpyBackend` is **not** a production path, is **not** registered as
`INDEX_BACKEND=numpy` in `Settings`, and has no configuration variable.
It is constructible from code and from tests; selecting it via env would
invite deploying a system that loads a 1.5 GB array at startup on every
instance.

The `IndexBackendEnum` (in `src/recsys/config/enums.py`) has two values:
`pgvector` and `faiss`. `numpy` is deliberately **not** a value. The
registry maps `pgvector` to `PgvectorBackend`; `faiss` stays a
`NotImplementedError` for production but is used by the benchmark script
directly, outside the registry.

### Protocol signature: keep it synchronous, keep it direct

The M0 signature is `async` and takes `vector: list[float]`. M2 changes
it to:

```python
class IndexBackend(Protocol):
    name: str

    def is_ready(self) -> bool: ...

    def search(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float]]: ...
```

The changes and their reasons:

- **Synchronous, not async.** Both implementations are synchronous.
  `PgvectorBackend` wraps a synchronous `psycopg` connection (psycopg3's
  async support exists but the project does not need it at this scale);
  `NumpyBackend` is a numpy operation. Making them `async` would add
  ceremony (running sync code in a thread pool) without adding concurrency:
  the retrieval layer will be called from the M3 API, and FastAPI's
  synchronous endpoint path already runs sync code in a thread pool. If
  M3 measures that async retrieval helps, the ADR is superseded, not
  edited; the change is mechanical.
- **`NDArray[np.float32]`, not `list[float]`.** The vectors come from the
  encoder (which returns `NDArray[np.float32]`, M1) and are consumed by
  numpy (`NumpyBackend`) or handed to psycopg for a `vector` parameter
  (`PgvectorBackend`). Converting to `list` and back at every layer would
  cost more than the type change.
- **`Mapping[str, str]`, not `dict[str, object]`.** Filter values come from
  the request as strings. The protocol refuses non-string values so that an
  accidental `int` or `bool` never reaches a SQL parameter binding as
  something other than text.

The `IndexBackend` protocol remains `runtime_checkable`. A test asserts
that both implementations satisfy it.

### The two implementations must agree on exact search

For the same vectors and the same query, with HNSW degenerate to exact
search (or with `NumpyBackend` compared against a brute-force query in
pgvector), `PgvectorBackend` and `NumpyBackend` must return:

- the same `k` item ids, in the same order;
- scores that agree within a documented tolerance (`1e-5`, from float32
  cosine arithmetic).

This is tested as a **conditional integration test**: it runs when
`RECSYS_TEST_DATABASE_URL` is set (so it runs in CI) and skips in Colab.
The test constructs both backends over a small fixture (10 items), queries
the same vector against both, and asserts equality. A failure here means
the two implementations disagree about what retrieval is — a bug that
would make the ANN-fidelity metric meaningless.

The tie-breaking rule (by `item_id` ascending) is applied by the
**caller**, not by the backends. A backend returns its top-k in score
order; if two items have equal scores, the order among them is
implementation-defined. The retrieval layer normalizes by sorting with
the deterministic key. This keeps the backends simple (pgvector's SQL
does not have a natural tie-break key; numpy's `argsort` is stable but the
input order depends on how the Parquet was sorted). The normalization is
tested with a fixture that deliberately contains a tie.

### pgvector integration tests run only in CI

`tests/integration/test_pgvector.py` is marked `integration` (not
`encoder`) and skips unless `RECSYS_TEST_DATABASE_URL` is set. Colab does
not set that variable, so the tests skip there. CI's `test-integration`
job already provisions a pgvector service container and sets the
variable.

This means Colab can:

- Develop and lint the `PgvectorBackend` code.
- Run its unit tests (the code that builds SQL strings, the parameter
  binding, the row-parsing logic — none of which need a live database).
- Run end-to-end evaluation via `NumpyBackend`.
- Run the full three-tier determinism suite (M1).

Colab cannot:

- Execute the integration test that actually opens a pgvector connection.

The gap is documented in `docs/retrieval-and-evaluation.md`. A developer
running Colab who wants to exercise pgvector locally must use a machine
with Docker (or install PostgreSQL + pgvector, which the design doc
describes as "possible but unsupported for the primary environment").

### `is_ready` semantics

The `is_ready()` method answers: "could this backend serve a search right
now?" It is used by `/readyz` (M3) and by the evaluation harness (M2) to
decide whether to run.

- `NumpyBackend.is_ready()` — `True` after construction, which loads the
  vectors. If construction failed, the backend does not exist.
- `PgvectorBackend.is_ready()` — `True` if a connection can be opened and
  the active `index_version` has rows in `embedding`. This is a real check,
  not a cached flag, because the M3 `/readyz` semantics require it (a
  drained instance should report not-ready).

The evaluation harness does not gate on `is_ready`; it calls `search`
directly and lets exceptions propagate. `is_ready` exists for the serving
path, and M2 wires it but does not depend on it.

### Registry changes

`src/recsys/retrieval/registry.py` is updated so that:

- `_pgvector_factory()` returns a `PgvectorBackend` when the `db` extra is
  installed and the database is reachable, and raises a clear error
  otherwise (missing extra, missing connection).
- `_faiss_factory()` continues to raise `NotImplementedError` for
  production, with a docstring that points at the benchmark script.
- A new internal mapping `NON_PRODUCTION_BACKENDS = {"faiss"}` records
  backends that exist but are not selectable via `INDEX_BACKEND` at
  runtime. This is not exposed; it exists so that a future ADR that adds a
  fourth backend has a place to declare its status.

### Handler thread model (M3.6 amendment)

The protocol is synchronous (above). The handler that calls it is
not: FastAPI runs an ``async def`` handler on the event loop, and
the retrieval path — encode, ANN query, re-rank — is blocking. The
arrangement is:

- **The handler is ``async def``.** The event loop stays free for
  other requests while this one waits.
- **One thread hop, not four.** ``anyio.to_thread.run_sync`` runs
  the entire synchronous pipeline (retrieve, re-rank, fallback)
  in one call. Wrapping each step separately would pay four
  hops for one request, and the intermediate results would have to
  cross the async/sync boundary each time.
- **The pool is bounded.** ``anyio.CapacityLimiter`` (set to
  ``sync_thread_limit``, default 20) caps how many requests can
  occupy a worker thread at once. Without the bound, more threads
  than the downstream dependency can serve is a longer queue in a
  different place; the bound is where the queue should be, because
  it is observable (``recsys_thread_pool_waiting``) and a saturated
  database is not.
- **A request-level timeout.** ``anyio.fail_after(request_timeout_seconds)``
  bounds the total wall clock. Cancelling ``to_thread.run_sync``
  does not stop the thread — it keeps running to completion and its
  result is discarded — but the request returns promptly and the
  latency budget is respected. A future variant that can cancel
  mid-flight would use a process pool; that is deferred.

The alternative — making the backend async and awaiting each call —
is what the M2 decision rejected: it would put four ``await``
points on the request path for no concurrency gain (the
dependencies are blocking), and every caller of the backend
(evaluation, scripts, tests) would pay for the handler's
concurrency model.

## Alternatives considered

| Option | Why not |
|---|---|
| **Skip `NumpyBackend`, require pgvector in Colab** | Colab would have no end-to-end evaluation path, and the exact kNN baseline would need its own code path outside the `Backend` interface. Both costs are larger than the ~40 lines `NumpyBackend` requires. |
| **Make `NumpyBackend` a selectable `INDEX_BACKEND` value** | Invites deployment of a backend that loads the whole catalog into RAM at startup, with O(N) per query. Fine on 200 items, catastrophic on 1M. The absence of the enum value is the guardrail. |
| **Keep the M0 `async` signature** | Neither implementation is async; making them async would wrap sync code in `asyncio.to_thread` for no benefit. The API layer (M3) already runs sync endpoints in a thread pool, so the async boundary would be false. |
| **Keep `list[float]` and convert inside each backend** | The conversion cost is small per query but the API boundary would hide the fact that the encoder produces `NDArray`. Numpy-to-list-to-numpy at every call is the kind of overhead that shows up in a p95. |
| **Accept `dict[str, object]` for filters** | An `int` value would flow to psycopg as an integer parameter, and pgvector would compare it to a text column. The error would surface as a type mismatch at query time, not at the boundary. The narrow type (`Mapping[str, str]`) rejects the mistake early. |
| **Do tie-breaking inside the backends** | Requires every backend to know the tie-break rule and to have a stable ordering of its internal data. pgvector's SQL would need `ORDER BY score DESC, item_id ASC` at every call site; `NumpyBackend`'s input order depends on the Parquet's row order. Moving the rule to the caller is one place instead of N. |
| **Skip the agreement test** | The two implementations must agree on the same question ("what are the nearest items to this vector?") or the ANN-fidelity metric is measuring something else. The test is 20 lines and catches a whole class of bugs. |
| **Register `NumpyBackend` for `faiss` and let the benchmark use it** | Conflates two different things: `NumpyBackend` is exact, FAISS is approximate. Registering one under the other's name is exactly the kind of misleading abstraction this ADR exists to prevent. |
| **Move `is_ready` to a separate protocol (`ReadinessCheck`)** | Would split the interface into two for a method that every backend has. If M3 finds that some component needs a readiness check without a search path, the split is a future ADR. |

## Consequences

**Positive**

- **Colab can develop and evaluate retrieval end-to-end.** The metrics, the
  report format, the threshold gate, and the golden set can all be
  exercised there. The only thing that requires CI is the actual pgvector
  connection.
- **The exact kNN baseline uses the production interface.** The evaluation
  harness has one code path for all backends; a baseline is a backend, not
  a special case.
- **The two implementations are tested against each other.** A future
  change to either that alters what "top-k by cosine distance" means will
  fail the agreement test. This is the check that keeps the comparison
  honest.
- **The tie-break rule lives in one place.** Retrieval ordering is a
  property of the caller, not of the storage layer. Adding a fourth backend
  does not require re-implementing tie-breaking.

**Negative / accepted trade-offs**

- **pgvector is not exercised in the primary development environment.**
  Colab developers must push to CI to run the pgvector integration tests.
  This is the same trade-off as M1's ONNX encoding (which is exercised in
  CI, not Colab) and is accepted for the same reason: the primary
  environment cannot host every dependency, and the CI job exists
  precisely for that.
- **The protocol signature change touches the M0 scaffold.** `base.py`'s
  `search` signature changes from async-`list[float]` to sync-`NDArray`.
  The only call site in M0 is the placeholder factories (which raise), so
  the change is mechanical. It is a change to a file that has been in the
  repo since M0, which is a reminder that "the scaffold is final" is not a
  rule.
- **`NumpyBackend` has a small amount of coupling to the artifact layout.**
  It reads the active run's Parquet. That coupling is deliberate: the
  baseline should read the same vectors the production path reads, or the
  comparison is not like-for-like. If the artifact layout changes, the
  backend changes with it — which is the same coupling `PgvectorBackend`
  has to the `embedding` table.
- **`is_ready()` is synchronous in a world where readiness is often async.**
  A synchronous `is_ready` that opens a connection blocks the caller. M3
  will call it from a readiness endpoint, which is not latency-sensitive.
  If M3 finds that a blocking readiness check causes head-of-line blocking
  under load, the method becomes async and the ADR is superseded. Not
  solving a problem M3 has not reported.
- **The agreement test requires pgvector and numpy to use the same
  floating-point arithmetic.** In practice this is true (both use IEEE 754
  float32 cosine), but a future hardware or BLAS change could introduce
  differences above 1e-5. The tolerance is documented and can be raised if
  measurements show that it needs to be; the test itself remains
  meaningful.

## References

- `docs/adr/0001-pgvector-as-default.md` — why pgvector and why FAISS is
  benchmark-only
- `docs/adr/0006-pgvector-schema.md` — the `embedding` table
  `PgvectorBackend` reads
- `docs/adr/0008-filter-strategy.md` — the filter semantics
  `PgvectorBackend` implements
- `docs/adr/0009-golden-set-and-metrics.md` — the exact kNN baseline and
  ANN-fidelity metric that motivate `NumpyBackend`
- `docs/adr/0011-faiss-benchmark-methodology.md` — where FAISS runs, and
  why it is not a backend
- `src/recsys/retrieval/base.py` — the protocol this ADR modifies
- `src/recsys/retrieval/registry.py` — the registry this ADR updates
- `docs/retrieval-and-evaluation.md` — the M2 design doc
