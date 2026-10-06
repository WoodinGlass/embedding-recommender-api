# Load tests

Locust (and optionally k6) scenarios used to measure the latency target
(p95 < 200 ms on 100k+ items at the target RPS). Populated in **M4**.

Planned layout:

```text
tests/load/
├── locustfile.py       # Locust user classes and scenarios
├── k6/                 # optional k6 scripts
├── README.md           # this file
└── results/            # git-ignored; committed under docs/ as reports
```

Every published result must record hardware, dataset size, and commit SHA.
Only measured numbers belong in the README — targets are marked `TBD` until
then. See `docs/decisions.md` § 2 (payload vs pipeline).
