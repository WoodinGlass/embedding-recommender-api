# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `docs/decisions.md`: pre-flight design decisions locked before feature work —
  problem statement, payload vs pipeline split, deterministic/non-deterministic
  idempotency keys, storage tiers, ingestion modes, failure modes, and the
  explicit out-of-scope list. (M0.2)
- `pyproject.toml` as the single source of truth for the package. Runtime
  dependencies live in self-contained optional groups (`api`, `db`, `cache`,
  `auth`, `embeddings`, `observability`, `experiments`, `churn`, `pipeline`,
  `bench`, `load`) with meta groups `service` and `dev`. Also configures Ruff,
  mypy (strict), pytest markers, and coverage. (M0.3)
- `.gitignore` covering secrets, Python build artifacts, test caches,
  notebook checkpoints, DVC, Terraform, Docker overrides, editor noise, and
  Colab scratch directories. Sample data under `data/sample/` is allowlisted.
  (M0.3)
- `.env.example` documenting every configuration variable referenced by the
  README, with backend selectors expressed as enums (`INDEX_BACKEND`,
  `APP_ENV`, `DEVICE`) instead of booleans. (M0.3)
- `CHANGELOG.md` (this file). (M0.3)

### Changed
- n/a

### Fixed
- n/a

### Security
- n/a

## Notes

Milestone tracking lives in [`README.md`](README.md#milestones). Only measured
numbers belong in the README; this changelog records what changed and when.
