"""Encoder interface and concrete implementations.

Two encoders ship in ``src/``:

- :class:`OnnxEncoder` — production path. Loads an exported ONNX artifact
  and runs inference through onnxruntime. No torch, no sentence-transformers.
- :class:`ReferenceEncoder` — wraps sentence-transformers. Used by the ONNX
  export script and by the parity test; never at serve time (ADR-0002).

Both implement :class:`Encoder` and return L2-normalized ``float32`` arrays
of shape ``(batch_size, embedding_dim)``. Normalization is part of the
contract: retrieval uses cosine distance, which on normalized vectors is
just a dot product.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from recsys._optional import optional_import


def sha256_file(path: Path) -> str:
    """Return the full hex SHA256 of the bytes of ``path``.

    Reads in chunks so a large artifact does not have to fit in memory at
    once.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@runtime_checkable
class Encoder(Protocol):
    """Protocol every encoder satisfies.

    Implementations are not required to be thread-safe; the batch pipeline
    calls them from a single worker per instance.
    """

    @property
    def model_version(self) -> str:
        """Identifier recorded in artifacts and manifests."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output vectors."""
        ...

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode ``texts`` into a ``(len(texts), embedding_dim)`` array.

        The returned array is ``float32`` and L2-normalized along axis 1.
        Implementations must produce deterministic output for a fixed
        process environment (thread count included). See
        ``docs/embedding-pipeline.md`` § 5.
        """
        ...


class ReferenceEncoder:
    """Wraps a sentence-transformers model.

    This is the *reference* implementation of :class:`Encoder`. It is used
    by the ONNX export script (to produce the artifact) and by the parity
    test in ``tests/integration``. It is not used at serve time — see
    ADR-0002 for why.
    """

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        st = optional_import("sentence_transformers")
        if st is None:
            raise RuntimeError(
                "ReferenceEncoder requires the 'export' extra. "
                "Install with: pip install -e '.[export]'"
            )
        self._model_name = model_name
        self._model = st.SentenceTransformer(model_name, device=device)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                f"model {model_name!r} does not expose a sentence embedding dimension"
            )
        self._dim = int(dim)

    @property
    def model_version(self) -> str:
        """The reference has no ONNX artifact, so no ``+<sha8>`` suffix."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        arr = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return cast(NDArray[np.float32], np.asarray(arr, dtype=np.float32))


class OnnxEncoder:
    """Production encoder backed by an exported ONNX artifact.

    Reads ``model.onnx``, ``model.onnx.sha256``, ``tokenizer.json``, and
    ``config.json`` from ``onnx_dir``. ``model_version`` is built from the
    operator-supplied ``label`` and the first 8 hex characters of the
    artifact SHA256 (see ``docs/contracts.md`` § 1.3).
    """

    def __init__(
        self,
        onnx_dir: Path,
        *,
        label: str,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
        verify_checksum: bool = True,
    ) -> None:
        ort = optional_import("onnxruntime")
        tokenizers_mod = optional_import("tokenizers")
        if ort is None or tokenizers_mod is None:
            raise RuntimeError(
                "OnnxEncoder requires the 'inference' extra. "
                "Install with: pip install -e '.[inference]'"
            )

        onnx_dir = Path(onnx_dir)
        onnx_path = onnx_dir / "model.onnx"
        sha_path = onnx_dir / "model.onnx.sha256"
        tok_path = onnx_dir / "tokenizer.json"
        cfg_path = onnx_dir / "config.json"

        for p in (onnx_path, sha_path, tok_path, cfg_path):
            if not p.is_file():
                raise FileNotFoundError(f"missing ONNX artifact file: {p}")

        recorded_sha = sha_path.read_text(encoding="utf-8").strip().lower()
        if len(recorded_sha) != 64 or any(c not in "0123456789abcdef" for c in recorded_sha):
            raise ValueError(f"malformed SHA256 sidecar: {sha_path}")
        if verify_checksum:
            actual = sha256_file(onnx_path)
            if actual != recorded_sha:
                raise ValueError(
                    "ONNX artifact checksum mismatch. The .onnx file and its "
                    ".sha256 sidecar disagree; re-export or restore.\n"
                    f"  recorded: {recorded_sha}\n"
                    f"  actual  : {actual}"
                )

        self._label = label
        self._sha256 = recorded_sha
        self._sha8 = recorded_sha[:8]
        self._onnx_dir = onnx_dir

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self._model_name: str = str(cfg["model_name"])
        self._max_seq_length: int = int(cfg["max_seq_length"])
        self._dim: int = int(cfg["embedding_dim"])

        # mypy: onnxruntime and tokenizers have no stubs; they are in the
        # mypy override list (ignore_missing_imports), so attribute access is
        # intentionally unchecked here.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = inter_op_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # warnings and above

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._output_name: str = self._session.get_outputs()[0].name

        tokenizer_class = tokenizers_mod.Tokenizer
        self._tokenizer = tokenizer_class.from_file(str(tok_path))
        self._tokenizer.enable_truncation(max_length=self._max_seq_length)
        pad_id = self._tokenizer.token_to_id("[PAD]")
        self._tokenizer.enable_padding(
            pad_id=pad_id if pad_id is not None else 0,
            # "[PAD]" is a tokenizer sentinel, not a secret. The bandit
            # rule S106 flags any string assigned to a `*_token` keyword;
            # the suppression documents that this call site is safe.
            pad_token="[PAD]",  # noqa: S106
        )

    # ------------------------------------------------------------------ props
    @property
    def model_version(self) -> str:
        return f"{self._label}+{self._sha8}"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        """The sentence-transformers model the artifact was exported from."""
        return self._model_name

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @property
    def sha256(self) -> str:
        """Full 64-hex SHA256 of the ONNX artifact."""
        return self._sha256

    # ---------------------------------------------------------------- encode
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)

        feeds: dict[str, NDArray[np.int64]] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Some exported graphs also declare token_type_ids. Pass zeros; the
        # reference model was trained on single-segment inputs.
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run([self._output_name], feeds)
        arr = np.asarray(outputs[0], dtype=np.float32)
        if arr.shape != (len(texts), self._dim):
            raise RuntimeError(
                f"ONNX output shape {arr.shape} does not match expected ({len(texts)}, {self._dim})"
            )
        return arr
