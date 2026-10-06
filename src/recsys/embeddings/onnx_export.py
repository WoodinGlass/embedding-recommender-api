"""Export a sentence-transformers model to ONNX.

Produces, in ``out_dir``:

- ``model.onnx`` — the exported graph, with mean pooling and L2 normalization
  folded in so that the graph produces the same final embedding as
  ``SentenceTransformer.encode(normalize_embeddings=True)``.
- ``model.onnx.sha256`` — SHA256 of ``model.onnx``, loaded and re-verified
  by :class:`~recsys.embeddings.encoder.OnnxEncoder`.
- ``tokenizer.json`` — fast-tokenizer file; the runtime tokenizes without
  ``transformers``.
- ``config.json`` — ``{model_name, max_seq_length, embedding_dim}``.

Idempotent: re-running with the artifact present is a no-op unless
``force=True``. See ``docs/embedding-pipeline.md`` § 9.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from recsys._optional import optional_import
from recsys.embeddings.encoder import sha256_file


def _require(name: str, extra: str) -> Any:
    mod = optional_import(name)
    if mod is None:
        raise RuntimeError(
            f"{name!r} is required for ONNX export. Install with: pip install -e '.[{extra}]'"
        )
    return mod


def _make_wrapper(auto_model: Any) -> Any:
    """Return a torch.nn.Module that performs mean pooling and L2 norm.

    The wrapper is defined inside the function so that importing this module
    does not require torch.
    """
    torch = _require("torch", "export")
    nn = torch.nn
    functional = torch.nn.functional

    class _Wrapper(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.inner(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            return functional.normalize(pooled, p=2, dim=1)

    wrapper = _Wrapper(auto_model)
    wrapper.eval()
    return wrapper


def export(
    model_name: str,
    out_dir: pathlib.Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Export ``model_name`` to ``out_dir``. Returns a JSON-serializable
    summary suitable for logging.
    """
    out_dir = pathlib.Path(out_dir)
    onnx_path = out_dir / "model.onnx"
    sha_path = out_dir / "model.onnx.sha256"

    if onnx_path.is_file() and sha_path.is_file() and not force:
        return {
            "event": "export.skipped",
            "reason": "artifact exists; pass force=True to re-export",
            "model_name": model_name,
            "onnx_path": str(onnx_path),
            "sha256": sha_path.read_text(encoding="utf-8").strip(),
        }

    torch = _require("torch", "export")
    st = _require("sentence_transformers", "export")
    transformers = _require("transformers", "export")

    out_dir.mkdir(parents=True, exist_ok=True)

    model = st.SentenceTransformer(model_name, device="cpu")
    try:
        auto_model = model[0].auto_model
    except (AttributeError, IndexError) as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"could not locate the underlying transformers model on {model_name!r}; "
            f"the sentence-transformers module layout may have changed"
        ) from exc
    auto_model.eval()

    max_seq_length = int(model.max_seq_length)
    dim = model.get_sentence_embedding_dimension()
    if dim is None:
        raise RuntimeError(f"model {model_name!r} does not expose a sentence embedding dimension")
    embedding_dim = int(dim)

    wrapper = _make_wrapper(auto_model)
    dummy_ids = torch.ones((1, 4), dtype=torch.long)
    dummy_mask = torch.ones((1, 4), dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    tok.save_pretrained(str(out_dir))

    config = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "embedding_dim": embedding_dim,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sha = sha256_file(onnx_path)
    sha_path.write_text(sha + "\n", encoding="utf-8")

    return {
        "event": "export.done",
        "model_name": model_name,
        "onnx_path": str(onnx_path),
        "max_seq_length": max_seq_length,
        "embedding_dim": embedding_dim,
        "size_bytes": onnx_path.stat().st_size,
        "sha256": sha,
    }
