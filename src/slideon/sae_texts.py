"""Stream raw text JSONL into the existing SAE token-cache format.

Each record requires string ``source`` and ``text`` fields. ``role`` defaults to
``document``; ``query`` is also accepted. An optional string ``id`` is validated
but does not trigger deduplication. Every record is encoded independently.
"""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import numpy as np


def _read_jsonl(paths, file_metadata):
    """Yield records in file order, accumulating fingerprints as files finish."""
    for path in paths:
        path = Path(path).resolve()
        digest = hashlib.sha256()
        size = rows = 0
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                digest.update(line)
                size += len(line)
                location = f"{path}:{line_number}"
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError) as error:
                    raise ValueError(f"{location}: invalid JSON record") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{location}: expected a JSON object")
                for key in ("source", "text"):
                    if not isinstance(record.get(key), str) or not record[key].strip():
                        raise ValueError(f"{location}: {key} must be a nonempty string")
                if "id" in record and (not isinstance(record["id"], str) or not record["id"].strip()):
                    raise ValueError(f"{location}: id must be a nonempty string when supplied")
                role = record.get("role", "document")
                if role not in ("query", "document"):
                    raise ValueError(f"{location}: role must be query or document")
                record["role"] = role
                rows += 1
                yield record
        file_metadata.append(dict(path=str(path), sha256=digest.hexdigest(), bytes=size, rows=rows))


def iter_jsonl(paths, expected_files=None):
    """Read without retaining texts; detect changes between scanning and encoding."""
    files = []
    yield from _read_jsonl(paths, files)
    if expected_files is not None and files != expected_files:
        raise ValueError("JSONL input changed during preparation; rebuild the cache from unchanged files")


def scan_jsonl(paths):
    """Count logical source/role groups so offset arrays can be written on disk."""
    files, groups = [], {}
    for record in _read_jsonl(paths, files):
        key = (record["source"], record["role"])
        groups[key] = groups.get(key, 0) + 1
    if not groups:
        raise ValueError("JSONL input contains no texts")
    if not any(role == "document" for _, role in groups):
        raise ValueError("JSONL input needs at least one document for SAE initialization")
    return dict(type="jsonl", version=1, files=files, groups=[
        dict(source=source, role=role, rows=rows) for (source, role), rows in groups.items()
    ])


def _write_manifest(root, manifest):
    temporary = root / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(root / "manifest.json")


def prepare_jsonl_cache(recipe, output, texts_jsonl, encode_batch_size, *, make_encoder, cache_recipe):
    """Prepare bounded batches without altering DenseOn extraction or SAE math.

The first pass counts rows and fingerprints files. The second writes FP16 tokens
and memory-mapped metadata arrays. Memory therefore depends on batch size and
the number of source/role groups, rather than the total number of records.
"""
    if encode_batch_size < 1:
        raise ValueError("encode_batch_size must be positive")
    if not texts_jsonl:
        raise ValueError("At least one JSONL file is required")
    paths = [Path(path).resolve() for path in texts_jsonl]
    inputs = scan_jsonl(paths)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    recipe_config = recipe.checkpoint_config()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("input") != inputs:
            raise ValueError("JSONL input differs from the existing cache; use a new cache directory")
        if cache_recipe(manifest["recipe"], "jsonl") != cache_recipe(recipe_config, "jsonl"):
            raise ValueError("Cache recipe mismatch")
        if manifest.get("complete"):
            return
    else:
        manifest = dict(recipe=recipe_config, dataset_revisions={}, input=inputs, sources=[], complete=False)
        _write_manifest(root, manifest)

    # A single stream per source/role spans all input files. Besides avoiding
    # arbitrary file boundaries, this retains initialization's documents-per-source rule.
    completed = {(source["split"], source["kind"]) for source in manifest["sources"]}
    encoder = make_encoder()
    writers = {}
    with ExitStack() as stack:
        for index, group in enumerate(inputs["groups"]):
            kind = "queries" if group["role"] == "query" else "documents"
            if (group["source"], kind) in completed:
                continue
            # Source labels never become paths: even labels containing '/' are safe.
            name = f"texts-{index:05d}-{kind}"
            offsets = np.lib.format.open_memmap(root / f"{name}-offsets.npy.partial", mode="w+",
                                              dtype=np.int64, shape=(group["rows"] + 1,))
            stack.callback(offsets._mmap.close)
            chars = np.lib.format.open_memmap(root / f"{name}-chars.npy.partial", mode="w+",
                                            dtype=np.int64, shape=(group["rows"],))
            stack.callback(chars._mmap.close)
            offsets[0] = 0
            stream = stack.enter_context((root / f"{name}.bin.partial").open("wb"))
            writers[(group["source"], group["role"])] = dict(
                name=name, split=group["source"], kind=kind, rows=group["rows"],
                tokens=0, scheduled_tokens=0, written=0, batch=[], stream=stream,
                offsets=offsets, chars=chars,
            )

        def flush(writer, role):
            batch = writer["batch"]
            if not batch:
                return
            limit = recipe.query_length if role == "query" else recipe.document_length
            # Match the original schedule: count raw text before applying prompts.
            writer["scheduled_tokens"] += sum(len(ids) for ids in encoder.tokenizer(
                batch, truncation=True, max_length=limit)["input_ids"])
            arrays = encoder.encode(batch, role)
            if len(arrays) != len(batch):
                raise ValueError("Frozen encoder returned an incorrect number of sequences")
            if writer["written"] + len(batch) > writer["rows"]:
                raise ValueError("JSONL row count changed during preparation")
            for text, array in zip(batch, arrays):
                if (array.ndim != 2 or array.shape[1] != recipe.d_model or not len(array)
                        or not np.isfinite(array).all()):
                    raise ValueError("Invalid frozen token embeddings")
                writer["stream"].write(array.astype(np.float16, copy=False).tobytes())
                row = writer["written"]
                writer["tokens"] += len(array)
                writer["offsets"][row + 1] = writer["tokens"]
                writer["chars"][row] = len(text)
                writer["written"] += 1
            batch.clear()

        for record in iter_jsonl(paths, expected_files=inputs["files"]):
            writer = writers.get((record["source"], record["role"]))
            if writer is None:
                if (record["source"], "queries" if record["role"] == "query" else "documents") not in completed:
                    raise ValueError("JSONL sources changed during preparation")
                continue
            writer["batch"].append(record["text"])
            if len(writer["batch"]) == encode_batch_size:
                flush(writer, record["role"])
        for (_, role), writer in writers.items():
            flush(writer, role)
            if writer["written"] != writer["rows"]:
                raise ValueError("JSONL row count changed during preparation")
            writer["offsets"].flush()
            writer["chars"].flush()

    # Publish only after every input fingerprint has been checked and files closed.
    # Interrupted streams remain .partial and are rebuilt on the next invocation.
    for writer in writers.values():
        for suffix in (".bin", "-offsets.npy", "-chars.npy"):
            (root / (writer["name"] + suffix + ".partial")).replace(root / (writer["name"] + suffix))
        source = {key: writer[key] for key in ("name", "split", "kind", "rows", "tokens", "scheduled_tokens")}
        manifest["sources"].append(source)
        _write_manifest(root, manifest)
        print(f"Cached {source['split']}/{source['kind']}: {source['rows']:,} texts, {source['tokens']:,} tokens", flush=True)
    manifest["complete"] = True
    _write_manifest(root, manifest)
