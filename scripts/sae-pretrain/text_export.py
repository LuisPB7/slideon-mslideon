"""Small, dependency-free writer shared by the SAE text corpus exporters."""

import hashlib
import json
from pathlib import Path


def default_data_dir():
    """Use this checkout's data/ directory, independent of the working directory."""
    return Path(__file__).resolve().parents[2] / "data"


def write_jsonl(records, output, *, provenance, limit=None, overwrite=False):
    """Write the SAE text contract without buffering the corpus in RAM.

Only blank texts are skipped, with an explicit count. Other invalid fields fail
the export. Nonblank text is kept verbatim, including whitespace and newlines.
An incomplete export never becomes the final JSONL file. A stale .partial file
must be removed explicitly before restarting; this function does not resume.
"""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    output = Path(output)
    manifest_path = Path(str(output) + ".manifest.json")
    partial = Path(str(output) + ".partial")
    manifest_partial = Path(str(manifest_path) + ".partial")
    for path in (output, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new output or use --overwrite")
    for path in (partial, manifest_partial):
        if path.exists():
            raise FileExistsError(f"Unfinished export exists: {path}; remove it before restarting")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    rows = read_rows = skipped_blank = size = 0
    print(f"Exporting to {output}", flush=True)

    # Open exclusively before entering cleanup, so a concurrent writer's file
    # cannot be deleted if this process fails to acquire the output filename.
    stream = partial.open("xb")
    owns_manifest_partial = False
    try:
        with stream:
            for record in records:
                read_rows += 1
                if not isinstance(record, dict):
                    raise ValueError(f"Record {read_rows}: expected a dictionary")
                for key in ("id", "source"):
                    if not isinstance(record.get(key), str) or not record[key].strip():
                        raise ValueError(f"Record {read_rows}: {key} must be a nonempty string")
                if record.get("role") not in ("query", "document"):
                    raise ValueError(f"Record {read_rows}: role must be query or document")
                text = record.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"Record {read_rows}: text must be a string")
                if not text.strip():
                    skipped_blank += 1
                    continue
                # Select these fields explicitly: titles and metadata never enter
                # the SAE text, even when the original dataset supplies them.
                row = {key: record[key] for key in ("id", "source", "role", "text")}
                line = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                stream.write(line)
                digest.update(line)
                size += len(line)
                rows += 1
                if rows % 100000 == 0:
                    print(f"  Wrote {rows:,} texts ({skipped_blank:,} blank texts skipped)", flush=True)
                if limit is not None and rows == limit:
                    break
        if not rows:
            raise ValueError("No nonblank texts were exported")
        manifest = dict(schema_version=1, provenance=provenance, rows=rows,
                        read_rows=read_rows, skipped_blank=skipped_blank,
                        sha256=digest.hexdigest(), bytes=size, limit=limit,
                        text_processing="unchanged_no_titles")
        with manifest_partial.open("x", encoding="utf-8") as metadata:
            owns_manifest_partial = True
            json.dump(manifest, metadata, ensure_ascii=False, indent=2, allow_nan=False)
            metadata.write("\n")
        manifest_partial.replace(manifest_path)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
        if owns_manifest_partial:
            manifest_partial.unlink(missing_ok=True)

    print(f"Wrote {rows:,} texts to {output}; skipped {skipped_blank:,} blank texts", flush=True)
    print(f"Provenance and checksum: {manifest_path}", flush=True)
    return manifest
