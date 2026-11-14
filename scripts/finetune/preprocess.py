"""Join LightOn's English HF tables into the JSONL consumed by train_slideon.py.

Example (run from the repository root):
    python scripts/finetune/preprocess.py \
        --dataset lightonai/embeddings-fine-tuning-filtered-en --output english.jsonl

Each source score row is exported once, with its source label, all ten negatives,
and all eleven raw teacher scores in their original order. No examples or
negatives are sampled, duplicated, deduplicated or reweighted during conversion.
Source and negative sampling belong in training; the manifest records this.
Queries and documents are held in RAM for ONE source at a time; no GPU is needed.
"""

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path


ENGLISH_SPLITS = (
    "fiqa_en", "hotpotqa_en", "nq_en", "msmarco_en", "fever_en",
    "squadv2_en", "trivia_en", "miracl_en", "mldr_en",
)


def text_lookup(table, id_column, text_column):
    """Resolve IDs within one source; IDs are not global across source splits."""
    result = {}
    for row in table:
        key, value = row[id_column], row[text_column]
        if key in result:
            raise ValueError(f"Duplicate {id_column}: {key}")
        # Keep the original text, including whitespace, newlines and Unicode.
        result[key] = value
    return result


def make_example(row, queries, documents, source):
    """Resolve every candidate to text, retaining its position and teacher score."""
    ids, scores = row["document_ids"], row["rerank_scores"]
    if len(ids) != 11 or len(scores) != 11:
        raise ValueError("Expected one positive, ten negatives and eleven aligned rerank_scores")
    if any(isinstance(s, bool) or not isinstance(s, Real) or not math.isfinite(s) for s in scores):
        raise ValueError("Teacher scores must be finite numbers")

    # Index zero is the positive. Keep every negative, even if the source repeats
    # a candidate: conversion must not silently change the published examples.
    texts = [documents[document_id] for document_id in ids]
    query = queries[row["query_id"]]
    if any(not isinstance(text, str) for text in [query] + texts):
        raise ValueError("Query/document texts must be strings")
    return {
        "source": source,
        "query": query,
        "positive": texts[0],
        "negatives": texts[1:],
        # Use raw cross-encoder rerank_scores, NOT bi-encoder mining scores.
        # The training loss applies softmax/temperature itself.
        "teacher_scores": list(scores),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="lightonai/embeddings-fine-tuning-filtered-en")
    parser.add_argument("--output", type=Path, default=Path("english.jsonl"))
    parser.add_argument("--revision", default="main", help="HF revision; resolved to a commit before loading")
    parser.add_argument("--splits", nargs="+", choices=ENGLISH_SPLITS, default=list(ENGLISH_SPLITS))
    parser.add_argument("--cache-dir", help="Optional Hugging Face datasets cache directory")
    parser.add_argument("--max-rows-per-split", type=int, help="Small conversion check; omit for the full dataset")
    args = parser.parse_args()
    if len(set(args.splits)) != len(args.splits):
        parser.error("--splits must not contain duplicates")
    if args.max_rows_per_split is not None and args.max_rows_per_split < 1:
        parser.error("--max-rows-per-split must be positive")
    return args


def main():
    args = parse_args()
    # These are already project dependencies. Import here so --help is cheap.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    output = args.output
    manifest_path = Path(str(output) + ".manifest.json")
    partial = Path(str(output) + ".partial")
    manifest_partial = Path(str(manifest_path) + ".partial")
    for path in (output, manifest_path, partial, manifest_partial):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose another --output")
    output.parent.mkdir(parents=True, exist_ok=True)

    # All three tables and every source must come from the SAME immutable commit.
    revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
    manifest = {
        "dataset": args.dataset, "requested_revision": args.revision, "revision": revision,
        "splits": args.splits, "source_column": "source", "num_negatives": 10,
        "negative_selection": "all_in_source_order",
        "source_sampling": "all_rows_once_no_reweighting",
        "teacher_score_column": "rerank_scores", "text_processing": "unchanged",
        "max_rows_per_split": args.max_rows_per_split, "sources": [],
    }
    digest, total = hashlib.sha256(), 0
    print(f"Dataset: {args.dataset}@{revision}", flush=True)

    # A failed conversion never leaves a partially written english.jsonl.
    # Large files are written incrementally; the output rows are not kept in RAM.
    try:
        with partial.open("xb") as stream:
            for split in args.splits:
                print(f"Loading {split} (queries, documents, scores)...", flush=True)

                def load_table(kind):
                    # Match the dataset card's source-specific file selection.
                    # HF calls the selected files "train" here, even though their
                    # original source name is, for example, "fiqa_en".
                    return load_dataset(
                        args.dataset, name=kind, revision=revision,
                        data_files=f"{kind}/{split}-*", split="train",
                        verification_mode="no_checks", cache_dir=args.cache_dir,
                    )

                queries = text_lookup(load_table("queries"), "query_id", "query")
                documents = text_lookup(load_table("documents"), "document_id", "document")
                scores = load_table("scores")
                count = len(scores) if args.max_rows_per_split is None else min(len(scores), args.max_rows_per_split)
                for index, row in enumerate(scores):
                    if index >= count:
                        break
                    try:
                        example = make_example(row, queries, documents, split)
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(f"{split} row {index}: {error}") from error
                    # json.dumps escapes text newlines: one object per physical line.
                    line = (json.dumps(example, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                    stream.write(line)
                    digest.update(line)
                    if (index + 1) % 100000 == 0:
                        print(f"  {split}: {index + 1:,}/{count:,} rows", flush=True)
                if count == 0:
                    raise ValueError(f"No score rows in {split}")
                manifest["sources"].append({"split": split, "available_rows": len(scores), "written_rows": count})
                total += count
                print(f"  {split}: wrote {count:,} rows", flush=True)
                # Release the largest text lookup before loading the next source.
                del queries, documents, scores

        manifest.update(total_rows=total, sha256=digest.hexdigest())
        with manifest_partial.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        manifest_partial.replace(manifest_path)
        try:
            partial.replace(output)
        except BaseException:
            # Do not leave a manifest claiming that a failed export completed.
            manifest_path.unlink(missing_ok=True)
            raise
    finally:
        partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)

    print(f"Wrote {total:,} rows to {output}; provenance: {manifest_path}", flush=True)
    print("Source proportions are unchanged; this does not implement alpha=0.5 source sampling.")


if __name__ == "__main__":
    main()
