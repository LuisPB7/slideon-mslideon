"""Export the full MS MARCO passage collection as SAE text JSONL.

Examples (run from the repository root):
    python scripts/sae-pretrain/prepare_msmarco.py --version 1
    python scripts/sae-pretrain/prepare_msmarco.py --version 2 --limit 100

Only doc.text from ir_datasets is exported. No title, query labels, additional
cleaning or deduplication is applied. A row limit limits the exported records;
ir_datasets may still download the collection before yielding its first document.
"""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from text_export import default_data_dir, write_jsonl


DATASET_IDS = {1: "msmarco-passage", 2: "msmarco-passage-v2"}


def iter_documents(dataset, source):
    """Convert documents lazily, preserving their order and exact API text."""
    for doc in dataset.docs_iter():
        yield {"id": str(doc.doc_id), "source": source, "role": "document", "text": doc.text}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", type=int, choices=tuple(DATASET_IDS), required=True,
                        help="Full MS MARCO passage collection: 1 or 2")
    parser.add_argument("--output", type=Path,
                        help="Output JSONL; defaults to repository data/msmarco-vVERSION.jsonl")
    parser.add_argument("--limit", type=int, help="Export only the first N documents; omit for the full collection")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing export")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.output is None:
        args.output = default_data_dir() / f"msmarco-v{args.version}.jsonl"
    return args


def main(argv=None):
    args = parse_args(argv)
    # Keep --help and argument validation independent of this optional package.
    try:
        import ir_datasets
    except ModuleNotFoundError as error:
        if error.name != "ir_datasets":
            raise
        raise SystemExit("MS MARCO export requires ir_datasets. Install it with: pip install ir_datasets") from error
    try:
        library_version = version("ir_datasets")
    except PackageNotFoundError:
        library_version = None

    dataset_id = DATASET_IDS[args.version]
    source = f"msmarco_v{args.version}"
    dataset = ir_datasets.load(dataset_id)
    provenance = dict(exporter="prepare_msmarco.py", dataset_id=dataset_id,
                      msmarco_version=args.version, ir_datasets_version=library_version,
                      source=source, role="document", text_field="doc.text",
                      text_processing="unchanged_from_ir_datasets", selection="full_collection_in_source_order")
    manifest = write_jsonl(iter_documents(dataset, source), args.output,
                           provenance=provenance, limit=args.limit, overwrite=args.overwrite)
    print(f"Exported {manifest['rows']:,} documents to {args.output}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
