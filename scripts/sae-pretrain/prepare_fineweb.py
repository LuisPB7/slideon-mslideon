"""Export SmolLM's FineWeb-Edu-Dedup texts to the SAE JSONL contract.

Only the source's existing text field is copied. Titles and metadata are never
prepended, and the dataset is streamed instead of downloaded into an Arrow table.
"""

import argparse
from pathlib import Path

from text_export import default_data_dir, write_jsonl


DATASET_ID = "HuggingFaceTB/smollm-corpus"
DATASET_CONFIG = "fineweb-edu-dedup"
DEFAULT_REVISION = "3ba9d605774198c5868892d7a8deda78031a781f"


def iter_documents(dataset):
    """Preserve the original ID and raw text, ignoring all other dataset fields."""
    for row in dataset:
        yield {"id": row["id"], "source": "fineweb", "role": "document", "text": row["text"]}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_data_dir() / "fineweb.jsonl",
                        help="Output JSONL (default: repository data/fineweb.jsonl)")
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="HF dataset commit or revision; resolved to a commit before streaming")
    parser.add_argument("--limit", type=positive_int,
                        help="Export at most this many nonblank texts; omitted means the full subset")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing completed output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Keep --help available before installation of the optional download tools.
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as error:
        raise SystemExit("Install the exporter dependencies with: pip install datasets huggingface-hub") from error

    revision = HfApi().dataset_info(DATASET_ID, revision=args.revision).sha
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True, revision=revision)
    provenance = dict(exporter="prepare_fineweb.py", dataset_id=DATASET_ID,
                      dataset_config=DATASET_CONFIG, split="train", revision=revision,
                      requested_revision=args.revision, text_field="text", source="fineweb", role="document")
    return write_jsonl(iter_documents(dataset), args.output, provenance=provenance,
                       limit=args.limit, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
