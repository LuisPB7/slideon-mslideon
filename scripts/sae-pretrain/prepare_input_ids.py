r"""Stream texts into compact pretokenized SAE input files, without embeddings.

Examples (run from the repository root):
    python scripts/sae-pretrain/prepare_input_ids.py --source msmarco-v1
    python scripts/sae-pretrain/prepare_input_ids.py --source msmarco-v2
    python scripts/sae-pretrain/prepare_input_ids.py --source fineweb
    python scripts/sae-pretrain/prepare_input_ids.py --source ir-datasets \
        --dataset-id msmarco-passage-v2
    python scripts/sae-pretrain/prepare_input_ids.py --source huggingface \
        --dataset-id HuggingFaceTB/smollm-corpus --dataset-config fineweb-edu-dedup
    python scripts/sae-pretrain/prepare_input_ids.py --source jsonl \
        --texts-jsonl data/msmarco-v1-artificial-queries-0.jsonl \
        --output data/msmarco-v1-artificial-queries-0.pt

The .pt output describes bounded packed-token shards beside it. Attention masks
are reconstructed from stored sequence boundaries; padding and dense embeddings
are never written. Query/document limits default to the 192-token settings in
run_pretraining.sh and must match the subsequent training configuration.

FineWeb is streamed directly from Hugging Face. MS MARCO uses ir_datasets, which
may cache the original collection download, but no intermediate JSONL is created.
"""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re

from text_export import default_data_dir


SOURCES = ("msmarco-v1", "msmarco-v2", "fineweb", "jsonl", "ir-datasets", "huggingface")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "training-configs/SlideOn-SAE/config.json"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--dataset-id", help="Required dataset name for ir-datasets or huggingface sources")
    parser.add_argument("--dataset-config", help="Optional Hugging Face subset/config; only for --source huggingface")
    parser.add_argument("--split", help="Hugging Face split (default: train); only for --source huggingface")
    parser.add_argument("--text-field", help="Hugging Face raw-text field (default: text); only for --source huggingface")
    parser.add_argument("--id-field", help="Hugging Face ID field (default: id); missing IDs use row numbers")
    parser.add_argument("--source-label", help="Optional stored source label for generic datasets (default: dataset ID)")
    parser.add_argument("--texts-jsonl", type=Path, nargs="+",
                        help="Existing raw-text JSONL files; only for --source jsonl")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="SAE config supplying the tokenizer revision and role prompts")
    parser.add_argument("--output", type=Path,
                        help="Output .pt descriptor; defaults to data/SOURCE.pt, dataset-name.pt, or texts.pt")
    parser.add_argument("--query-length", type=positive_int, default=192,
                        help="Maximum encoded query length including prompt/special tokens (default: 192)")
    parser.add_argument("--document-length", type=positive_int, default=192,
                        help="Maximum encoded document length including prompt/special tokens (default: 192)")
    parser.add_argument("--tokenize-batch-size", type=positive_int, default=512,
                        help="Texts tokenized together on CPU (default: 512)")
    parser.add_argument("--shard-tokens", type=positive_int, default=8_000_000,
                        help="Target maximum token count per packed shard (default: 8000000)")
    parser.add_argument("--limit", type=positive_int,
                        help="Maximum exported nonblank records; omit for the full collection")
    parser.add_argument("--revision", help="HF revision: pinned release for FineWeb, main for generic huggingface; resolved to a commit")
    args = parser.parse_args(argv)
    generic = args.source in ("ir-datasets", "huggingface")
    if generic and (not args.dataset_id or not args.dataset_id.strip()):
        parser.error(f"--source {args.source} requires a nonempty --dataset-id")
    if not generic and args.dataset_id is not None:
        parser.error("--dataset-id requires --source ir-datasets or huggingface")
    if args.source_label is not None and (not generic or not args.source_label.strip()):
        parser.error("--source-label must be nonempty and requires a generic dataset source")
    for name in ("dataset_config", "split", "text_field", "id_field"):
        value = getattr(args, name)
        if value is not None and (args.source != "huggingface" or not value.strip()):
            parser.error(f"--{name.replace('_', '-')} must be nonempty and requires --source huggingface")
    if args.source == "jsonl":
        if not args.texts_jsonl:
            parser.error("--source jsonl requires --texts-jsonl")
        paths = [path.resolve() for path in args.texts_jsonl]
        if len(set(paths)) != len(paths):
            parser.error("--texts-jsonl must contain distinct files")
        if any(path.suffix.lower() != ".jsonl" for path in paths):
            parser.error("--texts-jsonl inputs must end in .jsonl")
    elif args.texts_jsonl:
        parser.error("--texts-jsonl requires --source jsonl")
    if args.revision is not None and args.source not in ("fineweb", "huggingface"):
        parser.error("--revision applies only to FineWeb or generic Hugging Face datasets")
    if args.revision is not None and not args.revision.strip():
        parser.error("--revision must be nonempty")
    if args.output is None:
        if generic:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.dataset_id).strip(".-")
            if not name:
                parser.error("--dataset-id must contain letters or numbers, or specify --output")
        else:
            name = "texts" if args.source == "jsonl" else args.source
        args.output = default_data_dir() / f"{name}.pt"
    if args.output.suffix.lower() != ".pt":
        parser.error("--output must end in .pt")
    return args


def iter_huggingface_records(dataset, source, text_field="text", id_field="id"):
    """Copy one text field, with deterministic fallback IDs for datasets without IDs."""
    for index, row in enumerate(dataset):
        identifier = row.get(id_field)
        if identifier is None or not str(identifier).strip():
            identifier = index
        yield dict(id=str(identifier), source=source, role="document", text=row[text_field])


def iter_jsonl_records(paths):
    """Read raw JSONL strictly; the packed writer validates each record's contract."""
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError) as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON record") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield record


def nonblank_corpus_records(records, counts):
    """Match existing corpus exporters: preserve raw text, but count/skip blanks."""
    for record in records:
        counts["read_rows"] += 1
        if not isinstance(record, dict):
            raise ValueError("Corpus record must be a dictionary")
        for key in ("id", "source"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"Corpus {key} must be a nonempty string")
        if record.get("role") not in ("query", "document"):
            raise ValueError("Corpus role must be query or document")
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError("Corpus text must be a string")
        if not text.strip():
            counts["skipped_blank"] += 1
            continue
        yield record


def source_records(args):
    """Open one source lazily; corpus-specific code stays in its existing exporter."""
    provenance = dict(exporter="prepare_input_ids.py", source_kind=args.source,
                      limit=args.limit, text_processing="unchanged_no_titles")
    if args.source == "jsonl":
        for path in args.texts_jsonl:
            if not path.is_file():
                raise FileNotFoundError(path)
        provenance.update(input_files=[str(path.resolve()) for path in args.texts_jsonl],
                          selection="all_records_in_file_order", blank_text_policy="reject")
        return iter_jsonl_records(args.texts_jsonl), provenance

    counts = dict(read_rows=0, skipped_blank=0)
    provenance.update(counts=counts, blank_text_policy="skip_and_count")
    if args.source.startswith("msmarco-v") or args.source == "ir-datasets":
        import prepare_msmarco
        try:
            import ir_datasets
        except ModuleNotFoundError as error:
            if error.name != "ir_datasets":
                raise
            raise SystemExit("ir_datasets preparation requires: pip install ir_datasets") from error
        corpus_version = int(args.source[-1]) if args.source.startswith("msmarco-v") else None
        dataset_id = prepare_msmarco.DATASET_IDS[corpus_version] if corpus_version else args.dataset_id
        source = f"msmarco_v{corpus_version}" if corpus_version else (args.source_label or dataset_id)
        try:
            library_version = version("ir_datasets")
        except PackageNotFoundError:
            library_version = None
        dataset = ir_datasets.load(dataset_id)
        records = prepare_msmarco.iter_documents(dataset, source)
        provenance.update(dataset_id=dataset_id, source=source, role="document",
                          ir_datasets_version=library_version, text_field="doc.text",
                          selection="full_collection_in_source_order")
        if corpus_version:
            provenance["msmarco_version"] = corpus_version
    else:
        import prepare_fineweb
        from datasets import load_dataset
        from huggingface_hub import HfApi
        preset = args.source == "fineweb"
        dataset_id = prepare_fineweb.DATASET_ID if preset else args.dataset_id
        dataset_config = prepare_fineweb.DATASET_CONFIG if preset else args.dataset_config
        split = "train" if preset else (args.split or "train")
        text_field = "text" if preset else (args.text_field or "text")
        id_field = "id" if preset else (args.id_field or "id")
        source = "fineweb" if preset else (args.source_label or dataset_id)
        requested = args.revision or (prepare_fineweb.DEFAULT_REVISION if preset else "main")
        revision = HfApi().dataset_info(dataset_id, revision=requested).sha
        dataset = load_dataset(dataset_id, dataset_config, split=split, streaming=True, revision=revision)
        records = (prepare_fineweb.iter_documents(dataset) if preset else
                   iter_huggingface_records(dataset, source, text_field, id_field))
        provenance.update(dataset_id=dataset_id, dataset_config=dataset_config, split=split,
                          requested_revision=requested, revision=revision, text_field=text_field,
                          id_field=id_field, source=source, role="document")
    return nonblank_corpus_records(records, counts), provenance


def make_tokenizer_encoder(args):
    """Load tokenizer metadata on CPU, never DenseOn weights or a GPU model."""
    import torch
    from train_slideon_sae import FrozenTokens, read_config, recipe_from_parameters
    config = read_config("sae", args.config)
    parameters = dict(config["parameters"])
    parameters.update(query_length=args.query_length, document_length=args.document_length)
    recipe = recipe_from_parameters(config["model"].removesuffix("-SAE").lower(), parameters)
    return FrozenTokens(recipe, torch.device("cpu"), load_encoder=False)


def main(argv=None):
    args = parse_args(argv)
    # Fail before downloads if the destination is already occupied. The writer
    # additionally guards partial outputs/shards and publishes completion atomically.
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; choose another --output")
    encoder = make_tokenizer_encoder(args)
    records, provenance = source_records(args)
    from slideon.sae_tokens import write_pretokenized
    return write_pretokenized(records, args.output, encoder,
                              tokenize_batch_size=args.tokenize_batch_size,
                              shard_tokens=args.shard_tokens,
                              provenance=provenance, limit=args.limit)


if __name__ == "__main__":
    main()
