"""Export the original English SAE text pools without computing token embeddings.

The same selection helper serves the original cache preparation and this export:
filter mined negatives, sample seven, then deduplicate IDs within each source.
Only raw source/role/text records are written; pair relationships and teacher
scores do not enter the SAE objective. Completed compatible exports are reused.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts/training-configs/SlideOn-SAE/config.json"


def read_settings(config_path, *, seed=None, revision=None, dataset_revisions=None):
    """Read only settings that select texts, independent of training/tokenization."""
    config = json.loads(Path(config_path).read_text())
    if config.get("model") != "SlideOn-SAE" or config.get("stage") != "sae":
        raise ValueError("Expected the English SlideOn-SAE configuration")
    parameters = config["parameters"]
    dataset = parameters["dataset_prefix"] + "en"
    pins = parameters.get("dataset_revisions", {})
    if dataset_revisions is not None:
        pins = json.loads(Path(dataset_revisions).read_text())
    if not isinstance(pins, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or not value.strip()
        for key, value in pins.items()
    ):
        raise ValueError("dataset_revisions must map dataset IDs to revision strings")
    if set(pins) - {dataset}:
        raise ValueError(f"The English export uses only {dataset}; unexpected dataset revision pins")
    settings = dict(
        dataset=dataset,
        requested_revision=revision if revision is not None else pins.get(dataset, "main"),
        splits=parameters["splits"],
        seed=parameters["seed"] if seed is None else seed,
        nv_threshold=parameters["nv_threshold"],
        num_negatives=parameters["num_negatives"],
        sampled_negatives=parameters["sampled_negatives"],
    )
    if not isinstance(settings["requested_revision"], str) or not settings["requested_revision"].strip():
        raise ValueError("Dataset revision must be a nonempty string")
    splits = settings["splits"]
    if (not isinstance(splits, list) or not splits or
            any(not isinstance(split, str) or not split.endswith("_en") or "/" in split for split in splits) or
            len(set(splits)) != len(splits)):
        raise ValueError("splits must contain unique English source names")
    for name in ("seed", "num_negatives", "sampled_negatives"):
        if type(settings[name]) is not int or settings[name] < (1 if name == "num_negatives" else 0):
            raise ValueError(f"{name} must be a valid nonnegative integer")
    if settings["sampled_negatives"] > settings["num_negatives"]:
        raise ValueError("sampled_negatives must not exceed num_negatives")
    threshold = settings["nv_threshold"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("nv_threshold must be finite and nonnegative")
    return settings


def _fingerprint(path):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def export_texts(settings, output, *, load_table, resolve_revision, select_texts):
    """Write or validate an export; injectable data access keeps this CPU-testable.

load_table(dataset, kind, split, revision) returns one HF table. resolve_revision
returns an immutable SHA. select_texts is the shared original filter_texts helper.
"""
    output = Path(output)
    manifest_path = Path(str(output) + ".manifest.json")
    partial = Path(str(output) + ".partial")
    manifest_partial = Path(str(manifest_path) + ".partial")
    if partial.exists() or manifest_partial.exists():
        raise FileExistsError(f"Incomplete export beside {output}; choose another output or remove its partial files")
    if output.exists() or manifest_path.exists():
        if not output.is_file() or not manifest_path.is_file():
            raise ValueError(f"Incomplete export: both {output} and its manifest are required")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("version") != 1 or manifest.get("complete") is not True or
                manifest.get("settings") != settings or not manifest.get("revision")):
            raise ValueError("Existing export is incompatible with these selection settings; choose another output")
        size, digest = _fingerprint(output)
        if size != manifest.get("bytes") or digest != manifest.get("sha256"):
            raise ValueError("Existing JSONL differs from its manifest; choose another output")
        print(f"Reusing {output}: {manifest['rows']:,} texts at {manifest['revision']}", flush=True)
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    revision = resolve_revision(settings["dataset"], settings["requested_revision"])
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("Dataset revision resolver did not return an immutable revision")
    manifest = dict(version=1, complete=False, settings=settings, revision=revision,
                    sources=[], rows=0, bytes=0, sha256=None)
    digest = hashlib.sha256()
    # Track only temporary files created by this call. Failed exports never look
    # complete, and existing files are never truncated or silently replaced.
    owned_partials = []
    try:
        with partial.open("xb") as stream:
            owned_partials.append(partial)
            for split in settings["splits"]:
                print(f"Exporting {split} from {settings['dataset']}@{revision}", flush=True)
                tables = {kind: load_table(settings["dataset"], kind, split, revision)
                          for kind in ("scores", "queries", "documents")}
                pools = select_texts(**tables, **{name: settings[name] for name in
                                    ("seed", "nv_threshold", "num_negatives", "sampled_negatives")})
                counts = dict(source=split, documents=0, queries=0)
                for kind, role in (("documents", "document"), ("queries", "query")):
                    if not pools[kind]:
                        raise ValueError(f"No usable texts in {split}/{kind}")
                    for text in pools[kind]:
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(f"Invalid selected text in {split}/{kind}; expected a nonblank string")
                        record = dict(source=split, role=role, text=text)
                        encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                        stream.write(encoded)
                        digest.update(encoded)
                        manifest["bytes"] += len(encoded)
                        manifest["rows"] += 1
                        counts[kind] += 1
                manifest["sources"].append(counts)
                print(f"  {counts['documents']:,} documents, {counts['queries']:,} queries", flush=True)
                # Release source tables and pools before loading the next source.
                del pools, tables
        manifest.update(complete=True, sha256=digest.hexdigest())
        with manifest_partial.open("x") as stream:
            owned_partials.append(manifest_partial)
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        # The manifest is the completion marker. An interrupted rename leaves an
        # incomplete pair which the reuse checks above reject on the next launch.
        partial.replace(output)
        manifest_partial.replace(manifest_path)
    except BaseException:
        for path in owned_partials:
            path.unlink(missing_ok=True)
        raise
    print(f"Exported {manifest['rows']:,} texts to {output}; no token embeddings written", flush=True)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("data/slideon-original-texts.jsonl"))
    parser.add_argument("--seed", type=int, help="Override the original negative-sampling seed")
    parser.add_argument("--revision", help="Dataset revision; defaults to configured pin or main")
    parser.add_argument("--dataset-revisions", type=Path, help="Optional JSON mapping dataset IDs to exact revisions")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    settings = read_settings(args.config, seed=args.seed, revision=args.revision,
                             dataset_revisions=args.dataset_revisions)
    # Load the pure selection helper directly: importing slideon.__init__ would
    # unnecessarily import the model stack for this CPU-only text export.
    spec = importlib.util.spec_from_file_location("slideon_sae_data", REPO_ROOT / "src/slideon/sae_data.py")
    selection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selection)

    def resolve_revision(dataset, revision):
        from huggingface_hub import HfApi
        return HfApi().dataset_info(dataset, revision=revision).sha

    def load_table(dataset, kind, split, revision):
        from datasets import load_dataset
        return load_dataset(dataset, name=kind, revision=revision,
                            data_files=f"{kind}/{split}-*", split="train", verification_mode="no_checks")

    export_texts(settings, args.output, load_table=load_table,
                 resolve_revision=resolve_revision, select_texts=selection.filter_texts)


if __name__ == "__main__":
    main()
