"""Export local DocT5Query pickle parts to the SAE text JSONL contract.

Each trusted input pickle is a dictionary mapping integer IDs to query strings.
One whole part is loaded at a time; records are written without building a
second copy of the dictionary. Input query text and IDs are preserved.
"""

import argparse
from numbers import Integral
from pathlib import Path
import pickle

from text_export import default_data_dir, write_jsonl


SOURCE = "msmarco_v1_artificial_queries"


def iter_queries(path):
    """Load one trusted pickle dictionary, then yield individual query records."""
    path = Path(path)
    with path.open("rb") as stream:
        queries = pickle.load(stream)
    if not isinstance(queries, dict):
        raise ValueError(f"{path}: expected a dictionary mapping integer IDs to query strings")
    for identifier, query in queries.items():
        if not isinstance(identifier, Integral) or isinstance(identifier, bool):
            raise ValueError(f"{path}: query IDs must be integers, got {type(identifier).__name__}")
        if not isinstance(query, str):
            raise ValueError(f"{path}: query {identifier} must be a string")
        yield {"id": str(int(identifier)), "source": SOURCE, "role": "query", "text": query}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_data_dir(),
                        help="Directory containing top5_doct5queries-_part_PART.pkl (default: repository data/)")
    parser.add_argument("--output-dir", type=Path, default=default_data_dir(),
                        help="Destination for one JSONL per selected part (default: repository data/)")
    parser.add_argument("--parts", type=int, nargs="+", choices=range(5), default=list(range(5)),
                        help="Parts to export, from 0 through 4 (default: all five)")
    parser.add_argument("--limit", type=int,
                        help="Maximum number of nonblank queries written per part; omit for all")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if len(args.parts) != len(set(args.parts)):
        parser.error("--parts must not contain duplicates")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Validate every selected filename before starting a potentially large export.
    inputs = [(part, args.input_dir / f"top5_doct5queries-_part_{part}.pkl") for part in args.parts]
    for _, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing artificial-query pickle: {path}")
    for part, path in inputs:
        output = args.output_dir / f"msmarco-v1-artificial-queries-{part}.jsonl"
        write_jsonl(iter_queries(path), output, provenance={
            "type": "pickle", "input_file": str(path.resolve()), "part": part,
            "source": SOURCE, "role": "query",
        }, limit=args.limit, overwrite=args.overwrite)
        print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
