"""Offline checks for corpus exporters; no dataset downloads or models needed."""

import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/sae-pretrain"


def load_script(name):
    spec = importlib.util.spec_from_file_location("slideon_export_test_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
        spec.loader.exec_module(module)
    return module


text_export = load_script("text_export")
msmarco = load_script("prepare_msmarco")
fineweb = load_script("prepare_fineweb")
artificial = load_script("prepare_artificial_queries")


class TextExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="slideon-text-export-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @staticmethod
    def row(identifier="1", text="  café\nraw text  "):
        return {"id": identifier, "source": "corpus", "role": "document", "text": text}

    @staticmethod
    def read_rows(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_writer_preserves_text_skips_blanks_and_records_integrity_metadata(self):
        output = self.root / "nested" / "texts.jsonl"
        records = [dict(self.row(), unused_title="MUST NOT BE ADDED"),
                   self.row("2", " \n\t"), self.row("3", "final")]
        provenance = {"dataset_id": "test/corpus"}
        manifest = text_export.write_jsonl(records, output, provenance=provenance)
        self.assertEqual(self.read_rows(output), [self.row(), self.row("3", "final")])
        self.assertEqual(manifest["rows"], 2)
        self.assertEqual(manifest["read_rows"], 3)
        self.assertEqual(manifest["skipped_blank"], 1)
        self.assertEqual(manifest["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(manifest["bytes"], output.stat().st_size)
        self.assertEqual(manifest["provenance"], provenance)
        self.assertEqual(manifest["text_processing"], "unchanged_no_titles")
        self.assertEqual(json.loads(Path(str(output) + ".manifest.json").read_text()), manifest)

    def test_limit_counts_written_rows_without_consuming_an_extra_record(self):
        def records():
            yield self.row("blank", "  ")
            yield self.row("first", "one")
            yield self.row("second", "two")
            raise AssertionError("The writer consumed a record beyond --limit")

        output = self.root / "limited.jsonl"
        manifest = text_export.write_jsonl(records(), output, provenance={}, limit=2)
        self.assertEqual([row["id"] for row in self.read_rows(output)], ["first", "second"])
        self.assertEqual(manifest["limit"], 2)
        self.assertEqual(manifest["read_rows"], 3)

    def test_writer_refuses_overwrite_and_preserves_existing_files_on_failed_replacement(self):
        output = self.root / "texts.jsonl"
        text_export.write_jsonl([self.row()], output, provenance={})
        sidecar = Path(str(output) + ".manifest.json")
        original = output.read_bytes(), sidecar.read_bytes()
        with self.assertRaises(FileExistsError):
            text_export.write_jsonl([self.row("new")], output, provenance={})
        with self.assertRaises(ValueError):
            text_export.write_jsonl([self.row("new"), self.row("bad", None)], output,
                                   provenance={}, overwrite=True)
        self.assertEqual((output.read_bytes(), sidecar.read_bytes()), original)
        text_export.write_jsonl([self.row("replacement", "changed")], output,
                               provenance={}, overwrite=True)
        self.assertEqual(self.read_rows(output), [self.row("replacement", "changed")])

    def test_invalid_or_empty_exports_never_publish_a_final_file(self):
        cases = [[], [self.row("blank", " \n")], [self.row("bad", 42)],
                 [dict(self.row(), id=1)], [dict(self.row(), source="")],
                 [dict(self.row(), role="passage")]]
        for index, records in enumerate(cases):
            with self.subTest(records=records):
                output = self.root / f"invalid-{index}.jsonl"
                with self.assertRaises(ValueError):
                    text_export.write_jsonl(records, output, provenance={})
                self.assertFalse(output.exists())
                self.assertFalse(Path(str(output) + ".manifest.json").exists())

    def test_msmarco_exports_exact_document_text_for_both_collections(self):
        docs = [SimpleNamespace(doc_id="p1", text=" raw passage\n", title="ignored title"),
                SimpleNamespace(doc_id="p2", text="another", url="ignored URL")]
        for version, dataset_id in ((1, "msmarco-passage"), (2, "msmarco-passage-v2")):
            with self.subTest(version=version):
                dataset = SimpleNamespace(docs_iter=lambda: iter(docs))
                load = mock.Mock(return_value=dataset)
                output = self.root / f"msmarco-{version}.jsonl"
                with mock.patch.dict(sys.modules, {"ir_datasets": SimpleNamespace(load=load)}), \
                     mock.patch.object(msmarco, "version", return_value="test-version"):
                    manifest = msmarco.main(["--version", str(version), "--output", str(output)])
                load.assert_called_once_with(dataset_id)
                self.assertEqual(self.read_rows(output), [
                    {"id": doc.doc_id, "source": f"msmarco_v{version}", "role": "document", "text": doc.text}
                    for doc in docs
                ])
                self.assertEqual(manifest["provenance"]["dataset_id"], dataset_id)

    def test_fineweb_streams_pinned_subset_and_ignores_titles(self):
        records = [{"id": "urn:uuid:test", "text": " raw web text\n", "title": "ignored"}]
        resolved_revision = "a" * 40
        hub = SimpleNamespace(dataset_info=mock.Mock(return_value=SimpleNamespace(sha=resolved_revision)))
        load_dataset = mock.Mock(return_value=iter(records))
        output = self.root / "fineweb.jsonl"
        with mock.patch.dict(sys.modules, {
            "datasets": SimpleNamespace(load_dataset=load_dataset),
            "huggingface_hub": SimpleNamespace(HfApi=lambda: hub),
        }):
            manifest = fineweb.main(["--output", str(output), "--revision", "release"])
        hub.dataset_info.assert_called_once_with("HuggingFaceTB/smollm-corpus", revision="release")
        load_dataset.assert_called_once_with("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
                                             split="train", streaming=True, revision=resolved_revision)
        self.assertEqual(self.read_rows(output), [
            {"id": "urn:uuid:test", "source": "fineweb", "role": "document", "text": " raw web text\n"}
        ])
        self.assertEqual(manifest["provenance"]["revision"], resolved_revision)

    def write_pickle(self, part, contents):
        path = self.root / f"top5_doct5queries-_part_{part}.pkl"
        with path.open("wb") as stream:
            pickle.dump(contents, stream)
        return path

    def test_artificial_queries_preserve_ids_text_and_parts_without_deduplicating(self):
        self.write_pickle(0, {17: "  first query\n", 18: " "})
        self.write_pickle(4, {17: "second query"})
        output = self.root / "out"
        artificial.main(["--input-dir", str(self.root), "--output-dir", str(output), "--parts", "0", "4"])
        first = output / "msmarco-v1-artificial-queries-0.jsonl"
        last = output / "msmarco-v1-artificial-queries-4.jsonl"
        self.assertEqual(self.read_rows(first), [
            {"id": "17", "source": "msmarco_v1_artificial_queries", "role": "query", "text": "  first query\n"}
        ])
        self.assertEqual(self.read_rows(last), [
            {"id": "17", "source": "msmarco_v1_artificial_queries", "role": "query", "text": "second query"}
        ])
        self.assertEqual(sorted(path.name for path in output.glob("*.jsonl")), [first.name, last.name])
        self.assertEqual(json.loads(Path(str(first) + ".manifest.json").read_text())["skipped_blank"], 1)

    def test_artificial_query_pickles_require_integer_keys_and_string_values(self):
        for contents in (["query"], {"17": "query"}, {True: "query"}, {17: ["query"]}):
            with self.subTest(contents=contents):
                path = self.write_pickle(0, contents)
                with self.assertRaises(ValueError):
                    list(artificial.iter_queries(path))

    def test_artificial_query_limit_is_per_part_and_filename_matches_exactly(self):
        for part in (0, 1):
            self.write_pickle(part, {1: "first", 2: "second"})
        output = self.root / "out"
        artificial.main(["--input-dir", str(self.root), "--output-dir", str(output),
                         "--parts", "0", "1", "--limit", "1"])
        for part in (0, 1):
            self.assertEqual(len(self.read_rows(output / f"msmarco-v1-artificial-queries-{part}.jsonl")), 1)
        with self.assertRaises(FileNotFoundError):
            artificial.main(["--input-dir", str(self.root), "--output-dir", str(output), "--parts", "2"])

    def test_default_paths_and_argument_validation(self):
        self.assertEqual(text_export.default_data_dir(), ROOT / "data")
        for version in (1, 2):
            self.assertEqual(msmarco.parse_args(["--version", str(version)]).output,
                             ROOT / "data" / f"msmarco-v{version}.jsonl")
        self.assertEqual(fineweb.parse_args([]).output, ROOT / "data/fineweb.jsonl")
        self.assertRegex(fineweb.DEFAULT_REVISION, r"^[0-9a-f]{40}$")
        args = artificial.parse_args([])
        self.assertEqual(args.input_dir, ROOT / "data")
        self.assertEqual(args.output_dir, ROOT / "data")
        self.assertEqual(args.parts, [0, 1, 2, 3, 4])
        for module, prefix in ((msmarco, ["--version", "1"]), (fineweb, []), (artificial, [])):
            with self.subTest(module=module.__name__), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                module.parse_args([*prefix, "--limit", "0"])
        for parts in (["0", "0"], ["5"]):
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                artificial.parse_args(["--parts", *parts])


if __name__ == "__main__":
    unittest.main()
