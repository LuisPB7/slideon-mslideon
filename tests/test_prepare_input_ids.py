"""Offline checks for direct corpus-to-token preparation (no model downloads)."""

import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/sae-pretrain"
spec = importlib.util.spec_from_file_location("prepare_input_ids_test", SCRIPTS / "prepare_input_ids.py")
prepare = importlib.util.module_from_spec(spec)
with mock.patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    spec.loader.exec_module(prepare)


class PrepareInputIdsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="slideon-input-id-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path_patch = mock.patch.object(sys, "path", [str(SCRIPTS), *sys.path])
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def test_defaults_and_argument_validation(self):
        for source in ("msmarco-v1", "msmarco-v2", "fineweb"):
            args = prepare.parse_args(["--source", source])
            self.assertEqual(args.output, ROOT / "data" / f"{source}.pt")
            self.assertEqual((args.query_length, args.document_length), (192, 192))
            self.assertEqual(args.tokenize_batch_size, 512)
            self.assertEqual(args.shard_tokens, 8_000_000)
            self.assertEqual(args.config, ROOT / "scripts/training-configs/SlideOn-SAE/config.json")
        cases = [
            ["--source", "jsonl"],
            ["--source", "msmarco-v1", "--texts-jsonl", "texts.jsonl"],
            ["--source", "jsonl", "--texts-jsonl", "texts.pt"],
            ["--source", "jsonl", "--texts-jsonl", "texts.jsonl", "./texts.jsonl"],
            ["--source", "msmarco-v1", "--revision", "release"],
            ["--source", "fineweb", "--revision", ""],
            ["--source", "fineweb", "--output", "texts.jsonl"],
            ["--source", "ir-datasets"],
            ["--source", "huggingface", "--dataset-id", ""],
            ["--source", "fineweb", "--dataset-id", "wrong/corpus"],
            ["--source", "msmarco-v1", "--text-field", "body"],
            ["--source", "ir-datasets", "--dataset-id", "beir/scifact", "--split", "train"],
            ["--source", "huggingface", "--dataset-id", "org/corpus", "--text-field", ""],
            ["--source", "fineweb", "--source-label", "override"],
        ]
        cases.extend(["--source", "fineweb", flag, "0"] for flag in (
            "--query-length", "--document-length", "--tokenize-batch-size", "--shard-tokens", "--limit"))
        for argv in cases:
            with self.subTest(argv=argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                prepare.parse_args(argv)

    def test_generic_ir_datasets_dispatches_exact_id_and_keeps_document_text(self):
        dataset_id = "beir/scifact"
        args = prepare.parse_args(["--source", "ir-datasets", "--dataset-id", dataset_id])
        self.assertEqual(args.output, ROOT / "data/beir-scifact.pt")
        load = mock.Mock(return_value=SimpleNamespace(docs_iter=lambda: iter([
            SimpleNamespace(doc_id=17, text="raw scientific text", title="never prepend")])) )
        with mock.patch.dict(sys.modules, {"ir_datasets": SimpleNamespace(load=load)}):
            records, provenance = prepare.source_records(args)
            self.assertEqual(list(records), [{"id": "17", "source": dataset_id,
                                            "role": "document", "text": "raw scientific text"}])
        load.assert_called_once_with(dataset_id)
        self.assertEqual(provenance["dataset_id"], dataset_id)
        self.assertNotIn("msmarco_version", provenance)

    def test_generic_huggingface_streams_custom_fields_with_pinned_revision_and_fallback_ids(self):
        args = prepare.parse_args(["--source", "huggingface", "--dataset-id", "org/corpus",
            "--dataset-config", "subset", "--split", "validation", "--text-field", "body",
            "--id-field", "key", "--source-label", "custom_web", "--revision", "release"])
        self.assertEqual(args.output, ROOT / "data/org-corpus.pt")
        hub = SimpleNamespace(dataset_info=mock.Mock(return_value=SimpleNamespace(sha="b" * 40)))
        load = mock.Mock(return_value=iter([
            {"key": 42, "body": " raw body ", "title": "never add", "text": "wrong field"},
            {"body": "without id"}, {"key": None, "body": "null id"}]))
        with mock.patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=load),
                                          "huggingface_hub": SimpleNamespace(HfApi=lambda: hub)}):
            records, provenance = prepare.source_records(args)
            self.assertEqual(list(records), [
                {"id": "42", "source": "custom_web", "role": "document", "text": " raw body "},
                {"id": "1", "source": "custom_web", "role": "document", "text": "without id"},
                {"id": "2", "source": "custom_web", "role": "document", "text": "null id"}])
        hub.dataset_info.assert_called_once_with("org/corpus", revision="release")
        load.assert_called_once_with("org/corpus", "subset", split="validation", streaming=True, revision="b" * 40)
        self.assertEqual(provenance["revision"], "b" * 40)
        self.assertEqual(provenance["text_field"], "body")
        self.assertEqual(provenance["id_field"], "key")

    def test_generic_huggingface_defaults_resolve_main_and_use_text_field(self):
        args = prepare.parse_args(["--source", "huggingface", "--dataset-id", "org/corpus"])
        hub = SimpleNamespace(dataset_info=mock.Mock(return_value=SimpleNamespace(sha="c" * 40)))
        load = mock.Mock(return_value=iter([{"text": "body"}]))
        with mock.patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=load),
                                          "huggingface_hub": SimpleNamespace(HfApi=lambda: hub)}):
            records, _ = prepare.source_records(args)
            self.assertEqual(list(records), [{"id": "0", "source": "org/corpus",
                                            "role": "document", "text": "body"}])
        hub.dataset_info.assert_called_once_with("org/corpus", revision="main")
        load.assert_called_once_with("org/corpus", None, split="train", streaming=True, revision="c" * 40)

    def test_help_does_not_import_model_or_download_dependencies(self):
        # -S removes site-packages, so Torch/Transformers/datasets cannot be imported.
        result = subprocess.run([sys.executable, "-S", str(SCRIPTS / "prepare_input_ids.py"), "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--tokenize-batch-size", result.stdout)
        self.assertIn("msmarco-v2", result.stdout)

    def test_msmarco_sources_use_raw_text_skip_blanks_and_keep_ids(self):
        docs = [SimpleNamespace(doc_id=12, text=" raw text\n", title="NEVER ADD"),
                SimpleNamespace(doc_id=13, text=" \n", title="NEVER ADD"),
                SimpleNamespace(doc_id=14, text="last", title="NEVER ADD")]
        for corpus_version, dataset_id in ((1, "msmarco-passage"), (2, "msmarco-passage-v2")):
            with self.subTest(version=corpus_version):
                loader = mock.Mock(return_value=SimpleNamespace(docs_iter=lambda: iter(docs)))
                args = prepare.parse_args(["--source", f"msmarco-v{corpus_version}"])
                with mock.patch.dict(sys.modules, {"ir_datasets": SimpleNamespace(load=loader)}), \
                     mock.patch.object(prepare, "version", return_value="test-version"):
                    rows, provenance = prepare.source_records(args)
                    rows = list(rows)
                loader.assert_called_once_with(dataset_id)
                self.assertEqual(rows, [
                    {"id": "12", "source": f"msmarco_v{corpus_version}", "role": "document", "text": " raw text\n"},
                    {"id": "14", "source": f"msmarco_v{corpus_version}", "role": "document", "text": "last"},
                ])
                self.assertEqual(provenance["counts"], {"read_rows": 3, "skipped_blank": 1})
                self.assertEqual(provenance["dataset_id"], dataset_id)
                self.assertEqual(provenance["ir_datasets_version"], "test-version")

    def test_fineweb_streams_pinned_or_requested_revision_and_preserves_raw_text(self):
        import prepare_fineweb
        for requested in (None, "release"):
            with self.subTest(requested=requested):
                hub = SimpleNamespace(dataset_info=mock.Mock(return_value=SimpleNamespace(sha="a" * 40)))
                load_dataset = mock.Mock(return_value=iter([
                    {"id": "urn:uuid:test", "text": " web text\n", "title": "NEVER ADD"}]))
                argv = ["--source", "fineweb"]
                if requested is not None:
                    argv += ["--revision", requested]
                with mock.patch.dict(sys.modules, {
                    "datasets": SimpleNamespace(load_dataset=load_dataset),
                    "huggingface_hub": SimpleNamespace(HfApi=lambda: hub),
                }):
                    rows, provenance = prepare.source_records(prepare.parse_args(argv))
                    self.assertEqual(list(rows), [{"id": "urn:uuid:test", "source": "fineweb",
                                                   "role": "document", "text": " web text\n"}])
                hub.dataset_info.assert_called_once_with(prepare_fineweb.DATASET_ID,
                    revision=requested or prepare_fineweb.DEFAULT_REVISION)
                load_dataset.assert_called_once_with("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
                                                     split="train", streaming=True, revision="a" * 40)
                self.assertEqual(provenance["revision"], "a" * 40)
                self.assertEqual(provenance["text_processing"], "unchanged_no_titles")

    def test_jsonl_preserves_source_role_id_duplicates_and_file_order(self):
        rows = [{"id": "7", "source": "generated", "role": "query", "text": " query\n"},
                {"source": "web", "text": "raw doc"}]
        paths = [self.root / "queries.jsonl", self.root / "web.jsonl"]
        for path, row in zip(paths, rows):
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        args = prepare.parse_args(["--source", "jsonl", "--texts-jsonl", *map(str, paths)])
        records, provenance = prepare.source_records(args)
        self.assertEqual(list(records), [rows[0], rows[0], rows[1], rows[1]])
        self.assertEqual(provenance["input_files"], [str(path.resolve()) for path in paths])
        self.assertEqual(provenance["blank_text_policy"], "reject")

    def test_jsonl_rejects_malformed_rows_and_does_not_silently_drop_blanks(self):
        path = self.root / "input.jsonl"
        for content in ("\n", "not json\n", "[]\n", "null\n"):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "input.jsonl:1"):
                    list(prepare.iter_jsonl_records([path]))
        # Field validation belongs to the shared writer; no user record disappears
        # in the reader before that strict validation takes place.
        row = {"source": "custom", "text": " "}
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.assertEqual(list(prepare.iter_jsonl_records([path])), [row])

    def test_encoder_uses_config_and_cpu_tokenizer_only(self):
        args = prepare.parse_args(["--source", "fineweb", "--query-length", "64", "--document-length", "256"])
        config = {"model": "SlideOn-SAE", "parameters": {"query_length": 8192,
                  "document_length": 8192, "backbone": "test/dense", "revision": "fixed"}}
        recipe = object()
        read_config = mock.Mock(return_value=config)
        make_recipe = mock.Mock(return_value=recipe)
        frozen_tokens = mock.Mock(return_value="tokenizer encoder")
        device = mock.Mock(return_value="cpu-device")
        with mock.patch.dict(sys.modules, {
            "torch": SimpleNamespace(device=device),
            "train_slideon_sae": SimpleNamespace(FrozenTokens=frozen_tokens,
                read_config=read_config, recipe_from_parameters=make_recipe),
        }):
            self.assertEqual(prepare.make_tokenizer_encoder(args), "tokenizer encoder")
        read_config.assert_called_once_with("sae", args.config)
        make_recipe.assert_called_once_with("slideon", {"query_length": 64, "document_length": 256,
                                                       "backbone": "test/dense", "revision": "fixed"})
        device.assert_called_once_with("cpu")
        frozen_tokens.assert_called_once_with(recipe, "cpu-device", load_encoder=False)
        self.assertEqual(config["parameters"]["query_length"], 8192)

    def test_main_forwards_batch_shard_and_limit_settings_and_lazy_provenance(self):
        def docs():
            yield SimpleNamespace(doc_id=1, text=" ")
            yield SimpleNamespace(doc_id=2, text="first")
            raise AssertionError("Read beyond the requested limit")
        encoder = object()
        output = self.root / "corpus.pt"
        captured = {}

        def write(records, output_path, actual_encoder, **kwargs):
            captured.update(kwargs)
            captured["records"] = list(itertools.islice(records, kwargs["limit"]))
            self.assertEqual(output_path, output)
            self.assertIs(actual_encoder, encoder)
            return "writer result"

        writer = mock.Mock(side_effect=write)
        with mock.patch.dict(sys.modules, {
            "ir_datasets": SimpleNamespace(load=lambda _: SimpleNamespace(docs_iter=docs)),
            "slideon.sae_tokens": SimpleNamespace(write_pretokenized=writer),
        }), mock.patch.object(prepare, "make_tokenizer_encoder", return_value=encoder):
            result = prepare.main(["--source", "msmarco-v1", "--output", str(output), "--limit", "1",
                                   "--tokenize-batch-size", "64", "--shard-tokens", "1024"])
        self.assertEqual(result, "writer result")
        writer.assert_called_once()
        self.assertEqual(captured["tokenize_batch_size"], 64)
        self.assertEqual(captured["shard_tokens"], 1024)
        self.assertEqual(captured["records"][0]["id"], "2")
        self.assertEqual(captured["provenance"]["limit"], 1)
        self.assertEqual(captured["provenance"]["counts"], {"read_rows": 2, "skipped_blank": 1})
        self.assertEqual(list(self.root.iterdir()), [])  # No intermediate JSONL.

    def test_existing_destination_fails_before_loading_tokenizer_or_data(self):
        output = self.root / "existing.pt"
        output.write_bytes(b"keep")
        with mock.patch.object(prepare, "make_tokenizer_encoder") as encoder, \
             mock.patch.object(prepare, "source_records") as records, self.assertRaises(FileExistsError):
            prepare.main(["--source", "fineweb", "--output", str(output)])
        encoder.assert_not_called()
        records.assert_not_called()
        self.assertEqual(output.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
