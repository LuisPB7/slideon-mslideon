"""CPU-only checks for exporting the original SAE text pools without embeddings.

The Hub is replaced at the table-loading boundary. Selection, JSONL writing,
provenance, completed-file reuse and failure handling run as production code.
No Torch, Transformers, datasets installation or GPU is required.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selection = load_module("original_sae_selection_test_target", ROOT / "src/slideon/sae_data.py")
exporter = load_module("original_sae_export_test_target", ROOT / "scripts/sae-pretrain/export_original_texts.py")


def tables():
    """Boundary/missing candidates precede twelve usable negative documents."""
    negative_ids = [f"n{index}" for index in range(12)]
    ids = ["positive", "boundary", "missing", *negative_ids]
    return {
        "scores": [{"query_id": "q", "document_ids": ids,
                    "scores": [100.0, 95.0, 1.0, *([50.0] * 12)]}],
        "queries": {"query_id": ["q"], "query": ["  café\nquery?  "]},
        "documents": {
            "document_id": ["positive", "boundary", *negative_ids],
            "document": ["  positive\ntext  ", "excluded boundary", *negative_ids],
        },
    }


class OriginalSAESelectionTests(unittest.TestCase):
    def test_strict_threshold_first_ten_and_seeded_seven_selection(self):
        result = selection.filter_texts(**tables())
        self.assertEqual(result, {
            "documents": ["  positive\ntext  ", "n1", "n0", "n4", "n9", "n6", "n5", "n8"],
            "queries": ["  café\nquery?  "],
        })

    def test_deduplication_is_by_id_not_equal_text(self):
        source = tables()
        source["queries"] = {"query_id": ["q0", "q1"], "query": ["same query", "same query"]}
        source["documents"]["document_id"].append("positive2")
        source["documents"]["document"].append("  positive\ntext  ")
        base = source["scores"][0]
        source["scores"] = [
            dict(base, query_id="q0"),
            dict(base, query_id="q0"),
            dict(base, query_id="q1", document_ids=["positive2", *base["document_ids"][1:]]),
        ]
        result = selection.filter_texts(**source, sampled_negatives=0)
        self.assertEqual(result["queries"], ["same query", "same query"])
        self.assertEqual(result["documents"], ["  positive\ntext  "] * 2)

    def test_each_source_resets_seed_without_mutating_global_rng(self):
        before = random.getstate()
        first = selection.filter_texts(**tables())
        second = selection.filter_texts(**tables())
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), before)

    def test_incomplete_negative_sets_or_missing_queries_are_excluded(self):
        source = tables()
        source["scores"][0]["document_ids"] = ["positive", *[f"n{i}" for i in range(9)]]
        source["scores"][0]["scores"] = [100.0, *([50.0] * 9)]
        self.assertEqual(selection.filter_texts(**source), {"documents": [], "queries": []})
        source = tables()
        source["scores"][0]["query_id"] = "unknown"
        self.assertEqual(selection.filter_texts(**source), {"documents": [], "queries": []})


class OriginalSAEExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="slideon-original-texts-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "data" / "original.jsonl"
        self.manifest_path = Path(str(self.output) + ".manifest.json")
        self.settings = dict(
            dataset="example/english", requested_revision="main", splits=["a_en", "b_en"],
            seed=42, nv_threshold=0.95, num_negatives=10, sampled_negatives=7,
        )
        self.revision = "a" * 40
        self.resolve = mock.Mock(return_value=self.revision)
        self.load = mock.Mock(side_effect=lambda dataset, kind, split, revision: tables()[kind])

    def export(self, **overrides):
        options = dict(load_table=self.load, resolve_revision=self.resolve,
                       select_texts=selection.filter_texts)
        options.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return exporter.export_texts(self.settings, self.output, **options)

    def test_export_preserves_text_roles_sources_and_records_provenance(self):
        result = self.export()
        raw = self.output.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        expected_pool = selection.filter_texts(**tables())
        expected_rows = [
            {"source": source, "role": role, "text": text}
            for source in ["a_en", "b_en"]
            for kind, role in [("documents", "document"), ("queries", "query")]
            for text in expected_pool[kind]
        ]
        self.assertEqual(rows, expected_rows)
        self.assertTrue(all(set(row) == {"source", "role", "text"} for row in rows))
        self.assertEqual(result, json.loads(self.manifest_path.read_text()))
        self.assertTrue(result["complete"])
        self.assertEqual(result["settings"], self.settings)
        self.assertEqual(result["revision"], self.revision)
        self.assertEqual(result["rows"], 18)
        self.assertEqual(result["bytes"], len(raw))
        self.assertEqual(result["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result["sources"], [
            {"source": "a_en", "documents": 8, "queries": 1},
            {"source": "b_en", "documents": 8, "queries": 1},
        ])
        self.resolve.assert_called_once_with("example/english", "main")
        self.assertEqual(len(self.load.call_args_list), 6)
        for call in self.load.call_args_list:
            self.assertEqual(call.args[0], "example/english")
            self.assertIn(call.args[1], ("scores", "queries", "documents"))
            self.assertIn(call.args[2], ("a_en", "b_en"))
            self.assertEqual(call.args[3], self.revision)
        self.assertFalse(list(self.output.parent.glob("*.partial")))

    def test_completed_export_reuses_verified_file_without_hub_access(self):
        manifest = self.export()
        before = self.output.stat().st_mtime_ns
        blocked = mock.Mock(side_effect=AssertionError("Completed export must be reusable offline"))
        actual = self.export(load_table=blocked, resolve_revision=blocked, select_texts=blocked)
        self.assertEqual(actual, manifest)
        self.assertEqual(self.output.stat().st_mtime_ns, before)
        blocked.assert_not_called()

    def test_reuse_rejects_selection_changes_and_same_size_corruption(self):
        self.export()
        self.settings["seed"] = 43
        with self.assertRaises((ValueError, FileExistsError)):
            self.export()
        self.settings["seed"] = 42
        raw = self.output.read_bytes()
        self.output.write_bytes(raw.replace(b"positive", b"negative", 1))
        self.assertEqual(len(raw), self.output.stat().st_size)
        with self.assertRaises((ValueError, FileExistsError)):
            self.export()

    def test_reuse_rejects_incomplete_manifest_or_missing_partner(self):
        manifest = self.export()
        manifest["complete"] = False
        self.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises((ValueError, FileExistsError)):
            self.export()
        self.manifest_path.unlink()
        with self.assertRaises((ValueError, FileExistsError)):
            self.export()

    def test_stale_partial_is_rejected_before_any_hub_work(self):
        self.output.parent.mkdir(parents=True)
        Path(str(self.output) + ".partial").write_text("interrupted export")
        with self.assertRaises((ValueError, FileExistsError)):
            self.export()
        self.resolve.assert_not_called()
        self.load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failure_or_invalid_selected_text_never_publishes_complete_export(self):
        def load_failure(dataset, kind, split, revision):
            if split == "b_en":
                raise RuntimeError("mocked download interruption")
            return tables()[kind]

        with self.assertRaisesRegex(RuntimeError, "mocked download interruption"):
            self.export(load_table=load_failure)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest_path.exists())
        for partial in self.output.parent.glob("*.partial"):
            partial.unlink()
        with self.assertRaises((ValueError, TypeError)):
            self.export(select_texts=lambda **kwargs: {"documents": [" \t "], "queries": ["query"]})
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest_path.exists())

    def test_read_settings_uses_original_config_and_explicit_revision_overrides(self):
        config_path = ROOT / "scripts/training-configs/SlideOn-SAE/config.json"
        settings = exporter.read_settings(config_path)
        self.assertEqual(settings["dataset"], "lightonai/embeddings-fine-tuning-filtered-en")
        self.assertEqual(settings["seed"], 42)
        self.assertEqual(settings["num_negatives"], 10)
        self.assertEqual(settings["sampled_negatives"], 7)
        self.assertEqual(len(settings["splits"]), 9)
        pinned = exporter.read_settings(config_path, seed=71, revision=self.revision)
        self.assertEqual(pinned["seed"], 71)
        self.assertEqual(pinned["requested_revision"], self.revision)

    def test_main_loads_source_files_at_one_immutable_revision_without_model_stack(self):
        config = json.loads((ROOT / "scripts/training-configs/SlideOn-SAE/config.json").read_text())
        config["parameters"]["splits"] = ["a_en", "b_en"]
        config_path = self.root / "small-config.json"
        config_path.write_text(json.dumps(config))
        hub = mock.Mock()
        hub.dataset_info.return_value = SimpleNamespace(sha=self.revision)
        load_dataset = mock.Mock(side_effect=lambda dataset, **kwargs: tables()[kwargs["name"]])
        modules = {
            "datasets": SimpleNamespace(load_dataset=load_dataset),
            "huggingface_hub": SimpleNamespace(HfApi=mock.Mock(return_value=hub)),
        }
        with mock.patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            exporter.main(["--config", str(config_path), "--output", str(self.output), "--revision", "release"])
        hub.dataset_info.assert_called_once_with("lightonai/embeddings-fine-tuning-filtered-en", revision="release")
        self.assertEqual(load_dataset.call_count, 6)
        self.assertEqual([call.kwargs["data_files"] for call in load_dataset.call_args_list],
                         [f"{kind}/{source}-*" for source in ["a_en", "b_en"]
                          for kind in ["scores", "queries", "documents"]])
        for call in load_dataset.call_args_list:
            self.assertEqual(call.kwargs["revision"], self.revision)
            self.assertEqual(call.kwargs["split"], "train")
        self.assertEqual(len(self.output.read_text().splitlines()), 18)


if __name__ == "__main__":
    unittest.main()
