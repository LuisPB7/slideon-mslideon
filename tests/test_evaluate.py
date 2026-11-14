"""Offline checkpoint resolution, cache isolation, and evaluation CLI checks.

Requires the project's ``eval`` extra. Benchmarks and pretrained model downloads
are replaced at their boundaries; the adapter and local model loading are real.
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from sentence_transformers import SparseEncoder

from test_finetune import TinyInput


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slideon_evaluate_test_target", ROOT / "scripts/eval/evaluate.py"
)
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


class EvaluateCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="slideon-evaluate-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        cache_environment = mock.patch.dict("os.environ", {"MTEB_CACHE": str(self.root / "mteb-cache")})
        cache_environment.start()
        self.addCleanup(cache_environment.stop)
        self.checkpoint = self.root / "final"
        self.checkpoint.mkdir()
        # Small stand-ins suffice for content hashing; the separate load test
        # creates a valid SparseEncoder checkpoint with real safetensors.
        self.files = {
            "modules.json": '[{"idx":0,"name":"0","path":"","type":"slideon.SlideOn"}]',
            "config.json": '{"model_type":"slideon","setting":1}',
            "tokenizer.json": '{"model":{"type":"WordLevel"},"version":"1"}',
            "slideon.py": "# model adapter version A\n",
            "model.safetensors": "weight-content-A",
        }
        for name, content in self.files.items():
            (self.checkpoint / name).write_text(content)

    def test_local_resolution_never_queries_hub(self):
        with mock.patch("huggingface_hub.HfApi.model_info", side_effect=AssertionError("Unexpected Hub lookup")):
            resolved = evaluate.resolve_checkpoint(str(self.checkpoint))
        self.assertTrue(resolved["local"])
        self.assertEqual(resolved["model"], str(self.checkpoint.resolve()))
        self.assertEqual(resolved["load_kwargs"], {"local_files_only": True})
        self.assertRegex(resolved["revision"], r"^sha256-[0-9a-f]{64}$")
        self.assertTrue(resolved["model_id"])
        self.assertEqual(resolved, evaluate.resolve_checkpoint(str(self.checkpoint / ".")))

    def test_hub_resolution_keeps_commit_pinning(self):
        sha = "b" * 40
        with mock.patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(sha=sha)) as model_info:
            resolved = evaluate.resolve_checkpoint("lightonai/SlideOn", revision="release")
        model_info.assert_called_once_with("lightonai/SlideOn", revision="release")
        self.assertEqual(resolved, {
            "model": "lightonai/SlideOn", "model_id": "lightonai/SlideOn", "revision": sha,
            "local": False, "load_kwargs": {"revision": sha},
        })

    def test_cache_identity_tracks_same_size_inference_content_changes(self):
        original = evaluate.resolve_checkpoint(str(self.checkpoint))["revision"]
        for name in ("model.safetensors", "config.json", "slideon.py", "tokenizer.json"):
            with self.subTest(file=name):
                path = self.checkpoint / name
                content = self.files[name]
                replacement = content.replace("A", "B") if "A" in content else content.replace("1", "2")
                self.assertNotEqual(content, replacement)
                self.assertEqual(len(content), len(replacement))
                path.write_text(replacement)
                self.assertNotEqual(original, evaluate.resolve_checkpoint(str(self.checkpoint))["revision"])
                path.write_text(content)
        self.assertEqual(original, evaluate.resolve_checkpoint(str(self.checkpoint))["revision"])

    def test_training_state_documentation_and_cache_do_not_change_identity(self):
        original = evaluate.resolve_checkpoint(str(self.checkpoint))["revision"]
        ignored = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                   "training_args.bin", "run_config.json", "README.md", ".cache/huggingface/download/metadata")
        for name in ignored:
            path = self.checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("training-only information")
        self.assertEqual(original, evaluate.resolve_checkpoint(str(self.checkpoint))["revision"])

    def test_invalid_local_inputs_fail_before_hub_lookup(self):
        raw_sae = self.root / "raw-sae"
        raw_sae.mkdir()
        (raw_sae / "config.json").write_text('{"base_model":"lightonai/DenseOn"}')
        (raw_sae / "sae.safetensors").write_text("raw weights")
        training_root = self.root / "training-run"
        (training_root / "final").mkdir(parents=True)
        (training_root / "trainer_state.json").write_text("{}")
        cases = [
            (str(self.root / "missing"), None),
            ("./definitely-missing-slideon-checkpoint", None),
            (str(self.checkpoint / "model.safetensors"), None),
            (str(raw_sae), None),
            (str(training_root), None),
            (str(self.checkpoint), "revision-is-not-valid-for-local"),
        ]
        with mock.patch("huggingface_hub.HfApi.model_info", side_effect=AssertionError("Unexpected Hub lookup")):
            for model, revision in cases:
                with self.subTest(model=model, revision=revision):
                    with self.assertRaises((ValueError, FileNotFoundError, NotADirectoryError)):
                        evaluate.resolve_checkpoint(model, revision=revision)

    def test_saved_sparse_encoder_loads_locally_offline(self):
        path = self.root / "valid-local"
        model = SparseEncoder(modules=[TinyInput()], device="cpu", local_files_only=True)
        model.save(str(path), create_model_card=False)
        with mock.patch("huggingface_hub.HfApi.model_info", side_effect=AssertionError("Unexpected Hub lookup")):
            resolved = evaluate.resolve_checkpoint(str(path))
            loaded = SparseEncoder(resolved["model"], device="cpu", trust_remote_code=True,
                                   **resolved["load_kwargs"])
        self.assertIsInstance(loaded[0], TinyInput)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)

    def _run_main(self, model, output, *, revision=None, query_budget=64):
        import mteb
        task = SimpleNamespace(metadata=SimpleNamespace(
            name="ToyRetrieval", type="Retrieval", eval_splits=["test"], main_score="ndcg_at_10"
        ), hf_subsets=["default"])
        result = SimpleNamespace(task_name="ToyRetrieval", scores={
            "test": [{"hf_subset": "default", "ndcg_at_10": 0.5}]
        })
        encoder = mock.Mock()
        encoder.device = torch.device("cpu")
        encoder.encode_query.return_value = torch.eye(2)
        encoder.encode_document.return_value = torch.eye(2)
        searches = []

        def run_task(search, tasks, **kwargs):
            searches.append(search)
            self.assertEqual(tasks, [task])
            # Exercise the actual adapter's public encoding-budget forwarding.
            search.encode_dataset(None, task.metadata, "query")
            search.encode_dataset(None, task.metadata, "document")
            return SimpleNamespace(task_results=[result])

        argv = ["--model", str(model), "--output", str(output), "--suite", "beir",
                "--device", "cpu", "--trust-remote-code", "--query-max-active-dims", str(query_budget),
                "--document-max-active-dims", "512"]
        if revision is not None:
            argv += ["--revision", revision]
        with mock.patch.object(evaluate, "select_suites", return_value=([task], {"beir": ["ToyRetrieval"]})), \
             mock.patch.object(mteb, "evaluate", side_effect=run_task) as run, \
             mock.patch("sentence_transformers.SparseEncoder", return_value=encoder) as load, \
             mock.patch("mteb._create_dataloaders.create_dataloader", return_value=[{"text": ["1", "2"]}]), \
             contextlib.redirect_stdout(io.StringIO()):
            evaluate.main(argv)
        run.assert_called_once()
        self.assertEqual(encoder.encode_query.call_args.kwargs["max_active_dims"], query_budget)
        self.assertEqual(encoder.encode_document.call_args.kwargs["max_active_dims"], 512)
        self.assertEqual(searches[0].query_max_active_dims, query_budget)
        self.assertEqual(searches[0].document_max_active_dims, 512)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["suites"]["beir"]["ndcg_at_10"], 0.5)
        return load.call_args, json.loads((output / "run_config.json").read_text()), searches[0]

    def test_main_local_budgets_and_output_cache_guards(self):
        output = self.root / "local-results"
        with mock.patch("huggingface_hub.HfApi.model_info", side_effect=AssertionError("Unexpected Hub lookup")):
            call, config, search = self._run_main(self.checkpoint, output)
            self.assertEqual(call.args, (str(self.checkpoint.resolve()),))
            self.assertEqual(call.kwargs, {"local_files_only": True, "trust_remote_code": True, "device": "cpu"})
            self.assertEqual(config["mteb_model_id"], search.mteb_model_meta.name)
            self.assertEqual(config["revision"], search.mteb_model_meta.revision)
            # Reopening identical output is allowed; changed budgets or weights are not.
            self._run_main(self.checkpoint, output)
            with self.assertRaisesRegex(ValueError, "different evaluation configuration"):
                self._run_main(self.checkpoint, output, query_budget=32)
            (self.checkpoint / "model.safetensors").write_text("weight-content-B")
            with self.assertRaisesRegex(ValueError, "different evaluation configuration"):
                self._run_main(self.checkpoint, output)
        occupied = self.root / "occupied-results"
        occupied.mkdir()
        (occupied / "unrelated.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "empty output directory"):
            self._run_main(self.checkpoint, occupied)
        with self.assertRaisesRegex(ValueError, "outside the local checkpoint"):
            self._run_main(self.checkpoint, self.checkpoint / "results")

    def test_main_hub_preserves_existing_configuration_schema(self):
        sha = "b" * 40
        with mock.patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(sha=sha)):
            call, config, search = self._run_main("lightonai/SlideOn", self.root / "hub-results", revision="release")
        self.assertEqual(call.args, ("lightonai/SlideOn",))
        self.assertEqual(call.kwargs, {"revision": sha, "trust_remote_code": True, "device": "cpu"})
        self.assertEqual(search.mteb_model_meta.name, "lightonai/SlideOn")
        self.assertEqual(config["revision"], sha)
        self.assertEqual(set(config), {"model", "revision", "mteb_version", "score", "suites",
                                     "query_max_active_dims", "document_max_active_dims", "batch_size", "tasks"})


if __name__ == "__main__":
    unittest.main()
