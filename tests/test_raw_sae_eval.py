"""Offline raw-SAE evaluation, including real conversion and CPU inference.

The integration fixture replaces only the backbone download with a tiny local
ModernBERT checkpoint. The production loader, exporter, tokenizer, custom model
code and SparseEncoder inference all run normally.
"""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast
from transformers import dynamic_module_utils

from slideon import loading


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slideon_raw_sae_evaluate_test_target", ROOT / "scripts/eval/evaluate.py"
)
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


class RawSAEEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="slideon-raw-eval-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw sae" / "final"
        self.raw.mkdir(parents=True)
        self.config = {
            "name": "SlideOn", "base_model": "test/TinyDenseOn",
            "base_model_revision": "a" * 40,
            "d_model": 16, "n_latents": 8, "k": 2, "d_lift": 0,
            "normalized_inputs": True, "backbone_only": True, "truncate_after": 0,
            "query_length": 32, "document_length": 48,
            "query_prompt": "query: ", "document_prompt": "document: ",
        }
        generator = torch.Generator().manual_seed(123)
        self.weights = {
            "W_enc": torch.randn(16, 8, generator=generator),
            "b_enc": torch.linspace(0.1, 0.8, 8),
            "b_dec": torch.linspace(-0.1, 0.1, 16),
            "W_dec": torch.randn(8, 16, generator=generator),
            "steps_since_fired": torch.arange(8),
        }
        self.write_config()
        save_file(self.weights, str(self.raw / "sae.safetensors"))

    def write_config(self):
        (self.raw / "config.json").write_text(json.dumps(self.config))

    @staticmethod
    def contents(directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def test_resolve_raw_checkpoint_preserves_saved_metadata_without_network(self):
        before = self.contents(self.raw)
        with mock.patch("huggingface_hub.HfApi.model_info", side_effect=AssertionError("Unexpected Hub lookup")):
            checkpoint = evaluate.resolve_checkpoint(str(self.raw))
        self.assertTrue(checkpoint["local"])
        self.assertEqual(checkpoint["model"], str(self.raw.resolve()))
        self.assertRegex(checkpoint["revision"], r"^sha256-[0-9a-f]{64}$")
        raw = checkpoint["raw_sae"]
        self.assertEqual(raw["base_model"], self.config["base_model"])
        self.assertEqual(raw["backbone_revision"], "a" * 40)
        self.assertEqual(raw["query_length"], 32)
        self.assertEqual(raw["document_length"], 48)
        self.assertRegex(raw["loader_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertEqual(self.contents(self.raw), before)

    def test_raw_weights_and_config_changes_invalidate_results(self):
        original = evaluate.resolve_checkpoint(str(self.raw))["revision"]
        changed = dict(self.weights)
        changed["W_enc"] = self.weights["W_enc"] + 0.01
        save_file(changed, str(self.raw / "sae.safetensors"))
        self.assertNotEqual(evaluate.resolve_checkpoint(str(self.raw))["revision"], original)
        save_file(self.weights, str(self.raw / "sae.safetensors"))
        self.assertEqual(evaluate.resolve_checkpoint(str(self.raw))["revision"], original)
        self.config["query_length"] = 64
        self.write_config()
        self.assertNotEqual(evaluate.resolve_checkpoint(str(self.raw))["revision"], original)

    def test_raw_training_state_and_documentation_do_not_invalidate_results(self):
        original = evaluate.resolve_checkpoint(str(self.raw))["revision"]
        for name in ("training_state.pt", "optimizer.pt", "trainer_state.json", "README.md",
                     "run_config.json", ".cache/huggingface/download/metadata"):
            path = self.raw / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not used for raw SAE inference")
        self.assertEqual(evaluate.resolve_checkpoint(str(self.raw))["revision"], original)

    def test_resolved_backbone_commit_participates_in_cache_identity(self):
        self.config["base_model_revision"] = "release"
        self.write_config()
        with mock.patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(sha="b" * 40)) as lookup:
            first = evaluate.resolve_checkpoint(str(self.raw))
        lookup.assert_called_once_with("test/TinyDenseOn", revision="release")
        with mock.patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(sha="c" * 40)):
            second = evaluate.resolve_checkpoint(str(self.raw))
        self.assertEqual(first["raw_sae"]["backbone_revision"], "b" * 40)
        self.assertEqual(second["raw_sae"]["backbone_revision"], "c" * 40)
        self.assertNotEqual(first["revision"], second["revision"])

    def test_conversion_code_fingerprint_participates_in_cache_identity(self):
        with mock.patch.object(evaluate, "raw_sae_loader_fingerprint", return_value="b" * 64):
            first = evaluate.resolve_checkpoint(str(self.raw))
        with mock.patch.object(evaluate, "raw_sae_loader_fingerprint", return_value="c" * 64):
            second = evaluate.resolve_checkpoint(str(self.raw))
        self.assertEqual(first["raw_sae"]["loader_fingerprint"], "b" * 64)
        self.assertEqual(second["raw_sae"]["loader_fingerprint"], "c" * 64)
        self.assertNotEqual(first["revision"], second["revision"])

    def test_missing_or_invalid_raw_metadata_fails_before_network_lookup(self):
        original = dict(self.config)
        cases = [(key, None) for key in ("base_model", "n_latents", "k", "query_length", "document_length")]
        cases += [("query_length", 0), ("document_length", -1), ("query_length", True)]
        with mock.patch("huggingface_hub.HfApi.model_info", side_effect=AssertionError("Unexpected Hub lookup")):
            for key, value in cases:
                with self.subTest(key=key, value=value):
                    self.config = dict(original)
                    if value is None:
                        self.config.pop(key)
                    else:
                        self.config[key] = value
                    self.write_config()
                    with self.assertRaisesRegex(ValueError, "[Rr]aw|[Ss][Aa][Ee]|length|base_model"):
                        evaluate.resolve_checkpoint(str(self.raw))

    def test_raw_loading_requires_explicit_custom_code_trust(self):
        checkpoint = evaluate.resolve_checkpoint(str(self.raw))
        with mock.patch.object(loading, "from_sae") as load:
            with self.assertRaisesRegex(ValueError, "trust"):
                evaluate.load_evaluation_model(checkpoint, trust_remote_code=False, device="cpu")
        load.assert_not_called()

    def make_backbone(self):
        directory = self.root / "tiny backbone"
        directory.mkdir()
        config = ModernBertConfig(
            vocab_size=10, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            max_position_embeddings=128, local_attention=32,
            pad_token_id=0, bos_token_id=1, cls_token_id=1,
            eos_token_id=2, sep_token_id=2, layer_types=["full_attention"],
        )
        # Seed only this fixture; do not alter another test's random generator.
        with torch.random.fork_rng():
            torch.manual_seed(123)
            backbone = ModernBertModel(config)
        backbone.save_pretrained(directory)
        tokenizer = Tokenizer(models.WordLevel({
            "[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "[MASK]": 4,
            "word": 5, "query": 6, "document": 7, ":": 8, "other": 9,
        }, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[CLS]",
            eos_token="[SEP]", unk_token="[UNK]", mask_token="[MASK]",
            model_max_length=128,
        ).save_pretrained(directory)
        (directory / "modules.json").write_text(json.dumps([
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"}
        ]))
        (directory / "config_sentence_transformers.json").write_text("{}")
        return directory, backbone

    def test_actual_raw_loader_preserves_weights_lengths_device_and_source(self):
        base, original_backbone = self.make_backbone()
        before_raw, before_base = self.contents(self.raw), self.contents(base)
        checkpoint = evaluate.resolve_checkpoint(str(self.raw))
        custom_cache = str(self.root / "custom-code-cache")
        with mock.patch.object(loading, "snapshot_download", return_value=str(base)) as download, \
             mock.patch.object(dynamic_module_utils, "HF_MODULES_CACHE", custom_cache), \
             mock.patch("huggingface_hub.HfApi.model_info", side_effect=AssertionError("Unexpected Hub lookup")):
            model = evaluate.load_evaluation_model(checkpoint, trust_remote_code=True, device="cpu")
            download.assert_called_once_with("test/TinyDenseOn", revision="a" * 40)
            self.assertEqual(model.device.type, "cpu")
            module = model[0]
            for name, expected in self.weights.items():
                torch.testing.assert_close(getattr(module.model.head, name), expected, rtol=0, atol=0)
            self.assertFalse(module.model.head.W_dec.requires_grad)
            torch.testing.assert_close(module.model.head.gate_relu_phi, torch.ones(8), rtol=0, atol=0)
            for name, expected in original_backbone.state_dict().items():
                torch.testing.assert_close(module.model.backbone.state_dict()[name], expected, rtol=0, atol=0)
            long_text = " ".join(["word"] * 100)
            self.assertEqual(module.preprocess([long_text], task="query")["input_ids"].shape, (1, 32))
            self.assertEqual(module.preprocess([long_text], task="document")["input_ids"].shape, (1, 48))
            effective = evaluate.apply_length_overrides(model, query_length=64)
            self.assertEqual(effective, {"query_length": 64, "document_length": 48})
            self.assertEqual(module.preprocess([long_text], task="query")["input_ids"].shape, (1, 64))
            self.assertEqual(module.preprocess([long_text], task="document")["input_ids"].shape, (1, 48))
            model.eval()
            for role in ("query", "document"):
                vectors = getattr(model, "encode_" + role)(
                    ["word other", "other word word"], convert_to_tensor=True,
                    show_progress_bar=False, max_active_dims=3,
                )
                if vectors.layout != torch.strided:
                    vectors = vectors.to_dense()
                self.assertEqual(vectors.shape, (2, 8))
                self.assertEqual(vectors.device.type, "cpu")
                self.assertTrue(torch.isfinite(vectors).all())
                self.assertTrue(((vectors != 0).sum(1) <= 3).all())
        self.assertEqual(self.contents(self.raw), before_raw)
        self.assertEqual(self.contents(base), before_base)


if __name__ == "__main__":
    unittest.main()
