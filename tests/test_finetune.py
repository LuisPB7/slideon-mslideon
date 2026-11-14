"""Offline integration checks using the real Sentence Transformers training loop.

Run with ``python -m unittest discover -s tests -p test_finetune.py`` after
installing the project's training dependencies. No pretrained assets are used.
"""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

# Keep library initialization, model cards, and checkpoint reloads offline.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ACCELERATE_USE_CPU"] = "true"

import torch
from datasets import Dataset
from safetensors.torch import load_file, save_file
from sentence_transformers import SparseEncoder
from sentence_transformers.base.modules import InputModule
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast, TrainerCallback


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slideon_finetune_test_target", ROOT / "scripts/finetune/train_slideon.py"
)
finetune = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finetune)


class TinyInput(InputModule):
    """A four-dimensional trainable encoder with local checkpoint support."""

    config_file_name = "tiny_config.json"
    forward_kwargs = {"max_active_dims"}

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(4, 4, bias=False)
        self.model.config = SimpleNamespace(_commit_hash="offline-test")
        self.settings = {}
        self.max_seq_length = 8
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
            unk_token="[UNK]",
        )

    def get_embedding_dimension(self):
        return 4

    def preprocess(self, inputs, **kwargs):
        # Integer strings retain exact row/negative identities through collation.
        # Model-card statistics may also tokenize the nonnumeric source column.
        ids = torch.tensor([[int(text) if str(text).isdecimal() else 0] for text in inputs])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def forward(self, features, **kwargs):
        ids = features["input_ids"][:, 0]
        values = torch.nn.functional.one_hot(ids % 4, 4).float()
        embeddings = torch.sigmoid(self.model(values))
        features["sentence_embedding"] = embeddings
        features["load_balance"] = embeddings.square().mean()
        return features

    def save(self, output_path, **kwargs):
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / self.config_file_name).write_text("{}")
        save_file(self.state_dict(), str(path / "tiny.safetensors"))
        self.save_tokenizer(output_path)

    @classmethod
    def load(cls, model_name_or_path, subfolder="", **kwargs):
        module = cls()
        module.load_state_dict(load_file(str(Path(model_name_or_path) / subfolder / "tiny.safetensors")))
        return module


class StopAfter(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_training_stop = True
        return control


class FinetuneIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny CPU operations are faster and deterministic with one thread.
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="slideon-finetune-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = [
            {
                "source": "small" if i < 4 else "large",
                "query": str(i),
                "positive": str(100 + i),
                "negatives": [str(1000 + 10 * i + j) for j in range(10)],
                "teacher_scores": [10.0 + i] + [i + j / 10 for j in range(10)],
            }
            for i in range(12)
        ]
        self.data_path = self.root / "training.jsonl"
        self.data_path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def _run(self, name, *, sampling, resume=None, stop_after=None, steps=4):
        records, trainers, models_created = [], [], []
        actual_loss = finetune.SlideOnLoss
        actual_trainer = finetune.SparseEncoderTrainer
        actual_args = finetune.SparseEncoderTrainingArguments
        actual_load_dataset = finetune.load_dataset

        class RecordingLoss(actual_loss):
            def forward(self, sentence_features, labels=None):
                # Record actual loss inputs, not prefetched-but-unused batches.
                records.append((
                    [feature["input_ids"][:, 0].tolist() for feature in sentence_features],
                    labels.tolist() if labels is not None else None,
                ))
                return super().forward(sentence_features, labels)

        def make_model(*args, **kwargs):
            model = SparseEncoder(modules=[TinyInput()], device="cpu", local_files_only=True)
            models_created.append(model)
            return model

        def make_args(**kwargs):
            return actual_args(**kwargs, use_cpu=True, disable_tqdm=True, dataloader_pin_memory=False)

        def load_dataset(*args, **kwargs):
            return actual_load_dataset(*args, **kwargs, cache_dir=str(self.root / "dataset-cache"))

        def make_trainer(**kwargs):
            if stop_after is not None:
                kwargs["callbacks"].append(StopAfter(stop_after))
            trainer = actual_trainer(**kwargs)
            trainers.append(trainer)
            return trainer

        output = self.root / name
        argv = [
            "--train", str(self.data_path), "--output", str(output),
            "--max-steps", str(steps), "--batch-size", "2", "--global-batch-size", "4",
            "--token-k", "2", "--query-budgets", "2", "4",
            "--document-budgets", "2", "4", "--no-bf16", "--save-steps", "2",
            "--logging-steps", "10", "--learning-rate", "0.01", "--warmup-ratio", "0",
        ]
        if sampling:
            argv += ["--source-sampling-alpha", "0.5", "--sample-negatives", "7"]
        if resume is not None:
            argv += ["--resume", str(resume)]
        with mock.patch.object(finetune, "from_sae", side_effect=make_model), \
             mock.patch.object(finetune, "load_dataset", side_effect=load_dataset), \
             mock.patch.object(finetune, "SlideOnLoss", RecordingLoss), \
             mock.patch.object(finetune, "SparseEncoderTrainingArguments", side_effect=make_args), \
             mock.patch.object(finetune, "SparseEncoderTrainer", side_effect=make_trainer):
            finetune.main(argv)
        self.assertIsInstance(trainers[0], actual_trainer)
        self.assertIsInstance(trainers[0].train_dataset, Dataset)
        self.assertTrue((output / "final" / "tiny.safetensors").is_file())
        state = {key: value.detach().clone() for key, value in models_created[0].state_dict().items()}
        return records, state, output, trainers[0]

    def test_sampling_checkpoint_resume_matches_uninterrupted_training(self):
        # Six microbatches make one epoch. Resume partway through epoch two,
        # where Trainer skips old batches before continuing the sampler.
        full, full_weights, full_output, _ = self._run("full", sampling=True, steps=7)
        first, _, partial_output, _ = self._run("resumed", sampling=True, stop_after=4, steps=7)
        checkpoint = partial_output / "checkpoint-4"
        self.assertTrue((checkpoint / "trainer_state.json").is_file())
        rest, resumed_weights, _, resumed_trainer = self._run(
            "resumed", sampling=True, resume=checkpoint, steps=7
        )
        self.assertEqual(len(full), 14)  # Seven updates, two microbatches each.
        self.assertEqual(len(first), 8)
        self.assertEqual(full, first + rest)
        self.assertEqual(resumed_trainer.state.global_step, 7)
        for key in full_weights:
            torch.testing.assert_close(full_weights[key], resumed_weights[key], rtol=0, atol=0)

        config = json.loads((full_output / "run_config.json").read_text())
        sampling = config["resolved"]["sampling"]
        self.assertTrue(sampling["enabled"])
        self.assertEqual(sampling["source_sampling_alpha"], 0.5)
        self.assertEqual(sampling["sample_negatives"], 7)
        self.assertEqual(sampling["source_counts"], {"small": 4, "large": 8})
        for source, count in sampling["source_counts"].items():
            self.assertAlmostEqual(sampling["source_probabilities"][source],
                                   count ** 0.5 / (4 ** 0.5 + 8 ** 0.5))

        for groups, labels in full:
            self.assertEqual(len(groups), 9)  # Query, positive, seven negatives.
            self.assertEqual(len({self.rows[i]["source"] for i in groups[0]}), 1)
            for column, row_id in enumerate(groups[0]):
                selected = [group[column] for group in groups[2:]]
                self.assertEqual(len(set(selected)), 7)
                self.assertEqual(groups[1][column], 100 + row_id)
                self.assertEqual(labels[column][0], 10.0 + row_id)
                for offset, negative_id in enumerate(selected, 1):
                    index = negative_id - 1000 - 10 * row_id
                    self.assertIn(index, range(10))
                    self.assertAlmostEqual(labels[column][offset], row_id + index / 10, places=5)

    def test_disabled_flags_preserve_plain_dataset_and_all_negatives(self):
        args = finetune.parse_args(["--train", str(self.data_path), "--output", str(self.root / "unused")])
        self.assertIsNone(args.source_sampling_alpha)
        self.assertIsNone(args.sample_negatives)
        records, _, output, trainer = self._run("defaults", sampling=False, steps=2)
        self.assertIs(type(trainer.train_dataset), Dataset)
        self.assertIs(type(trainer.data_collator), finetune.RetrievalCollator)
        self.assertEqual(len(records), 4)
        for groups, labels in records:
            self.assertEqual(len(groups), 12)
            for column, row_id in enumerate(groups[0]):
                self.assertEqual([group[column] for group in groups[2:]],
                                 [int(text) for text in self.rows[row_id]["negatives"]])
                torch.testing.assert_close(torch.tensor(labels[column]),
                                           torch.tensor(self.rows[row_id]["teacher_scores"]))
        config = json.loads((output / "run_config.json").read_text())
        self.assertIsNone(config["parameters"]["source_sampling_alpha"])
        self.assertIsNone(config["parameters"]["sample_negatives"])
        self.assertIsNone(config["resolved"]["sampling"])


if __name__ == "__main__":
    unittest.main()
