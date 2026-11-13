"""Fine-tune SlideOn: data collation, retrieval losses, and Trainer setup."""

# Reading order: main() connects configuration, SAE loading, input data, and the
# Sentence Transformers Trainer. RetrievalCollator turns text rows into model
# inputs; SlideOnLoss defines what the resulting sparse vectors should learn.
# The Trainer owns the optimizer loop, gradient accumulation, and checkpoints.
import argparse
import itertools
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import SparseEncoderTrainer, SparseEncoderTrainingArguments
from torch import nn
from torch.nn import functional as F
from transformers import TrainerCallback, set_seed

from slideon.loading import from_sae


def prune(values, budget):
    # values is a dense tensor whose final axis is the latent vocabulary. Pruning
    # keeps that shape: it changes which weights are nonzero, not the vocabulary.
    # Zero is this training helper's explicit "keep everything" sentinel.
    if budget == 0:
        return values
    if not 0 < budget <= values.shape[-1]:
        raise ValueError("Invalid pruning budget")
    selected, indices = values.topk(budget, dim=-1)
    # scatter restores the selected weights to their original latent positions.
    # Gradients flow through selected weights; discarded entries contribute zero.
    return torch.zeros_like(values).scatter(-1, indices, selected)


class SlideOnLoss(nn.Module):
    # This is the retrieval-stage objective. SAE reconstruction and its decoder
    # are not used here: we optimize query/document scores and sparsity instead.
    # Return named loss components so SparseEncoderTrainer can combine/log them.
    def __init__(self, model, temperature=0.02, kd_weight=1.0, teacher_temperature=1.0,
                 balance_weight=0.01, query_flops=0.01, document_flops=0.01,
                 query_budgets=(128, 64, 32), document_budgets=(1024, 512),
                 flops_ramp_steps=3000):
        # model is the complete SparseEncoder, including its trainable backbone
        # and latent head. Query/document budgets are alternatives to train
        # jointly; the forward pass evaluates their full Cartesian product.
        super().__init__()
        if temperature <= 0 or teacher_temperature <= 0:
            raise ValueError("Temperatures must be positive")
        self.model = model
        self.temperature, self.teacher_temperature = temperature, teacher_temperature
        self.kd_weight, self.balance_weight = kd_weight, balance_weight
        self.query_flops, self.document_flops = query_flops, document_flops
        self.query_budgets, self.document_budgets = query_budgets, document_budgets
        self.flops_ramp_steps = flops_ramp_steps
        # RampCallback refreshes this from Trainer's optimizer-update counter.
        # It is not incremented for each individual forward/microbatch call.
        self.step = 0

    def forward(self, sentence_features, labels=None):
        # The collator/Trainer contract orders groups as queries, positives,
        # negative_0, negative_1, ...; each group contains the same B examples.
        # main() clears final-vector caps, so these encodings are unpruned after
        # pooling. Per-token TopK inside the model still applies. Copy the feature
        # dictionaries because the ST adapter adds its outputs to its input dict.
        outputs = [self.model(dict(feature), max_active_dims=None) for feature in sentence_features]
        q = outputs[0]["sentence_embedding"].float()
        docs = torch.cat([item["sentence_embedding"].float() for item in outputs[1:]])

        # q has shape [B, V]; docs has shape [B * (1 + N), V], where V is
        # vocabulary size and N is negatives per query. Positives occupy the
        # first B rows, hence query i's contrastive target is column i.
        batch = q.shape[0]
        target = torch.arange(batch, device=q.device)

        # KD compares each query only with its own positive/negative group.
        # Example B=2, N=2: docs=[p0,p1,n00,n10,n01,n11], so own_docs is
        # [[0,2,4],[1,3,5]]. Teacher labels must have shape [B, 1 + N] and use
        # exactly this positive-then-negatives order.
        own_docs = target[:, None] + torch.arange(len(outputs)-1, device=q.device)[None, :] * batch
        if self.kd_weight and (labels is None or labels.shape != own_docs.shape):
            raise ValueError("KD requires teacher_scores for positive then each negative")
        contrastive, kd = [], []

        # Reuse the expensive encodings for every budget pair. With three query
        # budgets and two document budgets, six retrieval losses train the same
        # parameters to work at different inference sparsity settings.
        for qb, db in itertools.product(self.query_budgets, self.document_budgets):
            scores = prune(q, qb) @ prune(docs, db).T / self.temperature
            # Cross-entropy sees every document in this local microbatch, so
            # other examples' positives/negatives also act as in-batch negatives.
            # There is no embedding all-gather here: gradient accumulation and
            # distributed gradient synchronization do not enlarge this matrix.
            contrastive.append(F.cross_entropy(scores, target))
            if self.kd_weight:
                # Student scores already contain the student temperature above.
                # kl_div takes student log-probabilities and teacher probabilities;
                # batchmean sums candidate contributions then divides by B.
                kd.append(F.kl_div(F.log_softmax(scores.gather(1, own_docs), -1),
                          F.softmax(labels.float() / self.teacher_temperature, -1),
                          reduction="batchmean"))

        # Introduce FLOPS pressure gradually so early retrieval learning is less
        # constrained. A zero-length ramp enables the full penalty immediately.
        ramp = min(1.0, self.step / self.flops_ramp_steps) ** 2 if self.flops_ramp_steps else 1.0

        # Average across budget pairs rather than scaling loss with their count.
        # Load balancing is supplied by each model call. FLOPS regularization
        # uses UNPRUNED pooled weights: square the mean activation of each latent,
        # then sum over latents. This is a differentiable sparsity/work surrogate,
        # not a count of measured hardware operations. The document penalty averages
        # positives and all negative groups; the query penalty is computed separately.
        result = {
            "contrastive": torch.stack(contrastive).mean(),
            "load_balance": self.balance_weight * torch.stack([o["load_balance"] for o in outputs]).mean(),
            "query_flops": ramp * self.query_flops * q.mean(0).square().sum(),
            "document_flops": ramp * self.document_flops * docs.mean(0).square().sum(),
        }
        if kd:
            result["kd"] = self.kd_weight * torch.stack(kd).mean()
        return result


class RetrievalCollator:
    # Dataset rows contain raw strings and a list of negatives. The Trainer calls
    # this object on a batch of rows before passing grouped features to the loss.
    # This attribute identifies the dataset's supervision column to ST tooling.
    valid_label_columns = ["teacher_scores"]

    def __init__(self, module):
        # module is model[0], the SlideOn InputModule. Its preprocess() method
        # applies the correct task prompt, tokenization, and truncation length.
        self.module = module

    def __call__(self, rows):
        # Rectangular groups are required because negatives are organized by
        # position across the batch. This script consumes prepared negatives;
        # it does not mine or resample them here.
        count = len(rows[0]["negatives"])
        if any(len(row["negatives"]) != count for row in rows):
            raise ValueError("Use a fixed number of negatives per example")
        result = {}
        columns = [[r["query"] for r in rows], [r["positive"] for r in rows]]
        columns += [[r["negatives"][i] for r in rows] for i in range(count)]

        # Group names such as text0_input_ids/text1_input_ids let the Trainer
        # reconstruct sentence_features in the order expected by SlideOnLoss.
        # Only group zero uses query preprocessing; all others are documents.
        for i, texts in enumerate(columns):
            features = self.module.preprocess(texts, task="query" if i == 0 else "document")
            result.update({f"text{i}_{key}": value for key, value in features.items()})

        # Teacher scores are already computed in the input data. Their order is
        # [positive, negative_0, ...]; the loss checks the resulting tensor shape.
        if "teacher_scores" in rows[0]:
            result["label"] = torch.tensor([r["teacher_scores"] for r in rows])
        return result


class RampCallback(TrainerCallback):
    # Adapt Trainer's progress state to the loss without writing a custom loop.
    def __init__(self, loss):
        # Keep the same loss instance supplied to SparseEncoderTrainer.
        self.loss = loss

    def on_step_begin(self, args, state, control, **kwargs):
        # global_step counts completed optimizer updates, including on resume.
        # All accumulation microbatches for an update therefore use the same ramp.
        self.loss.step = state.global_step


def read_config(stage, path=None):
    # Load either the default English retrieval JSON or a caller-selected file.
    # The stage/model envelope prevents accidentally using an SAE-stage config;
    # optional "resolved" metadata lets the same format describe recorded runs.
    default_model = "SlideOn"
    source = Path(path) if path else Path(__file__).resolve().parents[1] / "training-configs" / default_model / "config.json"
    config = json.loads(source.read_text())
    if not isinstance(config, dict) or set(config) - {"model", "stage", "parameters", "resolved"}:
        raise ValueError("Expected a training config with model, stage, and parameters")
    if config.get("stage") != stage or config.get("model") != default_model:
        raise ValueError(f"Expected the {default_model} {stage} configuration")
    if not isinstance(config.get("parameters"), dict):
        raise ValueError("Configuration parameters must be an object")
    return config


def parse_config_args(parser, stage, argv=None, extra_keys=()):
    # First inspect only --config, because JSON values must become defaults
    # before the full parser reads CLI overrides. Unknown options are retained
    # for that second pass rather than rejected by this lightweight probe.
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    selected, _ = probe.parse_known_args(argv)
    parser.add_argument("--config", help="Training JSON; defaults to scripts/training-configs/SlideOn/config.json")
    try:
        config = read_config(stage, selected.config)
        parameters = config["parameters"]
        actions = {action.dest: action for action in parser._actions if action.dest not in ("help", "config")}
        # A misspelled JSON key should fail instead of silently creating an
        # unused Namespace attribute. extra_keys allows explicit caller extras.
        unknown = set(parameters) - set(actions) - set(extra_keys)
        if unknown:
            raise ValueError(f"Unknown configuration parameters: {', '.join(sorted(unknown))}")
        for key, value in parameters.items():
            # argparse normally converts CLI strings, but JSON defaults arrive
            # already typed. Validate those values explicitly, including lists,
            # finite numbers, and bool-versus-int distinctions.
            action = actions.get(key)
            if action is None or value is None:
                continue
            if isinstance(action, argparse.BooleanOptionalAction):
                if type(value) is not bool:
                    raise ValueError(f"{key} must be a boolean")
            elif action.type:
                items = value if action.nargs in ("+", "*") else [value]
                if not isinstance(items, list):
                    raise ValueError(f"{key} must be a list")
                for item in items:
                    if action.type in (int, float) and (type(item) not in (int, float) or not math.isfinite(item)):
                        raise ValueError(f"{key} must be a finite number")
                    if action.type is int and type(item) is not int:
                        raise ValueError(f"{key} must be an integer")
                    if action.type is str and not isinstance(item, str):
                        raise ValueError(f"{key} must be a string")
        parser.set_defaults(**parameters)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    args = parser.parse_args(argv)
    # Explicit CLI arguments take precedence over the installed JSON defaults.
    # Keep the envelope's model name for the output provenance record.
    args.model = config["model"]
    return args


def run_config(model, stage, args, **resolved):
    # Separate user/effective parameters from facts discovered at runtime, such
    # as world size, dataset fingerprint, and the loaded checkpoint commit.
    parameters = {key: value for key, value in vars(args).items() if key not in ("config", "model")}
    return dict(model=model, stage=stage, parameters=parameters, resolved=resolved)


def write_run_config(output, config):
    # Write beside the destination and rename only after serialization succeeds,
    # avoiding a half-written JSON record. The caller creates output first.
    target = Path(output) / "run_config.json"
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


def parse_args(argv=None):
    # Declare CLI types here; defaults live in training-configs/SlideOn/config.json.
    # Grouping by type keeps the CLI schema and JSON validation synchronized.
    parser = argparse.ArgumentParser(description="Fine-tune a token SAE using SparseEncoderTrainer")
    for key in ("sae", "sae_revision", "backbone_revision", "train", "output", "resume",
                "query_prompt", "document_prompt", "lr_scheduler_type", "optim"):
        parser.add_argument("--" + key.replace("_", "-"), type=str)
    for key in ("learning_rate", "epochs", "temperature", "teacher_temperature", "kd_weight",
                "balance_weight", "query_flops", "document_flops", "weight_decay", "warmup_ratio",
                "adam_beta1", "adam_beta2", "adam_epsilon", "max_grad_norm"):
        parser.add_argument("--" + key.replace("_", "-"), type=float)
    for key in ("max_steps", "global_batch_size", "batch_size", "gradient_accumulation",
                "query_length", "document_length", "token_k", "head_chunk_size", "flops_ramp_steps",
                "seed", "save_steps", "logging_steps", "save_total_limit"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    for key in ("query_budgets", "document_budgets"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, nargs="+")
    for key in ("bf16", "gradient_checkpointing"):
        parser.add_argument("--" + key.replace("_", "-"), action=argparse.BooleanOptionalAction)
    args = parse_config_args(parser, "retrieval", argv)

    # Validate the merged JSON + CLI result before loading models or datasets.
    # Null train/output defaults require the caller to provide concrete paths.
    try:
        for key, value in vars(args).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not args.sae or args.query_prompt is None or args.document_prompt is None:
            raise ValueError("sae, query_prompt, and document_prompt must be configured")
        if not args.train or not args.output:
            raise ValueError("--train and --output are required")
        positive = ("batch_size", "query_length", "document_length", "token_k", "head_chunk_size",
                    "save_steps", "logging_steps", "save_total_limit", "learning_rate", "epochs",
                    "temperature", "teacher_temperature", "adam_epsilon")
        if any(getattr(args, key) is None or getattr(args, key) <= 0 for key in positive):
            raise ValueError("Batch sizes, lengths, step intervals, rates, and temperatures must be positive")
        if args.max_steps is None or args.max_steps == 0 or args.max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive integer")
        for key in ("kd_weight", "balance_weight", "query_flops", "document_flops", "weight_decay",
                    "flops_ramp_steps", "seed", "max_grad_norm"):
            if getattr(args, key) is None or getattr(args, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if not (0 <= args.warmup_ratio < 1 and 0 <= args.adam_beta1 < 1 and 0 <= args.adam_beta2 < 1):
            raise ValueError("Invalid warmup ratio or Adam betas")
        for budgets in (args.query_budgets, args.document_budgets):
            if not budgets or any(budget < 0 for budget in budgets):
                raise ValueError("Pruning budgets must be nonempty lists of nonnegative integers")

        # A global optimizer batch combines all devices and accumulation steps:
        # global_batch_size = batch_size * WORLD_SIZE * gradient_accumulation.
        # Example: 128 examples with 4/device on one GPU needs 32 microbatches.
        # This governs gradient averaging; each loss call still sees only its
        # own local query/document score matrix, as explained in forward().
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world < 1:
            raise ValueError("WORLD_SIZE must be positive")
        per_update = args.batch_size * world
        if args.global_batch_size is None:
            args.global_batch_size = per_update * (args.gradient_accumulation or 1)
        if args.global_batch_size < 1 or args.global_batch_size % per_update:
            raise ValueError("global_batch_size must be divisible by batch_size * WORLD_SIZE")
        accumulation = args.global_batch_size // per_update
        # If both ways of specifying batch size were supplied, demand agreement
        # rather than silently changing the requested effective optimizer batch.
        if args.gradient_accumulation is not None and args.gradient_accumulation != accumulation:
            raise ValueError("gradient_accumulation does not match the requested global_batch_size")
        args.gradient_accumulation = accumulation
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    # 1. Resolve the run and initialize the complete model from a trained SAE.
    # torchrun sets LOCAL_RANK/WORLD_SIZE; reject accidental single-process use
    # of several visible GPUs so distribution is chosen explicitly.
    args = parse_args(argv)
    if torch.cuda.device_count() > 1 and "LOCAL_RANK" not in os.environ:
        raise ValueError("Use torchrun for multiple GPUs, or set CUDA_VISIBLE_DEVICES to one GPU")
    set_seed(args.seed)
    model = from_sae(args.sae, revision=args.sae_revision, backbone_revision=args.backbone_revision,
                     query_length=args.query_length, document_length=args.document_length)
    # Backbone checkpointing is enabled directly because it is nested inside
    # the ST adapter. Head-chunk checkpointing is handled inside SlideOnModel.
    if args.gradient_checkpointing:
        model[0].model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    settings = model[0].settings
    # The loss performs each final-vector budget itself, after one encoding.
    # Clear model-level caps here to avoid pruning information before that loop.
    settings.update(token_k=args.token_k, head_chunk_size=args.head_chunk_size,
                    query_max_active_dims=None, document_max_active_dims=None,
                    prompts={"query": args.query_prompt, "document": args.document_prompt})
    model.prompts = dict(settings["prompts"])
    # TokenK and pooled-vector budgets share the latent vocabulary but operate
    # at different points: per token inside the model versus after pooling.
    dimensions = model.get_embedding_dimension()
    if args.token_k > dimensions or any(budget > dimensions for budget in args.query_budgets + args.document_budgets):
        raise ValueError("Token sparsity and pruning budgets must fit the model's latent dimension")

    # 2. Load prepared JSON/JSONL examples. Query, positive, and negative strings
    # are mandatory; teacher scores are mandatory only when KD is enabled.
    dataset = load_dataset("json", data_files=args.train, split="train")
    required = {"query", "positive", "negatives"} | ({"teacher_scores"} if args.kd_weight else set())
    if not required.issubset(dataset.column_names):
        raise ValueError(f"Dataset must contain {sorted(required)}")
    if len(dataset) < args.batch_size * int(os.environ.get("WORLD_SIZE", "1")):
        raise ValueError("Dataset is smaller than one per-step batch; lower --batch-size")

    # 3. Attach the objective and hand optimizer/scheduler/precision concerns to
    # SparseEncoderTrainer. A positive max_steps takes precedence over epochs
    # in Trainer's stopping rule. No backbone freeze is applied in this script.
    loss = SlideOnLoss(model, **{key: getattr(args, key) for key in (
        "temperature", "teacher_temperature", "kd_weight", "balance_weight", "query_flops",
        "document_flops", "query_budgets", "document_budgets", "flops_ramp_steps")})
    # remove_unused_columns=False preserves raw fields needed by our collator;
    # drop_last=True keeps local batches full. Trainer-level activation checkpointing
    # is disabled here because optional backbone checkpointing was configured above.
    training_args = SparseEncoderTrainingArguments(
        output_dir=args.output, learning_rate=args.learning_rate, num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, bf16=args.bf16,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type, adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2, adam_epsilon=args.adam_epsilon, optim=args.optim,
        max_grad_norm=args.max_grad_norm, gradient_checkpointing=False,
        save_steps=args.save_steps, logging_steps=args.logging_steps, save_total_limit=args.save_total_limit,
        report_to="none", remove_unused_columns=False, dataloader_drop_last=True, seed=args.seed,
        ddp_find_unused_parameters=False)
    trainer = SparseEncoderTrainer(model=model, args=training_args, train_dataset=dataset,
        loss=loss, data_collator=RetrievalCollator(model[0]), callbacks=[RampCallback(loss)])

    # Only the main distributed process writes this shared provenance record.
    if trainer.is_world_process_zero():
        Path(args.output).mkdir(parents=True, exist_ok=True)
        write_run_config(args.output, run_config(args.model, "retrieval", args,
            world_size=training_args.world_size, dataset_fingerprint=dataset._fingerprint,
            sae_commit=getattr(model[0].model.config, "_commit_hash", None)))

    # 4. Trainer executes forward/backward/update cycles and periodic checkpoint
    # saves. A successful run also writes final/; distributed cleanup runs even
    # if training raises an exception. Resume is delegated to Trainer's state.
    try:
        trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_model(str(Path(args.output) / "final"))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
