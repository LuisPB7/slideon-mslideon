# SlideOn: Training and Evaluation

<p align="center">
  <img src="assets/slideon-mslideon.png" alt="SlideOn and mSlideOn: a golden retriever sliding through a library with a bag of glowing stars" width="512">
</p>

This fork of [LightOn's SlideOn and mSlideOn repository](https://github.com/lightonai/slideon-mslideon)
focuses on faithfully reproducing **[SlideOn](https://huggingface.co/lightonai/SlideOn)**,
the English sparse retriever by Paulo Roberto Moura Junior and Raphaël Sourty.
The multilingual training scripts and configurations are excluded from this fork;
the shared model and evaluation code are retained.
This repository covers sparse autoencoder (SAE) pretraining, supervised retrieval
fine-tuning, and evaluation.

SlideOn starts from [DenseOn](https://huggingface.co/lightonai/DenseOn). SAE
pretraining followed by retrieval fine-tuning converts its token representations
into learned sparse vectors for inverted-index search. Matryoshka training allows
query and document sparsity to be adjusted at inference to control search and
storage costs.

For the method, training data, experiments, and results, see the authors'
[SlideOn and mSlideOn blog post](https://huggingface.co/blog/lightonai/slideon-mslideon).
The final retrieval models and pretrained SAE initializations are available in their
[Hugging Face collection](https://huggingface.co/collections/lightonai/slideon-and-mslideon).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Training

Each stage has its own editable training script under `scripts/`. Data
preparation, losses, and training are implemented directly in those files. `src/slideon/` contains the
model and checkpoint-loading code.

| Model | Training script | Configuration |
| --- | --- | --- |
| [SlideOn-SAE](https://huggingface.co/lightonai/SlideOn-SAE) | [SAE pretraining](scripts/sae-pretrain/train_slideon_sae.py) | [Config](scripts/training-configs/SlideOn-SAE/config.json) |
| [SlideOn](https://huggingface.co/lightonai/SlideOn) | [Retrieval fine-tuning](scripts/finetune/train_slideon.py) | [Config](scripts/training-configs/SlideOn/config.json) |

Each script loads its corresponding config automatically. Edit the JSON or pass
`--config my-config.json`; command-line arguments override its values. Every run
saves the effective settings in `run_config.json`. Use `--help` for available flags.

### SAE pretraining

Prepare frozen token embeddings once, then train the SAE:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sae-pretrain/train_slideon_sae.py \
  --prepare --cache token-cache/slideon

torchrun --standalone --nproc_per_node=8 scripts/sae-pretrain/train_slideon_sae.py \
  --cache token-cache/slideon --output outputs/slideon-sae
```

The SAE config selects DenseOn and AuxK, with 32,768 latents, token top-k 32,
a global token batch of 4,096, and AdamW at 0.0003. Change these and the data selection in the config. Token
preparation records dataset revisions; `--dataset-revisions pins.json` accepts
a mapping of dataset IDs to exact commits. The FP16 cache uses two bytes per
token dimension, so check storage before preparing a full training corpus.

Training writes metrics to `metrics.jsonl` and the completed SAE to `final/`.
Resume with `--resume path/to/training_state.pt`, keeping the same config, token
cache, world size, and buffer size. `--stop-after N` allows an early stop without
changing the learning-rate schedule.

### Retrieval fine-tuning

Provide JSONL with a fixed number of negatives. Teacher scores are ordered
positive first, then negatives:

```json
{"query":"What is sparse retrieval?","positive":"Sparse retrieval uses inverted indexes.","negatives":["Bread is baked in an oven."],"teacher_scores":[3.0,0.0]}
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune/train_slideon.py \
  --train english.jsonl --output outputs/slideon
```

The script starts from the pinned SAE release. Pass `--sae outputs/slideon-sae/final`
to use your own pretrained SAE. Use `torchrun` for multiple GPUs and `--resume PATH`
for a Trainer checkpoint. The completed model is saved under `final/`.

To prepare the authors' English data and enable source/negative sampling, run:

```bash
bash run_finetune.sh
```

The launcher creates `english.jsonl` if it is missing, then fine-tunes from the
released SAE using `--source-sampling-alpha 0.5 --sample-negatives 7`. It retains
the existing training configuration and writes to `outputs/slideon-alpha05-neg7`.
Set `PYTHON`, `TRAIN_JSONL`, `SFT_OUTPUT`, or `CUDA_VISIBLE_DEVICES` in the shell to
override those launcher settings. Additional arguments go to the training script,
for example `bash run_finetune.sh --resume outputs/slideon-alpha05-neg7/checkpoint-500`.

The converter can also be run separately:

```bash
python scripts/finetune/preprocess.py \
  --dataset lightonai/embeddings-fine-tuning-filtered-en \
  --revision 3464b0de66da5834f165a7eded49f1cc63d0b5eb --output english.jsonl
```

Each row retains its `source`, query, positive, all ten negatives and eleven raw
cross-encoder teacher scores. A sidecar manifest records the dataset revision,
row counts and checksum. Existing output files are never overwritten. Regenerate
older JSONL exports lacking `source` labels before enabling source sampling.

Both training flags are optional and independent:

| Flag | When enabled | When omitted |
| --- | --- | --- |
| `--source-sampling-alpha 0.5` | Draw each batch from one source, with probability proportional to that source's row count raised to alpha | Original Trainer sampling |
| `--sample-negatives 7` | Select seven stored negatives per example exposure, keeping teacher scores aligned | Use all negatives supplied by the input file |

Thus the original invocation above preserves the original training behavior;
with the new JSONL it consumes ten negatives. Alpha zero means uniform sampling
across sources, not disabled sampling. Each source batch contains distinct rows;
rows can recur in later batches. Every source must contain at least `batch_size`
rows. One sampled epoch draws approximately the input row count, rounded down to
complete batches on all processes, rather than guaranteeing a visit to every row.
Sampling probabilities and epoch size are recorded in `run_config.json`.

Keep sampling settings, seed, data, batch size and distributed world size fixed
when resuming a run. Source draws and negative subsets are deterministic from
the epoch and draw position, so worker prefetch does not change replay.

Run the CPU-only sampling and Trainer checkpoint tests with:

```bash
python -m unittest discover -s tests -v
```

The script exposes contrastive and distillation losses, load balancing, FLOPS
penalties, and Matryoshka query/document budgets. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune/train_slideon.py \
  --train english.jsonl --output outputs/slideon-experiment \
  --learning-rate 0.00002 --batch-size 8 --global-batch-size 128 \
  --query-budgets 32 64 --document-budgets 512 1024
```

`--batch-size` is per GPU; gradient accumulation is derived from the global batch.
Negatives and regularization statistics are computed per GPU, so accumulation
does not enlarge the negative pool. A positive `max_steps` takes precedence over
`epochs` (use `--max-steps -1` for epoch-based retrieval training).

The configs combine recorded checkpoint settings with explicit defaults for these
standalone scripts. Optional data sampling reconstructs the reported source and
negative sampling policies; the authors' exact historical loader is unavailable.
In particular, accumulation still does not enlarge the contrastive negative pool.
Some historical dataset revisions are unavailable, and full reproduction of the
released metrics has not been established.

## Evaluation

```bash
pip install -e '.[eval]'
python scripts/eval/evaluate.py --model lightonai/SlideOn --trust-remote-code \
  --suite beir --query-max-active-dims 32 --document-max-active-dims 512 --output results/slideon

python scripts/eval/evaluate.py --model lightonai/SlideOn --trust-remote-code \
  --suite mteb-eng-v2 --output results/slideon-mteb-eng-v2
```

The English workflows cover BEIR-15 and the retrieval tasks from
`MTEB(eng, v2)` (10 tasks). The MTEB v2 suite preserves the benchmark's
evaluation splits and language subsets and excludes the separate `InstructionRetrieval`
task type. Use `--tasks SciFact` for a smaller run.

Evaluation uses exact dot-product search. Vectors are unpruned by default;
`--query-max-active-dims` and `--document-max-active-dims` control final pruning. Outputs include MTEB
results, `summary.json` with scores on a 0–1 scale, and `efficiency.json` with
average nonzeros and retrieval FLOPS (expected active-latent overlap).
BEIR reports average nDCG@10. The English MTEB v2 suite averages each task's
nDCG@10 across its selected split/subset pairs, then averages tasks equally.
This average describes the retrieval slice, not the full MTEB benchmark.

Use `--revision` to pin a checkpoint. Lower `--batch-size` for encoder memory or
`--query-batch` / `--document-chunk` for scoring memory. Corpus vectors stay in
host RAM. Use a separate output directory for each configuration.

## Inference

```python
from sentence_transformers import SparseEncoder

model = SparseEncoder("lightonai/SlideOn", trust_remote_code=True)
queries = model.encode_query(["What is sparse retrieval?"], max_active_dims=32)
documents = model.encode_document(["Sparse retrieval uses inverted indexes."])
scores = model.similarity(queries, documents)
```

Set `max_active_dims=N` to keep the top-N latents with the highest weights in
each pooled vector and set the rest to zero. Omit it or use `None` for the
saved unpruned defaults. Positive budgets larger than the embedding dimension
keep all dimensions. `0` is invalid. `token_k` controls the independent per-token
activation budget (32 by default).

Use the `final/` directory from your own run to load a locally trained model.
Saved models include their backbone, sparse head, tokenizer,
and loading code.

## Citation

If you find this work useful for your research, please consider citing the original authors:

```bibtex
@misc{moura2026slideon,
  author       = {Moura Junior, Paulo Roberto and Sourty, Raphaël},
  title        = {{SlideOn} \& {mSlideOn}: Sparsified late-interaction and dense retrievers},
  year         = {2026},
  howpublished = {Hugging Face Blog},
  url          = {https://huggingface.co/blog/lightonai/slideon-mslideon}
}
```
