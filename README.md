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

The launcher trains online: frozen DenseOn encodes batches of texts, and the SAE
learns from their token vectors held temporarily in memory. It does not write a
full embedding cache. Edit the constants at the top of `run_pretraining.sh`, then
run `bash run_pretraining.sh`. Set `TEXTS_JSONL` to local `.jsonl` files,
pretokenized `.pt` descriptors, or a mixture of both,
`INDEX_DIR` to the compact text index, and `SAE_OUTPUT` to a fresh output directory.

The SAE config selects DenseOn and AuxK, with 32,768 latents, token top-k 32,
a global token batch of 4,096, and AdamW at 0.0003. The launcher uses these SAE
settings and three epochs, with **192-token query/document limits** and
`ENCODE_BATCH_SIZE=512`. These 512 texts are one physical DenseOn batch per GPU;
they are separate from the 4,096 token vectors used in each global SAE update.
Reduce the encoding batch if GPU memory is insufficient. The launcher passes
its editable settings as CLI arguments, overriding the corresponding JSON values.

#### Compact pretokenized inputs (no text JSONL required)

To avoid saving large text exports, tokenize the three corpora directly on CPU:

```bash
pip install -e '.[data]'
python scripts/sae-pretrain/prepare_input_ids.py --source msmarco-v1
python scripts/sae-pretrain/prepare_input_ids.py --source msmarco-v2
python scripts/sae-pretrain/prepare_input_ids.py --source fineweb
```

These commands write `data/msmarco-v1.pt`, `data/msmarco-v2.pt` and
`data/fineweb.pt`. FineWeb streams from the same pinned SmolLM-Corpus release as
the text exporter; MSMARCO uses the same `ir_datasets` passage collections.
Only raw passage text is tokenized, without titles. `ir_datasets` may still cache
its upstream corpus download, but this path creates no intermediate JSONL.
Preparation loads the tokenizer, never DenseOn weights or dense embeddings.

You can also pass dataset IDs directly instead of using a preset:

```bash
python scripts/sae-pretrain/prepare_input_ids.py \
    --source ir-datasets --dataset-id msmarco-passage \
    --source-label msmarco_v1 --output data/msmarco-v1.pt

python scripts/sae-pretrain/prepare_input_ids.py \
    --source huggingface --dataset-id HuggingFaceTB/smollm-corpus \
    --dataset-config fineweb-edu-dedup --split train --text-field text \
    --revision 3ba9d605774198c5868892d7a8deda78031a781f \
    --source-label fineweb --output data/fineweb.pt
```

These are alternatives to the preset commands. Generic `ir-datasets` inputs read
`doc.text` from `docs_iter()`. Generic Hugging Face inputs stream one text column
and use `--id-field id` by default, falling back to row numbers when IDs are absent.
They are treated as documents. Without `--source-label`, the dataset ID becomes
the stored source label. Without `--revision`, generic Hub inputs resolve `main`
to an immutable commit recorded in the output; the `fineweb` preset stays pinned.

Each `.pt` is a small descriptor referencing a sibling `NAME.shards/` directory.
Keep the descriptor and its shard directory together when copying the data.
Shards contain one flat tensor of packed token IDs (two bytes per token), offsets
marking sequence boundaries, and source/role/ID metadata. They contain no text,
padding, attention masks, or embeddings. Batch padding and masks are reconstructed
before each DenseOn forward. Tensor shards are memory-mapped with a bounded open
shard cache; the whole corpus is not loaded into RAM. A plain tensor saved with
`torch.save()` is not sufficient: the descriptor records the tokenizer/settings
needed to interpret it correctly. Serialized files use weights-only loading.

Both length limits default to **192**, matching `run_pretraining.sh`. To prepare
for 256-token training, add `--query-length 256 --document-length 256` to **each**
preparation command, and set both limits to 256 in the launcher. Prompts, tokenizer
revision and both limits must match exactly at training time; a mismatch fails
early. Long texts are truncated with the same role-specific prompts and special
tokens as the JSONL path. Different limits require a new tokenized export.
Current `.pt` support is for the English DenseOn backbone without token skip lists.

Replace the entries at the top of `run_pretraining.sh`, for example:

```bash
TEXTS_JSONL=(
    "data/msmarco-v1.pt"
    "data/msmarco-v2.pt"
    "data/fineweb.pt"
    # "data/msmarco-v1-artificial-queries-0.jsonl"
)
INDEX_DIR="data/slideon-pretokenized-index"
SAE_OUTPUT="outputs/slideon-sae-pretokenized"
```

Then run `bash run_pretraining.sh`. The array keeps its original name; each file's
suffix selects the loader. The Python CLI also accepts `--inputs` as an alias for
`--texts-jsonl`. Use a fresh index/output when switching an existing run's inputs.
Shuffling, frozen DenseOn encoding, SAE updates and the final checkpoint format
are unchanged. Fine-tune by pointing `SAE_ID_OR_PATH` in `run_finetune.sh` at
`outputs/slideon-sae-pretokenized/final`.

Existing JSONL files (including generated queries) can also be tokenized:

```bash
python scripts/sae-pretrain/prepare_input_ids.py --source jsonl \
    --texts-jsonl data/msmarco-v1-artificial-queries-0.jsonl \
    --output data/msmarco-v1-artificial-queries-0.pt
```

Use `--limit 1000` for a small preparation check, `--tokenize-batch-size` to control
CPU tokenization batches, and `--shard-tokens` to control shard size. The default
is eight million input tokens per shard. Completed outputs are not overwritten;
choose another output path for a new export. Preparation restarts from the
beginning after interruption; a hard kill may leave `.partial` outputs to remove.

#### Raw JSONL inputs

Each JSONL line is an independent text, for example:

```json
{"id":"p123","source":"msmarco_v1","role":"document","text":"A passage to encode."}
{"id":"q456","source":"msmarco_v1","role":"query","text":"A question to encode?"}
{"id":"f789","source":"fineweb","text":"A web document to encode."}
```

Build the three corpus files and the five artificial-query files with:

```bash
pip install -e '.[data]'
python scripts/sae-pretrain/prepare_msmarco.py --version 1
python scripts/sae-pretrain/prepare_msmarco.py --version 2
python scripts/sae-pretrain/prepare_fineweb.py
python scripts/sae-pretrain/prepare_artificial_queries.py --input-dir /path/to/pickles
```

Outputs default to this repository's `data/` directory, regardless of the working
directory: `msmarco-v1.jsonl`, `msmarco-v2.jsonl`, `fineweb.jsonl`, and
`msmarco-v1-artificial-queries-0.jsonl` through `-4.jsonl`. Each has a
`.manifest.json` sidecar containing provenance, row counts and an output checksum.
Generated data is ignored by Git.

The MSMARCO exporter uses the full passage collections
[`msmarco-passage`](https://ir-datasets.com/msmarco-passage.html) and
[`msmarco-passage-v2`](https://ir-datasets.com/msmarco-passage-v2.html), copying
only `doc.text` as provided by `ir_datasets`. FineWeb streams the
[`fineweb-edu-dedup` subset of SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/viewer/fineweb-edu-dedup),
copying only `text`, from a pinned Hub revision. Titles and metadata are not
prepended. No tokenization or truncation happens during export. Blank texts are
skipped and counted because the SAE input contract requires nonblank texts;
other texts, whitespace and duplicates are preserved.

The artificial-query exporter reads your trusted
`top5_doct5queries-_part_0.pkl` through `top5_doct5queries-_part_4.pkl` dictionaries.
It keeps integer IDs as strings and assigns `role="query"`. Each whole pickle
part must fit in RAM; only one part is loaded at a time. Use `--parts 0 2` to
export selected parts, or omit `--input-dir` when the pickles are already in
`data/`. The other three files contain documents. The launcher lists these three
corpora by default, with optional query-part lines ready to uncomment.

The JSONL exporters accept `--limit N` for small output checks and `--overwrite` to
replace a completed export. A limit does not necessarily limit upstream downloads:
`ir_datasets` may download the collection before yielding its first passage.
Failures do not publish a partial final JSONL. Export restarts from the beginning;
after a hard interruption, remove any stale `.partial` files before retrying.

`source` and nonblank `text` are required strings. `role` defaults to `document`;
`id` is optional. Texts and duplicates are preserved, with no balancing, negative
selection or deduplication. Queries and documents use their existing prompts and
length limits; long texts are truncated, not split into chunks. At least one
document is required for the unchanged SAE initialization.

Before online training, a CPU scan validates and fingerprints the JSONL files,
records byte offsets and counts retained tokens with the tokenizer. This builds
an index of approximately 16 bytes per text plus metadata, without running
DenseOn or saving embeddings. The index supports random text access and is
checked for compatibility on reuse. Keep it under `data/`, as in the launcher,
to keep generated index files out of Git.
For `.pt` inputs the index reads saved sequence lengths without retokenizing;
fingerprints cover the descriptor and every tensor shard.

Each epoch shuffles all texts without replacement and visits each once. Frozen
DenseOn encodes them in batches; retained token vectors are shuffled in a bounded
in-memory pool and consumed in global SAE token batches. Leftover vectors carry
across encoder batches and epochs. Only the final incomplete global token batch
at the end of all epochs is dropped. The token counts determine the update and
learning-rate schedule; `MAX_STEPS` optionally caps it. There is no source
reweighting. Dictionary initialization still samples up to 128 documents per
source, irrespective of how many input files contain that source's documents.

One selected GPU uses plain Python and builds or validates the index before
training. With multiple selected GPUs, the launcher first runs CPU-only
`--online --index-only` in a single process, then starts `torchrun`. This phase
loads the tokenizer but no DenseOn model, and completes the full file fingerprint
checks before workers start. Distributed workers require the completed index and
check file paths, sizes and modification times without repeating that long scan.
Do not change input files during a run.

If launching `torchrun` directly, first run the same training arguments with plain
Python, `CUDA_VISIBLE_DEVICES=""` and the additional `--index-only` flag. Keep
the input files, index directory and encoding settings identical; omit
`--index-only` from the subsequent `torchrun` command. `run_pretraining.sh`
performs these two commands automatically, avoiding distributed synchronization
timeouts while indexing a large corpus.

Each GPU runs frozen DenseOn as well as an SAE replica. The global SAE token batch
is divided across workers, while `ENCODE_BATCH_SIZE` applies per GPU. This avoids
embedding-cache disk usage at the cost of re-encoding texts each epoch.

Training writes `metrics.jsonl` and saves the completed SAE under `final/`.
Set `RESUME_CHECKPOINT` to a `training_state.pt` file to resume, keeping the input
files, encoding settings, seed, world size, and both batch sizes unchanged. The
trainer reconstructs pending token vectors from recorded text positions. The
Python CLI also supports `--stop-after N` to stop without changing the planned
learning-rate schedule.

#### Original training texts without an embedding cache

`run_pretraining_original.sh` now exports the authors' English training texts and
uses online frozen DenseOn encoding. Edit its settings at the top, then run:

```bash
bash run_pretraining_original.sh
```

The new `scripts/sae-pretrain/export_original_texts.py` exporter reuses the same
selection function as the cached preparation path: the mining-score filter,
first ten eligible negatives, seven sampled negatives, and query/document ID
deduplication within each source. It writes source/role/text JSONL, with no
ranking relationships or teacher losses. Repeated text under distinct IDs stays
repeated. The 1.57 million pair rows therefore do not map one-to-one to output
text records. For the same dataset revision and selection settings, the selected
text pools match the cached preparation path.

By default, the export is `data/slideon-original-texts.jsonl`; its manifest records
the resolved HF revision, selection settings, counts and checksum. Compatible
completed exports are verified and reused. Incomplete or incompatible exports
are rejected; choose a fresh output path after changing data selection. Export
uses CPU only and loads one source's tables at a time. The usual HF raw-data
downloads and text index still occupy disk, but dense token vectors stay in
memory and are recreated each epoch.

The launcher keeps the original model/loss/optimizer settings, three epochs and
8,192-token limits. Its encoding batch defaults to four texts per GPU at those
long limits. Online token order and the exact retained-token epoch/update budget
differ from the cached path's buffered order and approximate raw-token schedule;
this is not an identical training trajectory. Shortening the length limits or
changing the number of epochs introduces further differences.

The default final SAE is `outputs/slideon-sae-original-online/final`. Set
`SAE_ID_OR_PATH` in `run_finetune.sh` to that directory. Existing cached-workflow
checkpoints cannot resume into this online workflow; use a new training output.
The launcher leaves existing embedding caches untouched.

#### Optional: the original cached workflow

The Python script still supports preparing all frozen embeddings first and then
training from that cache. Without `--texts-jsonl`, preparation uses the authors'
HF data; add `--texts-jsonl file1.jsonl file2.jsonl` to prepare local texts instead:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sae-pretrain/train_slideon_sae.py \
  --prepare --cache token-cache/slideon

torchrun --standalone --nproc_per_node=8 scripts/sae-pretrain/train_slideon_sae.py \
  --cache token-cache/slideon --output outputs/slideon-sae
```

These commands use the JSON defaults, including the 8,192-token length limits;
the online launcher's 192-token overrides do not apply. The FP16 cache costs
1,536 bytes per retained token with the 768-dimensional DenseOn backbone. All
inputs must finish preparation before training can use the cache. Interrupted
local-JSONL encoding restarts uncommitted streams from the beginning. Changing
input files or encoding settings requires a new cache directory.

HF preparation records dataset revisions; `--dataset-revisions pins.json` accepts
a mapping of dataset IDs to exact commits. To resume cached training, pass
`--resume path/to/training_state.pt` and retain the same config, cache, world size
and `--buffer-tokens` setting. This cached workflow remains separate from the
online launcher.

The completed SAE is saved to `SAE_OUTPUT/final`. In `run_finetune.sh`, set
`SAE_ID_OR_PATH="outputs/slideon-sae-texts/final"` (or your chosen output path).
The default DenseOn revision already matches; if you change the SAE backbone,
also match `BACKBONE_REVISION` in the fine-tuning launcher.

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

To prepare the authors' English data, fine-tune and evaluate the local result,
edit the constants at the top of `run_finetune.sh`, then run the script. For example:

```bash
# Edit these lines inside run_finetune.sh:
export CUDA_VISIBLE_DEVICES="0,1"
BATCH_SIZE=4
GLOBAL_BATCH_SIZE=128
RESUME_CHECKPOINT=""  # Or a saved checkpoint-N directory.
```

```bash
pip install -e '.[eval]'
bash run_finetune.sh
```

The launcher creates `english.jsonl` if it is missing, then fine-tunes from the
released SAE with `SOURCE_SAMPLING_ALPHA=0.5` and `SAMPLE_NEGATIVES=7`; an empty
value disables either option independently. Set GPU selection, paths, checkpoint
pins, batch sizes, token limits and training settings in the same configuration
block. Multiple selected GPUs automatically use `torchrun`, and accumulation is
derived from the global batch size. Remaining loss and optimizer settings stay
in `TRAIN_CONFIG`. The default output is `outputs/slideon-alpha05-neg7`.

After successful training, it evaluates that directory's `final/` model on BEIR-15
with 64 query / 512 document active dimensions and encoder batch size 8. Empty
`EVAL_QUERY_LENGTH` and `EVAL_DOCUMENT_LENGTH` preserve the checkpoint's saved
limits; empty `EVAL_OUTPUT` derives the results directory automatically.
Set `PREPARE_DATA_IF_MISSING=false` to require existing training data, or
`RUN_EVALUATION=false` to stop after training.

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

`run_eval.sh` evaluates one checkpoint per run. Open the file and edit the
assignments at the top. For example, to evaluate our fine-tuned model with
128-token limits on BEIR-Small, replace the corresponding lines with:

```bash
MODEL_ID_OR_PATH="outputs/slideon-alpha05-neg7/final"
EVAL_SUITE="BEIR-Small"
QUERY_LENGTH=128
DOCUMENT_LENGTH=128
EVAL_OUTPUT="results/our-finetune-small-128"
```

Save the file, then run this command from the repository directory:

```bash
bash run_eval.sh
```

The default evaluates the released fine-tuned `lightonai/SlideOn` on BEIR with
64 query / 512 document active dimensions. `QUERY_LENGTH` and `DOCUMENT_LENGTH`
default to empty strings, preserving each checkpoint's saved token limit;
either can be set independently. `BATCH_SIZE` defaults to 8. An empty
`EVAL_OUTPUT` generates a directory under `results/` named from the model, suite,
token limits and sparsity settings.

| Weights | `MODEL_ID_OR_PATH` example |
| --- | --- |
| Released SAE | `lightonai/SlideOn-SAE` |
| Our pretrained SAE | `outputs/slideon-sae/final` |
| Released fine-tuned model | `lightonai/SlideOn` |
| Our fine-tuned model | `outputs/slideon-alpha05-neg7/final` |

Relative paths are resolved from the repository. `MODEL_REVISION=auto` preserves
the launcher's existing release pins for the two author models and omits a Hub
revision for local paths. Set an explicit commit for another Hub revision, or an
empty string to resolve the current default branch. Use a new output directory
when changing the revision or other evaluation settings.

The same Python entry point can also be called directly:

```bash
pip install -e '.[eval]'
python scripts/eval/evaluate.py --model lightonai/SlideOn --trust-remote-code \
  --suite beir --query-max-active-dims 64 --document-max-active-dims 512 --output results/slideon

python scripts/eval/evaluate.py --model lightonai/SlideOn --trust-remote-code \
  --suite mteb-eng-v2 --output results/slideon-mteb-eng-v2

# Evaluate an already fine-tuned local model without running training again:
python scripts/eval/evaluate.py \
  --model outputs/slideon-alpha05-neg7/final --trust-remote-code \
  --suite beir --query-max-active-dims 64 --document-max-active-dims 512 \
  --output outputs/slideon-alpha05-neg7/beir-q64-d512
```

`--model` accepts a Hub ID, a complete local Sentence Transformers checkpoint
(`final/` or `checkpoint-N/`, containing `modules.json`), or a local raw SAE
checkpoint (`config.json` plus `sae.safetensors`). Pass the checkpoint itself,
not its parent training-output directory. Complete local models load directly
from disk. For raw SAE checkpoints, the evaluator uses the existing SAE loader
to attach the recorded backbone and preserve saved query/document lengths;
the backbone may need downloading. This temporary conversion does not modify
the checkpoint. Benchmark datasets may also require downloads.

For a local model, `run_config.json` records the resolved path and a SHA256
fingerprint of the inference files. Changed weights, tokenizer, configuration or
custom code require a new evaluation output directory, preventing stale scores
from being reused. Optimizer state and generated caches are excluded from this
fingerprint. Raw SAE fingerprints also include the resolved backbone commit and
local conversion code, and exclude training resume state. Keep evaluation output
outside the checkpoint directory.

The English workflows cover BEIR-15 and the retrieval tasks from
`MTEB(eng, v2)` (10 tasks). The MTEB v2 suite preserves the benchmark's
evaluation splits and language subsets and excludes the separate `InstructionRetrieval`
task type. `--suite BEIR-Small` selects the nine BEIR datasets remaining after
excluding ClimateFEVER, DBPedia, HotpotQA, MSMARCO, NQ and FEVER.
Use `--tasks SciFact` for a single-dataset run.

Evaluation uses exact dot-product search. Vectors are unpruned by default;
`--query-max-active-dims` and `--document-max-active-dims` control final pruning. Outputs include MTEB
results, `summary.json` with scores on a 0–1 scale, and `efficiency.json` with
average nonzeros and retrieval FLOPS (expected active-latent overlap).
BEIR reports average nDCG@10. The English MTEB v2 suite averages each task's
nDCG@10 across its selected split/subset pairs, then averages tasks equally.
This average describes the retrieval slice, not the full MTEB benchmark.

Use `--revision` to pin a Hub checkpoint; omit it for local directories.
Use `--query-length` and/or `--document-length` to override saved token limits
for evaluation only; omitting either preserves that role's saved limit.
Lower `--batch-size` for encoder memory or
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

Use the `final/` directory from your own fine-tuning run to load a locally trained
model this way. Saved fine-tuned models include their backbone, sparse head,
tokenizer and loading code; raw SAE outputs need assembly as described above.

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
