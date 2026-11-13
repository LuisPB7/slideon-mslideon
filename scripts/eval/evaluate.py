"""MTEB retrieval through the checkpoint's ST adapter; no ANN or normalization."""

# Reading order: main() selects tasks and loads a model, then MTEB calls the
# ExactRetrieval index()/search() interface. This file supplies ranked results;
# MTEB owns dataset relevance judgments and computes retrieval quality metrics.
# Here "no normalization" concerns final retrieval vectors; the SlideOn model
# can still normalize token embeddings internally before its sparse head.
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import sparse


# Named suites are convenience task lists, not model variants. BEIR contains
# 15 task names; a task such as CQADupstack may itself contain several subsets.
BEIR = ("ArguAna", "CQADupstackRetrieval", "ClimateFEVER", "DBPedia", "FEVER", "FiQA2018",
        "HotpotQA", "MSMARCO", "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS", "SciFact",
        "TRECCOVID", "Touche2020")
CODE = ("CodeSearchNetCCRetrieval", "COIRCodeSearchNetRetrieval", "CodeFeedbackST",
        "SyntheticText2SQL", "CodeFeedbackMT", "CodeEditSearchRetrieval", "StackOverflowQA",
        "CosQA", "AppsRetrieval", "CodeSearchNetRetrieval", "CodeTransOceanContest", "CodeTransOceanDL")
SUITES = dict(beir=BEIR, miracl=("MIRACLRetrievalHardNegatives",), mldr=("MultiLongDocRetrieval",), code=CODE)
MTEB_SUITES = {
    "mteb-eng-v2": "MTEB(eng, v2)",
    "mteb-multilingual-v2": "MTEB(Multilingual, v2)",
}


def write_json(path, value):
    # Replace a completed record atomically within its directory. Incremental
    # evaluation updates should not leave truncated JSON if serialization fails.
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def efficiency(queries, documents):
    """Expected active-latent overlap for independent uniform query/document draws."""
    # Inputs are CSR matrices [number_of_texts, vocabulary_size]. getnnz(axis=0)
    # counts the texts that use each latent, independent of activation magnitude.
    if queries.shape[0] == 0 or documents.shape[0] == 0:
        raise ValueError("Cannot measure an empty collection")
    q_frequency = np.asarray(queries.getnnz(axis=0), dtype=np.float64) / queries.shape[0]
    d_frequency = np.asarray(documents.getnnz(axis=0), dtype=np.float64) / documents.shape[0]
    # For latent j, P(query uses j) * P(document uses j) is its expected overlap
    # contribution. Summing gives a sparsity/search-work proxy named "flops";
    # it is not measured latency or a hardware instruction/operation count.
    # nnz totals divided by row counts report average active dimensions per text.
    return dict(queries=queries.shape[0], documents=documents.shape[0],
                query_nnz=queries.nnz / queries.shape[0], document_nnz=documents.nnz / documents.shape[0],
                flops=float(q_frequency @ d_frequency))


@torch.inference_mode()
def exact_dot(queries, documents, top_k, device="cpu", query_batch=64, document_chunk=16384):
    """Exhaustive sparse-document x dense-query multiplication, with bounded score blocks."""
    # Both inputs are CPU SciPy CSR matrices sharing vocabulary dimension V.
    # Return [number_of_queries, min(top_k, number_of_documents)] score/id arrays.
    # These ids are row positions, converted back to corpus IDs by search().
    # top_k here means documents to retrieve, not the model's latent/token budget.
    if min(top_k, query_batch, document_chunk) < 1:
        raise ValueError("Search sizes must be positive")
    if queries.shape[1] != documents.shape[1] or documents.shape[0] == 0:
        raise ValueError("Invalid embedding shapes")
    device = torch.device(device)
    k = min(top_k, documents.shape[0])
    # Keep only top-k on CPU; never materialize the full query x document score matrix.
    best_scores = torch.full((queries.shape[0], k), -torch.inf)
    best_ids = torch.full((queries.shape[0], k), -1, dtype=torch.long)

    # The outer loop visits EVERY corpus block. Blocking limits temporary score
    # memory, but does not approximate nearest neighbors or skip candidates.
    # The full corpus remains stored as CSR on CPU by ExactRetrieval.index().
    for start in range(0, documents.shape[0], document_chunk):
        block = documents[start:start+document_chunk]
        # CSR stores row boundaries, column indices, and nonzero values. Move
        # those three arrays to the scoring device without densifying documents.
        csr = torch.sparse_csr_tensor(
            torch.from_numpy(block.indptr.astype(np.int64)).to(device),
            torch.from_numpy(block.indices.astype(np.int64)).to(device),
            torch.from_numpy(block.data.astype(np.float32)).to(device), size=block.shape, check_invariants=True)

        # Only one query block is dense: [Qb, V]. Multiplication yields a bounded
        # [Qb, Db] score block, where Db is the current document-chunk length.
        for qstart in range(0, queries.shape[0], query_batch):
            qend = min(qstart+query_batch, queries.shape[0])
            dense = torch.from_numpy(queries[qstart:qend].toarray()).float().to(device)
            scores = torch.sparse.mm(csr, dense.T).T
            values, ids = scores.topk(min(k, block.shape[0]), dim=-1)

            # A block can contribute at most k winners to the global top-k.
            # Merge its winners with the previous best list and prune again.
            # Add start because block-local row IDs begin at zero each time.
            merged_scores = torch.cat((best_scores[qstart:qend].to(device), values), dim=1)
            merged_ids = torch.cat((best_ids[qstart:qend].to(device), ids+start), dim=1)
            selected, positions = merged_scores.topk(k, dim=-1)
            best_scores[qstart:qend] = selected.cpu()
            best_ids[qstart:qend] = merged_ids.gather(1, positions).cpu()

    # No dense [all_queries, all_documents] matrix is ever retained.
    return best_scores.numpy(), best_ids.numpy()


class ExactRetrieval:
    # Adapter between MTEB's retrieval interface and SparseEncoder. State consists
    # of the currently indexed corpus, its row-to-document-ID mapping, encoder /
    # scoring batch sizes, pruning budgets, and saved sparsity statistics.
    def __init__(self, model, model_id, revision, output, batch_size=8, query_max_active_dims=None,
                 document_max_active_dims=None, query_batch=64, document_chunk=16384):
        # batch_size controls Transformer encoding; query_batch/document_chunk
        # separately control score multiplication memory. model_id/revision let
        # MTEB identify the result cache with the checkpoint actually evaluated.
        from mteb.models.model_meta import ModelMeta, ScoringFunction
        self.model, self.output = model, Path(output)
        self.batch_size = batch_size
        self.query_max_active_dims, self.document_max_active_dims = query_max_active_dims, document_max_active_dims
        self.query_batch, self.document_chunk = query_batch, document_chunk
        self.mteb_model_meta = ModelMeta.create_empty().model_copy(update=dict(
            name=model_id, revision=revision, similarity_fn_name=ScoringFunction.DOT_PRODUCT, model_type=["sparse"]))
        # Keep statistics from completed tasks when continuing the same run.
        # The in-memory corpus starts empty and is replaced by index() per task.
        self.efficiency_path = self.output / "efficiency.json"
        self.statistics = json.loads(self.efficiency_path.read_text()) if self.efficiency_path.exists() else {}
        self.corpus = None

    def encode_dataset(self, dataset, task_metadata, task):
        # MTEB's loader formats benchmark records into text. The task argument
        # selects query/document preparation here AND the corresponding model
        # encode method, preserving the checkpoint's asymmetric prompt handling.
        from mteb._create_dataloaders import create_dataloader
        from mteb.types import PromptType
        loader = create_dataloader(dataset, task_metadata=task_metadata,
            prompt_type=PromptType.query if task == "query" else PromptType.document, batch_size=self.batch_size)
        blocks = []
        for batch in loader:
            # max_active_dims is a final-vector cap; it does not change the
            # per-token TopK setting stored in the checkpoint. None is passed
            # through to the adapter, which resolves its checkpoint defaults.
            vectors = getattr(self.model, "encode_" + task)(
                batch["text"], batch_size=self.batch_size, convert_to_tensor=True,
                show_progress_bar=False, max_active_dims=self.query_max_active_dims if task == "query" else self.document_max_active_dims)

            # Each encoding batch is briefly dense for a uniform conversion to
            # SciPy CSR. Subsequent storage retains only its nonzero weights.
            # Reject NaN/Inf before they can silently contaminate ranking metrics.
            if vectors.layout != torch.strided:
                vectors = vectors.to_dense()
            array = vectors.detach().float().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError("Non-finite model embeddings")
            blocks.append(sparse.csr_matrix(array))
        if not blocks:
            raise ValueError("Dataset is empty")
        # Row order is preserved, which is essential for the ID mappings used
        # by index() and search(). This method returns vectors, not text IDs.
        return sparse.vstack(blocks, format="csr")

    def index(self, corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs, num_proc=None):
        # MTEB calls index before search for a task/subset/split. "Index" here
        # means encode and retain CSR vectors; no ANN or inverted index is built.
        # Extra keyword arguments belong to MTEB's interface; encoder batching
        # is controlled by this adapter's own self.batch_size.
        self.corpus = None  # release the previous corpus before encoding the next one
        self.document_ids = [str(value) for value in corpus["id"]]
        print(f"Encoding {task_metadata.name}/{hf_subset}/{hf_split}: {len(corpus):,} documents", flush=True)
        self.corpus = self.encode_dataset(corpus, task_metadata, "document")

    def search(self, queries, *, task_metadata, hf_split, hf_subset, top_k, encode_kwargs,
               top_ranked=None, num_proc=None):
        # Retrieval considers the full indexed corpus. A preselected candidate
        # list would be a reranking request, which this adapter explicitly rejects.
        if top_ranked is not None:
            raise ValueError("This evaluator supports retrieval, not reranking")
        query_vectors = self.encode_dataset(queries, task_metadata, "query")
        # Include all three scope components so statistics from different
        # languages/subsets or splits do not overwrite one another.
        key = f"{task_metadata.name}/{hf_subset}/{hf_split}"
        self.statistics[key] = efficiency(query_vectors, self.corpus)
        write_json(self.efficiency_path, self.statistics)
        print(f"Exact dot search: {key}; {self.statistics[key]}", flush=True)
        scores, indices = exact_dot(query_vectors, self.corpus, top_k, device=self.model.device,
                                   query_batch=self.query_batch, document_chunk=self.document_chunk)

        # MTEB expects {query_id: {document_id: score}}, not positional arrays.
        # It compares this mapping with the task's qrels to compute metrics.
        return {str(qid): {self.document_ids[int(i)]: float(score) for i, score in zip(ids, values)}
                for qid, ids, values in zip(queries["id"], indices, scores)}


def select_tasks(names, subsets=None, splits=None, benchmark=None):
    # Resolve either explicit task names or the Retrieval portion of an MTEB
    # benchmark. Even a broad benchmark name does not run classification/STS
    # tasks through this retrieval-only adapter.
    import mteb
    tasks = ([task for task in mteb.get_benchmark(benchmark).tasks if task.metadata.type == "Retrieval"]
             if benchmark else list(mteb.get_tasks(tasks=names)))
    if not benchmark and {t.metadata.name for t in tasks} != set(names):
        raise ValueError("MTEB did not resolve every requested task")
    if not tasks:
        raise ValueError("No retrieval tasks selected")
    for task in tasks:
        if task.metadata.type != "Retrieval":
            raise ValueError(f"Not a retrieval task: {task.metadata.name}")

        # Explicit split overrides win. Native benchmark/code tasks otherwise
        # preserve MTEB's declared splits; the convenience retrieval lists use
        # dev for MS MARCO/MIRACL and test for the remaining tasks.
        desired = splits or (list(task.metadata.eval_splits) if benchmark or task.metadata.name in CODE else
                             ["dev"] if task.metadata.name in ("MSMARCO", "MIRACLRetrievalHardNegatives") else ["test"])
        if not set(desired).issubset(task.metadata.eval_splits):
            raise ValueError(f"Invalid splits for {task.metadata.name}: {desired}; available: {task.metadata.eval_splits}")
        task.metadata.eval_splits = list(desired)
        if subsets:
            # Subsets may be languages or other benchmark partitions. Validate
            # against EACH task rather than silently dropping unavailable ones.
            if not set(subsets).issubset(task.hf_subsets):
                raise ValueError(f"Invalid subsets for {task.metadata.name}: {subsets}; available: {task.hf_subsets}")
            task.hf_subsets = list(subsets)
    return tasks


def select_suites(suites, subsets=None, splits=None):
    # Return one deduplicated task list plus suite->task membership for summaries.
    # A task appearing in several suites should run only once, provided every
    # suite asks for the same splits/subsets; otherwise its result is ambiguous.
    tasks, members = {}, {}
    for suite in suites:
        selected = select_tasks(SUITES.get(suite), subsets, splits, MTEB_SUITES.get(suite))
        members[suite] = [task.metadata.name for task in selected]
        for task in selected:
            name = task.metadata.name
            if name in tasks and (
                set(tasks[name].metadata.eval_splits) != set(task.metadata.eval_splits)
                or set(tasks[name].hf_subsets) != set(task.hf_subsets)
            ):
                raise ValueError(f"Suites select different scopes for {name}; run them separately")
            tasks[name] = task
    return list(tasks.values()), members


def summarize(results, tasks, suite_members):
    # Convert MTEB result objects into a compact report. tasks is the full
    # requested scope; results can be a prefix while evaluation is still running.
    # Custom retrieval lists report nDCG@10; native MTEB benchmark suites use
    # each task's declared main_score (which need not be nDCG@10).
    by_name = {task.metadata.name: task for task in tasks}
    native_names = {name for suite, names in suite_members.items() if suite in MTEB_SUITES for name in names}
    metrics = {name: task.metadata.main_score if name in native_names else "ndcg_at_10"
               for name, task in by_name.items()}
    scores, ndcg_scores = {}, {}
    for result in results:
        name = result.task_name
        task = by_name[name]
        # Index result rows by (split, subset) so completeness checks and means
        # follow the selected task scope, independent of result serialization order.
        rows = {(split, row.get("hf_subset")): row
                for split, rows in result.scores.items() for row in rows}
        expected = [(split, subset) for split in task.metadata.eval_splits for subset in task.hf_subsets]
        metric = metrics[name]
        if any(key not in rows or metric not in rows[key] for key in expected):
            raise ValueError(f"Incomplete {metric} results for {name}")
        # First average within a task across its selected subset/split rows.
        # Keep nDCG separately so a task shared with a native benchmark can also
        # contribute to a convenience suite's nDCG average.
        scores[name] = float(np.mean([rows[key][metric] for key in expected]))
        if all("ndcg_at_10" in rows[key] for key in expected):
            ndcg_scores[name] = float(np.mean([rows[key]["ndcg_at_10"] for key in expected]))
    averages = {}
    for suite, members in suite_members.items():
        values = scores if suite in MTEB_SUITES else ndcg_scores
        metric = "main_score" if suite in MTEB_SUITES else "ndcg_at_10"
        complete = all(name in values for name in members)
        # Then average task scores with equal task weight. An incomplete suite
        # gets a null average plus completion counts; partial results must not
        # masquerade as a full-suite score. Scores stay on MTEB's 0-1 scale.
        averages[suite] = dict(completed=sum(name in values for name in members), total=len(members),
                              **{metric: float(np.mean([values[name] for name in members])) if complete else None})
    metric = next(iter(set(metrics.values()))) if len(set(metrics.values())) == 1 else "task_specific"
    return dict(metric=metric, scale="0-1", task_scores=scores, task_metrics=metrics, suites=averages,
                task_scope={name: dict(splits=list(task.metadata.eval_splits), subsets=list(task.hf_subsets))
                            for name, task in by_name.items()})


def main():
    # 1. Read the evaluation request. Task lists and suites are mutually
    # exclusive; omitting both selects BEIR. Encoding and scoring chunk sizes
    # are separate because they constrain different intermediate allocations.
    parser = argparse.ArgumentParser(description="Evaluate an HF sparse checkpoint with exact dot search")
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument("--revision", help="Optional HF commit/tag; resolved commit is recorded")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow the checkpoint's custom ST adapter")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--suite", choices=list(SUITES) + list(MTEB_SUITES), nargs="+",
                       help="Retrieval suites; MTEB v2 suites preserve benchmark splits and subsets")
    group.add_argument("--tasks", nargs="+", help="Explicit MTEB retrieval task names")
    parser.add_argument("--subsets", nargs="+", help="For example: en (must exist in every selected task)")
    parser.add_argument("--splits", nargs="+", help="Override the selected tasks' evaluation splits")
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-max-active-dims", type=int, default=None, help="Positive final vector budget; omit for unpruned")
    parser.add_argument("--document-max-active-dims", type=int, default=None, help="Positive final vector budget; omit for unpruned")
    parser.add_argument("--batch-size", type=int, default=8, help="Encoder batch size")
    parser.add_argument("--query-batch", type=int, default=64, help="Scoring query block size")
    parser.add_argument("--document-chunk", type=int, default=16384, help="Scoring document block size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # This evaluator is a single process. Unlike training, torchrun would start
    # duplicate evaluations sharing the output directory rather than shard work.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run evaluation with python, not torchrun")
    if min(args.batch_size, args.query_batch, args.document_chunk) < 1 or any(
        cap is not None and cap <= 0 for cap in (args.query_max_active_dims, args.document_max_active_dims)
    ):
        parser.error("Invalid batch size or pruning budget")
    import mteb
    from huggingface_hub import HfApi
    from mteb.cache import ResultCache
    from sentence_transformers import SparseEncoder

    # 2. Resolve the exact task scope before constructing the encoder. Explicit
    # --tasks produces per-task summaries without inventing a suite membership.
    suites = args.suite or ([] if args.tasks else ["beir"])
    if args.tasks:
        tasks, suite_members = select_tasks(args.tasks, args.subsets, args.splits), {}
    else:
        tasks, suite_members = select_suites(suites, args.subsets, args.splits)

    # Resolve a branch/tag/default revision to its immutable Hub commit and use
    # that same commit for model loading and provenance. This CLI therefore
    # expects a Hub model ID rather than an arbitrary local checkpoint path.
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    config = dict(model=args.model, revision=revision, mteb_version=mteb.__version__, score="dot",
                  suites=suite_members,
                  query_max_active_dims=args.query_max_active_dims, document_max_active_dims=args.document_max_active_dims, batch_size=args.batch_size,
                  tasks={t.metadata.name: dict(splits=list(t.metadata.eval_splits), subsets=list(t.hf_subsets)) for t in tasks})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    # Reusing a directory is allowed only with the same recorded configuration.
    # This protects comparisons from mixing checkpoints, budgets, or task scopes.
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output contains a different evaluation configuration; use another directory")
    if not config_path.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    write_json(config_path, config)

    # 3. Load the checkpoint's Sentence Transformers adapter and switch it to
    # inference behavior. The model handles text->vector conversion; the wrapper
    # below handles corpus storage and exact dot-product ranking.
    model = SparseEncoder(args.model, revision=revision, trust_remote_code=args.trust_remote_code, device=args.device)
    model.eval()
    search = ExactRetrieval(model, args.model, revision, output, args.batch_size, args.query_max_active_dims,
                            args.document_max_active_dims, args.query_batch, args.document_chunk)
    cache = ResultCache(output / "mteb")
    results = []

    # 4. MTEB orchestrates each task's index/search/metric calls. Its cache stores
    # benchmark results; this file does not persist the corpus embeddings. Save
    # summary.json after each completed task so progress survives a later failure.
    for task in tasks:
        result = mteb.evaluate(search, [task], cache=cache, co2_tracker=False,
                               encode_kwargs={"batch_size": args.batch_size}, raise_error=True)
        results.extend(result.task_results)
        write_json(output / "summary.json", summarize(results, tasks, suite_members))
    # The final console summary has the same schema as the saved JSON artifact.
    print(json.dumps(summarize(results, tasks, suite_members), indent=2))


if __name__ == "__main__":
    main()
