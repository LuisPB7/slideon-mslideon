"""Original SAE text selection, shared by cached preparation and JSONL export."""

import random


# Join three source tables by ID and build unique text pools for SAE preparation.
# scores rows contain query_id, document_ids, scores; the first document is positive.
# queries maps query_id -> query; documents maps document_id -> document.
# These mining scores select texts; cross-encoder teacher scores are not used here.
def filter_texts(scores, queries, documents, seed=42, nv_threshold=0.95,
                 num_negatives=10, sampled_negatives=7):
    """Filter negatives, sample the configured count, and deduplicate text IDs."""
    docs = dict(zip(documents["document_id"], documents["document"]))
    query_text = dict(zip(queries["query_id"], queries["query"]))
    kept_docs, kept_queries = {}, {}
    # A local seeded RNG makes sampled negatives reproducible without affecting
    # unrelated Python random state. Deduplication is within this one split only.
    rng = random.Random(seed)
    for row in scores:
        ids, scores_, qid = row["document_ids"], row["scores"], row["query_id"]
        # Keep the first num_negatives candidates whose mining score is below the
        # configured fraction of the positive score, and whose document text exists.
        negatives = [did for did, score in zip(ids[1:], scores_[1:]) if score < nv_threshold * scores_[0] and docs.get(did)][:num_negatives]
        if len(negatives) < num_negatives or not query_text.get(qid) or not docs.get(ids[0]):
            continue
        # Include every retained query and positive, plus sampled negative texts.
        # Dictionaries deduplicate IDs; multiple positives for a query remain docs.
        kept_queries[qid] = query_text[qid]
        kept_docs[ids[0]] = docs[ids[0]]
        for did in rng.sample(negatives, sampled_negatives):
            kept_docs[did] = docs[did]
    # Drop pairwise relationships deliberately: an SAE reconstructs individual
    # token embeddings, so preparation needs text pools rather than labeled triples.
    return dict(documents=list(kept_docs.values()), queries=list(kept_queries.values()))
