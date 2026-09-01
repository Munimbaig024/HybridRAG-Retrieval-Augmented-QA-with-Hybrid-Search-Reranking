# HybridRAG — Design Reference

Retrieval-Augmented QA with hybrid search (dense + BM25), cross-encoder reranking, and a real evaluation set.

---

## 1. Pipeline overview

```
User query
   |
   +--------------------+
   |                     |
Dense retrieval      BM25 retrieval
(semantic, top-k)    (lexical, top-k)
   |                     |
   +--------- Fusion ----+
           (RRF)
              |
       Cross-encoder rerank
       (scores top ~50 candidates)
              |
        Top-k context
              |
        LLM answer (Groq)
```

1. **Dense retrieval** — embed the query, cosine-search a vector index (e.g. Chroma, Qdrant, pgvector, FAISS). Catches paraphrases and conceptual matches.
2. **BM25 retrieval** — lexical scoring over the same corpus (e.g. `rank_bm25` in Python, or Elasticsearch/OpenSearch's built-in BM25). Catches exact terms, IDs, names.
3. **Fusion** — merge the two ranked lists into one. Default: Reciprocal Rank Fusion (RRF), which only needs rank positions, not raw scores.
4. **Cross-encoder rerank** — take the fused top ~50 candidates and score each (query, chunk) pair with a cross-encoder model. Much more accurate than the bi-encoder used for dense retrieval, but too slow to run over the whole corpus — so it only touches the shortlist.
5. **LLM answer generation** — feed the top-k reranked chunks + query to Groq's API, generate the final answer.
6. **Evaluation** — a separate, offline loop: run test queries through the pipeline, score retrieval quality (precision/recall/MRR/nDCG against a labeled qrels set) and answer faithfulness (is the generated answer actually supported by the retrieved context).

---

## 2. Concepts

### 2.1 BM25 (sparse / lexical retrieval)

A scoring function over word overlap — TF-IDF's more refined cousin. For each query term it rewards a document that:

- contains the term often, with **diminishing returns** (term frequency saturation — the 10th occurrence barely matters more than the 5th)
- contains a term that's **rare across the corpus** (rare words are more informative — inverse document frequency)
- isn't **abnormally long** relative to other documents (length normalization, so long docs don't win purely by containing more words)

No embeddings, no training — pure statistics over word counts. Fast, cheap, and strong at exact-match cases dense retrieval can miss (error codes, SKUs, names, acronyms).

**Libraries**: `rank_bm25` (pure Python, good for prototyping), Elasticsearch/OpenSearch (production-grade, built-in BM25).

### 2.2 Hybrid search (fusion)

Run dense and BM25 retrieval independently, then merge the two ranked lists. They aren't on the same numeric scale (cosine similarity vs. BM25 score), so you can't just add them directly. Two standard approaches:

**Reciprocal Rank Fusion (RRF)** — ignores raw scores, uses only each document's rank position in each list:

```
RRF_score(doc) = sum over each list of  1 / (k + rank_in_that_list)
```

`k` is a constant (commonly 60) that dampens the effect of very low ranks. No tuning, no score normalization needed — this is the default choice for most hybrid setups.

**Weighted score combination** — normalize both score sets (e.g. min-max scaling to [0,1]) and blend:

```
final_score = w * normalized_dense_score + (1 - w) * normalized_bm25_score
```

More tunable, but requires babysitting the weight `w`.

**Why bother**: dense retrieval catches paraphrases and conceptual similarity; BM25 catches exact lexical matches. Hybrid covers both failure modes at once.

### 2.3 Cross-encoder reranking

Your dense retriever is a **bi-encoder**: it embeds the query and each document *separately*, then compares vectors with cosine similarity. This is what makes it fast enough to search millions of chunks — documents are pre-embedded offline, and search is just a nearest-neighbor lookup.

A **cross-encoder** instead takes the query and *one candidate document together* as a single input and outputs a relevance score directly. The model attends across both texts jointly, so it captures interactions a bi-encoder's separate embeddings can't. This is far more accurate, but far slower — nothing can be precomputed, so it must run fresh per query-document pair.

**Standard pattern**: use hybrid search to cheaply narrow the corpus down to a shortlist (e.g. top 50–100), then run the cross-encoder over just that shortlist to produce the final top 5–10 chunks for the LLM's context.

**Common models**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, good baseline), `BAAI/bge-reranker-base` or `-large` (stronger, slower). Both available via `sentence-transformers`.

### 2.4 Evaluation

Move past "vibes-based testing" — read a few outputs, decide it looks fine. Build an actual eval set and score it.

**Retrieval precision (needs a labeled eval set)**

Build a small set of **query → relevant chunk(s)** pairs (often called *qrels*, relevance judgments). Options for building it:
- Manually label 30–100 queries against your known chunks.
- Generate synthetic questions per chunk with an LLM ("write a question this paragraph answers"), then use the source chunk as the known-relevant one.

Metrics to compute per query, then average:

| Metric | What it measures |
|---|---|
| Precision@k | Of the top-k retrieved chunks, what fraction are relevant? |
| Recall@k | Of all relevant chunks that exist, what fraction appear in the top-k? |
| MRR (Mean Reciprocal Rank) | How high up was the *first* relevant result — rewards ranking the right chunk near the top |
| nDCG | Like precision but position-weighted and supports graded relevance (not just relevant/irrelevant) |

**Answer faithfulness (separate axis from retrieval)**

Given the retrieved context and the LLM's generated answer: is every claim in the answer actually supported by the context, or is the model adding unsupported claims (hallucinating)?

- **LLM-as-judge** — give a strong LLM the context + generated answer, ask it to verify each claim is entailed by the context. This is the approach frameworks like RAGAS use.
- **NLI-based scoring** — use a natural language inference model to check entailment between each answer sentence and the retrieved chunks. Cheaper and more deterministic than an LLM judge.
- **Claim decomposition** — break the answer into atomic factual claims, verify each individually against the context. More fine-grained than scoring the whole answer at once.

Report faithfulness alongside **answer relevance** (does the answer actually address the question) as a separate metric — a faithful answer can still fail to answer the question, and a fluent-sounding answer can still be unfaithful to the context.

---

## 3. Suggested tech stack

| Component | Options |
|---|---|
| Chunking / embeddings | Your existing setup |
| Dense vector store | Chroma, Qdrant, FAISS, pgvector |
| BM25 | `rank_bm25` (prototyping) or Elasticsearch/OpenSearch (production) |
| Fusion | Hand-rolled RRF (simple, ~10 lines) |
| Cross-encoder | `sentence-transformers` + `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM generation | Groq API |
| Eval | Custom qrels set + a small script for precision/recall/MRR/nDCG; RAGAS or hand-rolled LLM-judge for faithfulness |

---

## 4. Build order (suggested)

1. Get dense-only RAG working end-to-end (you already have this foundation).
2. Add BM25 retrieval as a second, independent retriever over the same corpus.
3. Implement RRF fusion to merge the two ranked lists.
4. Add cross-encoder reranking on top of the fused shortlist.
5. Build the eval set (even 30–50 labeled queries is enough to start).
6. Write the retrieval metrics script (precision/recall/MRR/nDCG).
7. Add faithfulness scoring (start with LLM-as-judge — fastest to implement).
8. Iterate: use the eval numbers to tune fusion weights, reranker choice, and chunk size — not vibes.
