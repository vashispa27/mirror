# Practical RAG lab

This lab makes retrieval behavior visible before adding a framework or an LLM.
It uses Qdrant directly, local CPU embeddings through FastEmbed, and the small
BEIR SciFact benchmark. SciFact contains 5,183 documents, test queries, and
human relevance judgments, so experiments produce retrieval metrics instead of
subjective impressions.

## Why the lab does not start with LangChain

LangChain is useful for composing loaders, retrievers, prompts, tools, and
agents. It is not the retrieval algorithm. Starting with `qdrant-client` makes
the following decisions explicit:

- how documents are chunked;
- which dense and sparse representations are stored;
- how many candidates each retriever produces;
- how rankings are fused;
- whether a post-retrieval reranker helps;
- how the result is evaluated.

After the direct implementation is understood, install the optional
`frameworks` extra and reproduce one pipeline with LangChain. That comparison
will show what the framework saves and what it hides.

## Setup

Requires Python 3.11 and a Qdrant server at `http://localhost:6333`.

```bash
cd rag_lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

The first indexing run downloads two small ONNX models: a dense embedding model
and Qdrant's BM25 sparse model. No embedding API key or GPU is required.

## Lab 1 — inspect chunking

Run the same document through different algorithms:

```bash
rag-lab inspect --strategy whole
rag-lab inspect --strategy fixed --max-characters 500 --overlap-characters 100
rag-lab inspect --strategy sentence --max-characters 500 --overlap-sentences 1
rag-lab inspect --strategy recursive --max-characters 500
rag-lab inspect --strategy semantic --max-characters 500
```

Things to observe:

- Whole-document chunks preserve context but can dilute a specific fact.
- Fixed chunks are predictable but can split sentences and headings.
- Sentence chunks preserve language boundaries but depend on sentence quality.
- Recursive chunks prefer paragraphs, then sentences, then smaller units.
- Semantic chunks use embedding changes to detect topic boundaries; they cost
  more during indexing and introduce another model/threshold to tune.

## Lab 2 — build comparable indexes

Use a different collection for each chunking strategy:

```bash
rag-lab index --strategy whole --collection rag_lab_scifact_whole
rag-lab index --strategy sentence --max-characters 500 \
  --collection rag_lab_scifact_sentence_500
rag-lab index --strategy sentence --max-characters 1000 \
  --collection rag_lab_scifact_sentence_1000
```

Start with `--limit-documents 100` while learning. Use the full corpus before
running benchmark metrics, because the relevance judgments assume all documents
are present.

## Lab 3 — compare retrieval algorithms

Use the same query and collection so only the retrieval method changes:

```bash
rag-lab search "Vitamin D reduces the risk of multiple sclerosis" \
  --collection rag_lab_scifact_sentence_500 --mode dense

rag-lab search "Vitamin D reduces the risk of multiple sclerosis" \
  --collection rag_lab_scifact_sentence_500 --mode sparse

rag-lab search "Vitamin D reduces the risk of multiple sclerosis" \
  --collection rag_lab_scifact_sentence_500 --mode hybrid_rrf

rag-lab search "Vitamin D reduces the risk of multiple sclerosis" \
  --collection rag_lab_scifact_sentence_500 --mode hybrid_dbsf
```

- Dense search is strong on paraphrases and concepts.
- Sparse BM25 is strong on exact terminology, identifiers, and rare words.
- Hybrid search runs both and fuses their rankings.
- RRF is rank-based and a safe default when score scales are incomparable.
- DBSF normalizes score distributions before combining them.

## Lab 4 — post-retrieval reranking

Retrieve a wider candidate set, then score every query/document pair with a
cross-encoder:

```bash
rag-lab search "Vitamin D reduces the risk of multiple sclerosis" \
  --collection rag_lab_scifact_sentence_500 \
  --mode hybrid_rrf --candidate-limit 50 --rerank --limit 10
```

This is a post-search algorithm. It is usually more accurate and more expensive
than the first-stage retriever, so it should rerank tens of candidates—not the
entire corpus.

## Lab 5 — measure rather than guess

```bash
rag-lab evaluate --collection rag_lab_scifact_sentence_500 \
  --mode dense --max-queries 50 --k 10

rag-lab evaluate --collection rag_lab_scifact_sentence_500 \
  --mode sparse --max-queries 50 --k 10

rag-lab evaluate --collection rag_lab_scifact_sentence_500 \
  --mode hybrid_rrf --max-queries 50 --k 10

rag-lab evaluate --collection rag_lab_scifact_sentence_500 \
  --mode hybrid_rrf --candidate-limit 50 --rerank --max-queries 50 --k 10
```

The lab reports:

- `success@k`: whether at least one relevant document was retrieved;
- `recall@k`: how many known-relevant documents were retrieved;
- `MRR@k`: how early the first relevant document appeared;
- `nDCG@k`: ranking quality while respecting graded relevance.

Change one variable per experiment. Comparing a new chunker, embedding model,
retriever, top-k, and reranker simultaneously teaches nothing about causality.

## Where pre-search and generation fit

The implemented labs cover indexing, retrieval, fusion, and post-retrieval
reranking. The next labs should add these in order:

1. Metadata filters and query routing.
2. Query rewriting and multi-query expansion.
3. HyDE: generate a hypothetical answer, then retrieve using that text.
4. Parent-document retrieval: search small chunks, return larger parents.
5. Context assembly and an answer-generation prompt.
6. Answer evaluation with deterministic checks and then Ragas.
7. A LangChain version of one measured pipeline.
8. A HotpotQA subset for multi-hop and agentic retrieval.

Query rewriting, multi-query, HyDE, and generation require an LLM. They are
deliberately deferred until an LLM provider is selected, so the initial labs run
locally with no API key.

## Recommended primary references

- [Qdrant local quickstart](https://qdrant.tech/documentation/quickstart/)
- [Qdrant dense + sparse hybrid tutorial](https://qdrant.tech/documentation/tutorials-develop/hybrid-search-fastembed/)
- [Qdrant hybrid and multi-stage Query API](https://qdrant.tech/documentation/search/hybrid-queries/)
- [FastEmbed local embedding tutorial](https://qdrant.tech/documentation/tutorials-basics/generate-embeddings-fastembed/)
- [BEIR repository and evaluation examples](https://github.com/beir-cellar/beir)
- [LangChain learning tutorials](https://docs.langchain.com/oss/python/learn)
- [Ragas RAG evaluation tutorial](https://docs.ragas.io/en/stable/howtos/applications/evaluate-and-improve-rag/)
- [HotpotQA multi-hop benchmark](https://hotpotqa.github.io/)

