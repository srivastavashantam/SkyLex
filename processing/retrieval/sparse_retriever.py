"""
processing/retrieval/sparse_retriever.py

R2 — Sparse Retriever (BM25 Only).

Kya karta hai:
  Query ko tokenize karta hai aur BM25 index mein keyword search karta hai.
  Exact terms, section numbers, AD codes ke liye best hai.

Kab best kaam karta hai:
  - Exact section references: "§ 121.135"
  - AD numbers: "AD-2023-15-08"
  - Regulatory codes: "DGCA CAR Section 7"
  - Short precise queries with specific terminology

Kab fail karta hai:
  - Paraphrased queries: "stopping distance" vs "landing rollout"
  - Conceptual queries: "what happens if fuel runs out"
  - Queries with synonyms: "pilot" vs "flight crew member"

Monitoring:
  - tqdm: Query batch progress
  - MLflow: Latency, BM25 scores, chunk counts
  - LangSmith: @traceable — function-level tracing
"""

from __future__ import annotations

import time
from typing import Any

import mlflow
import numpy as np
from langsmith import traceable
from tqdm import tqdm

from monitoring.logger import get_logger
from processing.retrieval.bm25_indexer import load_bm25_index, tokenize
from processing.retrieval.dense_retriever import RetrievalResult
from processing.vector_store import CANDIDATE_STRATEGIES

# Sparse retrieval operations, BM25 scoring aur monitoring events trace karne ke liye.
logger = get_logger(__name__)


# ── Sparse Retriever ──────────────────────────────────────────────────────────


class SparseRetriever:
    """
    R2 — Sparse Retriever (BM25 Only).

    Disk se BM25 index load karta hai aur keyword-based search karta hai.
    Query embed karne ki zarurat nahi — zero API cost, instant.

    Index caching:
      Ek baar load karne ke baad index memory mein cache ho jaata hai.
      Dobara same source × strategy query karne pe disk se load nahi hoga.

    Usage:
        retriever = SparseRetriever()
        result = retriever.retrieve(
            query="§ 121.135 certificate holder requirements",
            source="FAA_CFR",
            strategy="recursive",
            n_results=5,
        )
    """

    # RAG variant identifier — experiment tracking mein use hoga.
    rag_variant: str = "bm25"

    def __init__(self) -> None:
        # In-memory cache — source_strategy → loaded index data.
        # Ek baar load karo, baar baar use karo.
        self._index_cache: dict[str, dict[str, Any]] = {}

    def _get_index(self, source: str, strategy: str) -> dict[str, Any]:
        """
        BM25 index load karo — cache miss pe disk se, cache hit pe memory se.

        Cache key format: "FAA_CFR__recursive"
        """
        cache_key = f"{source}__{strategy}"

        if cache_key not in self._index_cache:
            logger.debug(
                "BM25 index cache miss — loading from disk | source=%s | strategy=%s",
                source, strategy,
            )
            self._index_cache[cache_key] = load_bm25_index(source, strategy)
            logger.debug(
                "BM25 index cached | source=%s | strategy=%s | chunks=%d",
                source, strategy,
                self._index_cache[cache_key]["chunk_count"],
            )

        return self._index_cache[cache_key]

    @traceable(name="sparse_retrieve", project_name="skylex")
    def retrieve(
        self,
        query: str,
        source: str,
        strategy: str,
        n_results: int = 5,
    ) -> RetrievalResult:
        """
        Ek query ke liye BM25 sparse retrieval karo.

        Process:
          1. Query tokenize karo (same tokenizer jo index build mein use hua)
          2. BM25 scores calculate karo saare chunks ke liye
          3. Top-n_results chunks return karo

        Args:
            query     : Original query text
            source    : FAA_CFR, FAA_AD, etc.
            strategy  : recursive, hierarchical, etc.
            n_results : Kitne chunks chahiye (default: 5)

        Returns:
            RetrievalResult with chunks, BM25 scores, latency.
        """
        start_time = time.perf_counter()

        # BM25 index load karo (cache se ya disk se).
        index_data = self._get_index(source, strategy)
        bm25_index = index_data["bm25_index"]
        chunk_ids: list[str] = index_data["chunk_ids"]
        corpus_tokens: list[list[str]] = index_data["corpus_tokens"]

        # Query tokenize karo — same tokenizer jo index build mein use hua.
        # Consistency zaroori hai — alag tokenizer alag tokens dega.
        query_tokens = tokenize(query)

        if not query_tokens:
            # Query mein sirf stopwords ya punctuation hai — empty result.
            logger.warning(
                "Empty query tokens after tokenization | query='%s'", query
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RetrievalResult(
                query=query,
                source=source,
                strategy=strategy,
                rag_variant=self.rag_variant,
                chunk_ids=[],
                documents=[],
                metadatas=[],
                scores=[],
                latency_ms=round(latency_ms, 2),
                extra={"query_tokens": [], "warning": "empty_tokens"},
            )

        # BM25 scores calculate karo — saare chunks ke liye ek saath.
        # get_scores() numpy array return karta hai.
        scores_array: np.ndarray = bm25_index.get_scores(query_tokens)

        # Top-n_results indices find karo — argsort descending.
        safe_n = min(n_results, len(chunk_ids))
        top_indices: np.ndarray = np.argsort(scores_array)[::-1][:safe_n]

        # Results assemble karo.
        top_chunk_ids = [chunk_ids[i] for i in top_indices]
        top_scores = [round(float(scores_array[i]), 4) for i in top_indices]

        # Corpus tokens se document text reconstruct karo.
        # Note: Yeh tokenized text hai, original text nahi.
        # Original text ke liye ChromaDB fetch karna padega.
        # Abhi ke liye tokenized text use karte hain — RAGAS ke liye original chahiye.
        # isliye corpus_tokens se joined text use karte hain as placeholder.
        top_documents = [
            " ".join(corpus_tokens[i]) for i in top_indices
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            "BM25 retrieve | source=%s | strategy=%s | tokens=%d | "
            "results=%d | top_score=%.4f | latency=%.1fms",
            source, strategy, len(query_tokens),
            len(top_chunk_ids), top_scores[0] if top_scores else 0.0,
            latency_ms,
        )

        return RetrievalResult(
            query=query,
            source=source,
            strategy=strategy,
            rag_variant=self.rag_variant,
            chunk_ids=top_chunk_ids,
            documents=top_documents,
            metadatas=[{} for _ in top_chunk_ids],
            scores=top_scores,
            latency_ms=round(latency_ms, 2),
            extra={
                "query_tokens":  query_tokens,
                "top_bm25_score": top_scores[0] if top_scores else 0.0,
            },
        )

    def retrieve_with_original_texts(
        self,
        query: str,
        source: str,
        strategy: str,
        n_results: int = 5,
        store: Any = None,
    ) -> RetrievalResult:
        """
        BM25 retrieval karo aur ChromaDB se original texts fetch karo.

        Kyun zaroorat hai:
          BM25 index mein tokenized text store hai — original chunk text nahi.
          RAGAS evaluation ke liye original text chahiye.
          Isliye BM25 se chunk IDs milte hain, phir ChromaDB se original text.

        Args:
            query     : Query text
            source    : FAA_CFR, etc.
            strategy  : recursive, etc.
            n_results : Top-k
            store     : SkyLexVectorStore instance — original texts fetch karne ke liye

        Returns:
            RetrievalResult with original chunk texts.
        """
        # Pehle BM25 se top chunk IDs nikalo.
        bm25_result = self.retrieve(
            query=query,
            source=source,
            strategy=strategy,
            n_results=n_results,
        )

        if not bm25_result.chunk_ids or store is None:
            return bm25_result

        # ChromaDB se original texts fetch karo by chunk IDs.
        try:
            collection = store._collections.get((source, strategy))
            if collection is None:
                return bm25_result

            fetch_result = collection.get(
                ids=bm25_result.chunk_ids,
                include=["documents", "metadatas"],  # type: ignore[arg-type]
            )

            original_docs: list[str] = fetch_result.get("documents") or []
            original_meta: list[dict[str, Any]] = fetch_result.get("metadatas") or []

            # Order preserve karo — collection.get() order guarantee nahi karta.
            # chunk_id → (text, metadata) map banao.
            id_to_doc: dict[str, str] = dict(
                zip(fetch_result["ids"], original_docs)
            )
            id_to_meta: dict[str, dict[str, Any]] = dict(
                zip(fetch_result["ids"], original_meta)
            )

            ordered_docs = [
                id_to_doc.get(cid, "") for cid in bm25_result.chunk_ids
            ]
            ordered_meta = [
                id_to_meta.get(cid, {}) for cid in bm25_result.chunk_ids
            ]

            return RetrievalResult(
                query=bm25_result.query,
                source=bm25_result.source,
                strategy=bm25_result.strategy,
                rag_variant=bm25_result.rag_variant,
                chunk_ids=bm25_result.chunk_ids,
                documents=ordered_docs,
                metadatas=ordered_meta,
                scores=bm25_result.scores,
                latency_ms=bm25_result.latency_ms,
                extra=bm25_result.extra,
            )

        except Exception as e:
            logger.warning(
                "Original text fetch failed | source=%s | strategy=%s | error=%s",
                source, strategy, e,
            )
            return bm25_result

    def retrieve_batch(
        self,
        queries: list[str],
        source: str,
        strategy: str,
        n_results: int = 5,
        mlflow_run_name: str | None = None,
        store: Any = None,
    ) -> list[RetrievalResult]:
        """
        Multiple queries ke liye batch BM25 retrieval.

        tqdm progress bar + MLflow metrics logging included.

        Args:
            queries          : List of query strings
            source           : FAA_CFR, FAA_AD, etc.
            strategy         : recursive, hierarchical, etc.
            n_results        : Top-k per query
            mlflow_run_name  : MLflow nested run name
            store            : SkyLexVectorStore — original texts ke liye

        Returns:
            List of RetrievalResult — queries ke saath 1:1 mapping.
        """
        results: list[RetrievalResult] = []
        total_latency_ms = 0.0

        # tqdm progress bar.
        with tqdm(
            total=len(queries),
            desc=f"BM25 Retrieve | {source} × {strategy}",
            unit="query",
            colour="yellow",
        ) as pbar:
            for query in queries:
                if store is not None:
                    result = self.retrieve_with_original_texts(
                        query=query,
                        source=source,
                        strategy=strategy,
                        n_results=n_results,
                        store=store,
                    )
                else:
                    result = self.retrieve(
                        query=query,
                        source=source,
                        strategy=strategy,
                        n_results=n_results,
                    )

                results.append(result)
                total_latency_ms += result.latency_ms
                pbar.update(1)
                pbar.set_postfix({
                    "latency": f"{result.latency_ms:.1f}ms",
                    "top_score": f"{result.scores[0]:.2f}" if result.scores else "0",
                })

        # MLflow log karo.
        if mlflow_run_name:
            self._log_batch_metrics(
                results=results,
                source=source,
                strategy=strategy,
                total_latency_ms=total_latency_ms,
                mlflow_run_name=mlflow_run_name,
            )

        logger.info(
            "BM25 batch complete | source=%s | strategy=%s | queries=%d | "
            "avg_latency=%.1fms",
            source, strategy, len(queries),
            total_latency_ms / len(queries) if queries else 0,
        )

        return results

    def _log_batch_metrics(
        self,
        results: list[RetrievalResult],
        source: str,
        strategy: str,
        total_latency_ms: float,
        mlflow_run_name: str,
    ) -> None:
        """MLflow mein batch BM25 retrieval metrics log karo."""
        if not results:
            return

        avg_latency = total_latency_ms / len(results)
        avg_top_score = sum(
            r.extra.get("top_bm25_score", 0.0) for r in results
        ) / len(results)
        empty_results = sum(1 for r in results if r.result_count == 0)

        with mlflow.start_run(run_name=mlflow_run_name, nested=True):
            mlflow.log_params({
                "rag_variant":  self.rag_variant,
                "source":       source,
                "strategy":     strategy,
                "query_count":  len(results),
            })
            mlflow.log_metrics({
                "avg_latency_ms":    round(avg_latency, 2),
                "total_latency_ms":  round(total_latency_ms, 2),
                "avg_top_bm25_score":round(avg_top_score, 4),
                "empty_results":     empty_results,
            })