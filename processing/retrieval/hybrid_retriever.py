"""
processing/retrieval/hybrid_retriever.py

R3 — Hybrid Retriever with Reciprocal Rank Fusion (RRF).

Kya karta hai:
  Dense retriever (ChromaDB) aur Sparse retriever (BM25) dono se results
  fetch karta hai, phir RRF algorithm se ek unified ranking banata hai.

Kyun yeh best hai:
  - Dense: Semantic meaning pakadta hai — "fuel reserve" matches "minimum fuel"
  - BM25:  Exact terms pakadta hai — "§ 121.135" exact section number
  - RRF:   Dono lists ko rank-based fusion se combine karta hai —
           actual scores compare nahi karta (alag scales hain)

RRF Formula:
  score(doc) = Σ 1 / (k + rank(doc))
  k = 60 (Robertson et al. 2009 — most stable across domains)

Research backing:
  Hybrid RRF ne Recall@5 = 0.816 achieve kiya — dense-only se 39% better.
  (2025 RAG benchmark, technical document retrieval)

Monitoring:
  - tqdm: Query batch progress
  - MLflow: Latency, fusion stats, result counts
  - LangSmith: @traceable — function-level tracing
"""

from __future__ import annotations

import time
from typing import Any

import mlflow
from langsmith import traceable
from tqdm import tqdm

from monitoring.logger import get_logger
from processing.retrieval.bm25_indexer import load_bm25_index, tokenize
from processing.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from processing.retrieval.sparse_retriever import SparseRetriever
from processing.vector_store import CANDIDATE_STRATEGIES, SkyLexVectorStore

# Hybrid retrieval events, RRF fusion steps aur monitoring metrics trace karne ke liye.
logger = get_logger(__name__)

# RRF constant k — Robertson et al. 2009 se recommended value.
# k=60 pe top ranks zyada dominate nahi karte, results stable rehte hain.
_RRF_K: int = 60


# ── RRF Core Algorithm ────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """
    Multiple ranked lists ko ek unified ranking mein merge karo using RRF.

    Kaise kaam karta hai:
      Har list mein har document ka rank position dekho.
      Har document ka RRF score calculate karo: Σ 1/(k + rank).
      Sabse high RRF score wala document sabse top pe aata hai.

    Kyun rank use karte hain score nahi:
      Dense score (0.87) aur BM25 score (18.4) directly compare nahi ho sakte —
      alag scales hain. Rank position scale-independent hai.

    Args:
        ranked_lists : List of ranked chunk_id lists.
                       Har inner list ek retriever ka output hai —
                       index 0 = best match, index 1 = second best, etc.
        k            : RRF constant (default: 60)

    Returns:
        List of (chunk_id, rrf_score) tuples — descending order mein sorted.

    Example:
        dense_results = ["chunk_A", "chunk_B", "chunk_C"]
        bm25_results  = ["chunk_C", "chunk_A", "chunk_D"]

        chunk_A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
        chunk_B: 1/(60+2)             = 0.0161
        chunk_C: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
        chunk_D: 1/(60+2)             = 0.0161

        Final: [chunk_A(0.0325), chunk_C(0.0323), chunk_B(0.0161), chunk_D(0.0161)]
    """
    # Har chunk_id ka cumulative RRF score store karo.
    rrf_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            # rank 0-indexed hai — rank+1 se 1-indexed karo.
            # Formula: 1 / (k + rank_position)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                1.0 / (k + rank + 1)
            )

    # Descending order mein sort karo — highest RRF score pehle.
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return sorted_results


# ── Hybrid Retriever ──────────────────────────────────────────────────────────


class HybridRetriever:
    """
    R3 — Hybrid Retriever (Dense + BM25 + RRF Fusion).

    Internally DenseRetriever aur SparseRetriever dono use karta hai,
    phir RRF se merge karta hai.

    Retrieval multiplier:
      Final n_results chahiye hain, lekin RRF ke liye zyada candidates chahiye.
      Isliye Dense aur BM25 se n_results * fetch_multiplier chunks fetch karte hain,
      phir RRF ke baad top n_results rakhte hain.
      Default fetch_multiplier = 3 — research-backed sweet spot.

    Usage:
        retriever = HybridRetriever()
        result = retriever.retrieve(
            query="§ 121.135 fuel reserve IFR requirements",
            query_embedding=[0.023, -0.156, ...],
            source="FAA_CFR",
            strategy="recursive",
            n_results=5,
        )
    """

    # RAG variant identifier — experiment tracking mein use hoga.
    rag_variant: str = "hybrid_rrf"

    def __init__(
        self,
        store: SkyLexVectorStore | None = None,
        rrf_k: int = _RRF_K,
        fetch_multiplier: int = 3,
    ) -> None:
        """
        Args:
            store            : SkyLexVectorStore — inject karo ya fresh banao.
            rrf_k            : RRF constant k (default: 60).
            fetch_multiplier : Dense aur BM25 se kitne zyada chunks fetch karein
                               RRF ke liye. n_results * fetch_multiplier candidates
                               fetch honge, phir top n_results rakhenge.
        """
        self._store = store or SkyLexVectorStore()
        self._rrf_k = rrf_k
        self._fetch_multiplier = fetch_multiplier

        # Dense aur Sparse retrievers initialize karo — same store share karte hain.
        self._dense = DenseRetriever(store=self._store)
        self._sparse = SparseRetriever()

    @traceable(name="hybrid_rrf_retrieve", project_name="skylex")
    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        source: str,
        strategy: str,
        n_results: int = 5,
    ) -> RetrievalResult:
        """
        Ek query ke liye hybrid RRF retrieval karo.

        Process:
          Step 1 — Dense retrieval: n_results * fetch_multiplier chunks fetch karo.
          Step 2 — BM25 retrieval:  n_results * fetch_multiplier chunks fetch karo.
          Step 3 — RRF fusion:      Dono lists merge karo rank-based scoring se.
          Step 4 — Top n_results:   Final merged list se top chunks rakho.
          Step 5 — Original texts:  ChromaDB se actual chunk texts fetch karo.

        Args:
            query           : Original query text
            query_embedding : Pre-computed query vector (1536-dim)
            source          : FAA_CFR, FAA_AD, etc.
            strategy        : recursive, hierarchical, etc.
            n_results       : Final output mein kitne chunks chahiye

        Returns:
            RetrievalResult with RRF-fused chunks, scores, fusion stats.
        """
        start_time = time.perf_counter()

        # Fetch kandidates — final n_results se zyada, RRF ke liye pool chahiye.
        fetch_n = n_results * self._fetch_multiplier

        # Step 1 — Dense retrieval.
        dense_result = self._dense.retrieve(
            query=query,
            query_embedding=query_embedding,
            source=source,
            strategy=strategy,
            n_results=fetch_n,
        )

        # Step 2 — BM25 retrieval (original texts ke saath).
        sparse_result = self._sparse.retrieve_with_original_texts(
            query=query,
            source=source,
            strategy=strategy,
            n_results=fetch_n,
            store=self._store,
        )

        # Step 3 — RRF fusion.
        # Dono lists ke chunk IDs pass karo — rank order mein.
        rrf_results = reciprocal_rank_fusion(
            ranked_lists=[
                dense_result.chunk_ids,
                sparse_result.chunk_ids,
            ],
            k=self._rrf_k,
        )

        # Step 4 — Top n_results rakho.
        top_rrf = rrf_results[:n_results]
        top_chunk_ids = [chunk_id for chunk_id, _ in top_rrf]
        top_rrf_scores = [round(score, 6) for _, score in top_rrf]

        # Step 5 — ChromaDB se original texts fetch karo by chunk IDs.
        ordered_docs, ordered_meta = self._fetch_original_texts(
            chunk_ids=top_chunk_ids,
            source=source,
            strategy=strategy,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Fusion stats — debugging aur analysis ke liye useful.
        dense_only = set(dense_result.chunk_ids) - set(sparse_result.chunk_ids)
        sparse_only = set(sparse_result.chunk_ids) - set(dense_result.chunk_ids)
        both = set(dense_result.chunk_ids) & set(sparse_result.chunk_ids)

        logger.debug(
            "Hybrid RRF | source=%s | strategy=%s | "
            "dense=%d | bm25=%d | overlap=%d | final=%d | latency=%.1fms",
            source, strategy,
            len(dense_result.chunk_ids), len(sparse_result.chunk_ids),
            len(both), len(top_chunk_ids), latency_ms,
        )

        return RetrievalResult(
            query=query,
            source=source,
            strategy=strategy,
            rag_variant=self.rag_variant,
            chunk_ids=top_chunk_ids,
            documents=ordered_docs,
            metadatas=ordered_meta,
            scores=top_rrf_scores,
            latency_ms=round(latency_ms, 2),
            extra={
                "rrf_k":           self._rrf_k,
                "fetch_multiplier": self._fetch_multiplier,
                "dense_count":     len(dense_result.chunk_ids),
                "bm25_count":      len(sparse_result.chunk_ids),
                "overlap_count":   len(both),
                "dense_only":      len(dense_only),
                "bm25_only":       len(sparse_only),
                "top_rrf_score":   top_rrf_scores[0] if top_rrf_scores else 0.0,
            },
        )

    def _fetch_original_texts(
        self,
        chunk_ids: list[str],
        source: str,
        strategy: str,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """
        ChromaDB se chunk IDs ke basis pe original texts aur metadata fetch karo.

        RRF ne chunk IDs ka order decide kiya — ab us order mein texts chahiye.
        ChromaDB.get() order guarantee nahi karta, isliye ID-based mapping use karo.

        Returns:
            Tuple of (documents, metadatas) — chunk_ids ke saath 1:1 order.
        """
        if not chunk_ids:
            return [], []

        try:
            collection = self._store._collections.get((source, strategy))
            if collection is None:
                logger.warning(
                    "Collection not found | source=%s | strategy=%s", source, strategy
                )
                return [""] * len(chunk_ids), [{}] * len(chunk_ids)

            fetch_result = collection.get(
                ids=chunk_ids,
                include=["documents", "metadatas"],  # type: ignore[arg-type]
            )

            raw_docs: list[str] = fetch_result.get("documents") or []
            raw_meta: list[dict[str, Any]] = fetch_result.get("metadatas") or []  # type: ignore[assignment]
            raw_ids: list[str] = fetch_result.get("ids") or []

            # ID → (text, metadata) map banao — order preserve karne ke liye.
            id_to_doc: dict[str, str] = dict(zip(raw_ids, raw_docs))
            id_to_meta: dict[str, dict[str, Any]] = dict(zip(raw_ids, raw_meta))

            # RRF order mein texts return karo.
            ordered_docs = [id_to_doc.get(cid, "") for cid in chunk_ids]
            ordered_meta = [id_to_meta.get(cid, {}) for cid in chunk_ids]

            return ordered_docs, ordered_meta

        except Exception as e:
            logger.warning(
                "Text fetch failed | source=%s | strategy=%s | error=%s",
                source, strategy, e,
            )
            return [""] * len(chunk_ids), [{}] * len(chunk_ids)

    def retrieve_batch(
        self,
        queries: list[str],
        query_embeddings: list[list[float]],
        source: str,
        strategy: str,
        n_results: int = 5,
        mlflow_run_name: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Multiple queries ke liye batch hybrid RRF retrieval.

        tqdm progress bar + MLflow metrics logging included.

        Args:
            queries          : List of query strings
            query_embeddings : Pre-computed embeddings — queries ke saath 1:1 mapping
            source           : FAA_CFR, FAA_AD, etc.
            strategy         : recursive, hierarchical, etc.
            n_results        : Top-k per query (final, after RRF)
            mlflow_run_name  : MLflow nested run name

        Returns:
            List of RetrievalResult — queries ke saath 1:1 mapping.
        """
        if len(queries) != len(query_embeddings):
            raise ValueError(
                f"queries ({len(queries)}) aur query_embeddings "
                f"({len(query_embeddings)}) ki length equal honi chahiye."
            )

        results: list[RetrievalResult] = []
        total_latency_ms = 0.0

        # tqdm progress bar — green colour hybrid ke liye.
        with tqdm(
            total=len(queries),
            desc=f"Hybrid RRF | {source} × {strategy}",
            unit="query",
            colour="green",
        ) as pbar:
            for query, embedding in zip(queries, query_embeddings):
                result = self.retrieve(
                    query=query,
                    query_embedding=embedding,
                    source=source,
                    strategy=strategy,
                    n_results=n_results,
                )
                results.append(result)
                total_latency_ms += result.latency_ms
                pbar.update(1)
                pbar.set_postfix({
                    "latency":  f"{result.latency_ms:.1f}ms",
                    "overlap":  result.extra.get("overlap_count", 0),
                    "results":  result.result_count,
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
            "Hybrid RRF batch complete | source=%s | strategy=%s | "
            "queries=%d | avg_latency=%.1fms",
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
        """MLflow mein batch hybrid retrieval metrics log karo."""
        if not results:
            return

        avg_latency = total_latency_ms / len(results)
        avg_overlap = sum(
            r.extra.get("overlap_count", 0) for r in results
        ) / len(results)
        avg_top_rrf = sum(
            r.extra.get("top_rrf_score", 0.0) for r in results
        ) / len(results)

        with mlflow.start_run(run_name=mlflow_run_name, nested=True):
            mlflow.log_params({
                "rag_variant":      self.rag_variant,
                "source":           source,
                "strategy":         strategy,
                "query_count":      len(results),
                "rrf_k":            self._rrf_k,
                "fetch_multiplier": self._fetch_multiplier,
            })
            mlflow.log_metrics({
                "avg_latency_ms":   round(avg_latency, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "avg_overlap":      round(avg_overlap, 2),
                "avg_top_rrf_score":round(avg_top_rrf, 6),
            })