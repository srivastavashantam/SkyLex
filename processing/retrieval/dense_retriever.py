"""
processing/retrieval/dense_retriever.py

R1 — Dense Retriever (Naive RAG).

Kya karta hai:
  Query ko embed karta hai aur ChromaDB mein cosine similarity search karta hai.
  Yeh sabse simple RAG variant hai — sirf semantic meaning pe based hai.

Kab best kaam karta hai:
  - Conceptual queries: "fuel reserve requirements kya hain"
  - Paraphrased queries: "landing rollout distance" → "stopping distance on runway"
  - Long descriptive queries

Kab fail karta hai:
  - Exact term queries: "§ 121.135" — exact section number
  - AD numbers: "AD-2023-15-08"
  - Short precise queries: "DGCA CAR Section 7"

Monitoring:
  - tqdm: Query batch progress
  - MLflow: Latency, chunk counts, similarity scores
  - LangSmith: @traceable — function-level tracing
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
from langsmith import traceable
from tqdm import tqdm

from monitoring.logger import get_logger
from processing.vector_store import CANDIDATE_STRATEGIES, QueryResult, SkyLexVectorStore

# Dense retrieval operations, latency metrics aur monitoring events trace karne ke liye.
logger = get_logger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """
    Ek query ka complete retrieval result — saare RAG variants mein same structure.

    Attributes:
        query        : Original query string
        source       : Kaunse source se retrieve kiya (FAA_CFR, etc.)
        strategy     : Kaunsi chunking strategy ki collection (recursive, etc.)
        rag_variant  : Kaunsa RAG variant use hua (dense, bm25, hybrid, etc.)
        chunk_ids    : Retrieved chunk IDs
        documents    : Actual chunk texts
        metadatas    : Per-chunk metadata
        scores       : Relevance scores (similarity ya BM25 score)
        latency_ms   : Total retrieval time in milliseconds
        extra        : Extra info per variant (reranker scores, etc.)
    """

    query: str
    source: str
    strategy: str
    rag_variant: str
    chunk_ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    scores: list[float]
    latency_ms: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def top_document(self) -> str:
        """Sabse relevant chunk ka text."""
        return self.documents[0] if self.documents else ""

    @property
    def result_count(self) -> int:
        """Kitne chunks retrieve hue."""
        return len(self.documents)


# ── Dense Retriever ───────────────────────────────────────────────────────────


class DenseRetriever:
    """
    R1 — Dense Retriever (Naive RAG).

    ChromaDB vector similarity search use karta hai.
    Query embed hoti hai aur nearest neighbors return hote hain.

    Usage:
        retriever = DenseRetriever()
        results = retriever.retrieve(
            query="fuel reserve requirements",
            query_embedding=[0.023, -0.156, ...],
            source="FAA_CFR",
            strategy="recursive",
            n_results=5,
        )
    """

    # RAG variant identifier — experiment tracking mein use hoga.
    rag_variant: str = "dense"

    def __init__(self, store: SkyLexVectorStore | None = None) -> None:
        # Store inject karo ya fresh banao — testing ke liye inject useful hai.
        self._store = store or SkyLexVectorStore()

    @traceable(name="dense_retrieve", project_name="skylex")
    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        source: str,
        strategy: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """
        Ek query ke liye dense retrieval karo.

        Process:
          1. ChromaDB mein cosine similarity search
          2. Top-n_results chunks return karo
          3. Latency measure karo

        Args:
            query           : Original query text (LangSmith tracing ke liye)
            query_embedding : Pre-computed query vector (1536-dim)
            source          : FAA_CFR, FAA_AD, etc.
            strategy        : recursive, hierarchical, etc.
            n_results       : Kitne chunks chahiye (default: 5)
            where           : Optional metadata filter

        Returns:
            RetrievalResult with chunks, scores, latency.
        """
        start_time = time.perf_counter()

        # ChromaDB query.
        query_result: QueryResult = self._store.query(
            query_embedding=query_embedding,
            source=source,
            strategy=strategy,
            n_results=n_results,
            where=where,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # ChromaDB cosine distance ko similarity mein convert karo.
        # ChromaDB distance = 1 - cosine_similarity
        # Isliye: similarity = 1 - distance
        scores = [
            round(1.0 - dist, 4)
            for dist in query_result.distances
        ]

        logger.debug(
            "Dense retrieve | source=%s | strategy=%s | results=%d | latency=%.1fms",
            source, strategy, len(query_result.documents), latency_ms,
        )

        return RetrievalResult(
            query=query,
            source=source,
            strategy=strategy,
            rag_variant=self.rag_variant,
            chunk_ids=query_result.chunk_ids,
            documents=query_result.documents,
            metadatas=query_result.metadatas,
            scores=scores,
            latency_ms=round(latency_ms, 2),
            extra={"top_similarity": scores[0] if scores else 0.0},
        )

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
        Multiple queries ke liye batch dense retrieval.

        tqdm progress bar + MLflow metrics logging included.

        Args:
            queries          : List of query strings
            query_embeddings : Pre-computed embeddings — queries ke saath 1:1 mapping
            source           : FAA_CFR, FAA_AD, etc.
            strategy         : recursive, hierarchical, etc.
            n_results        : Top-k per query
            mlflow_run_name  : MLflow nested run name — None matlab log nahi karo

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

        # tqdm progress bar — batch retrieval progress monitor karne ke liye.
        with tqdm(
            total=len(queries),
            desc=f"Dense Retrieve | {source} × {strategy}",
            unit="query",
            colour="blue",
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
                    "last_latency": f"{result.latency_ms:.1f}ms",
                    "results": result.result_count,
                })

        # MLflow metrics log karo agar run name diya hai.
        if mlflow_run_name:
            self._log_batch_metrics(
                results=results,
                source=source,
                strategy=strategy,
                total_latency_ms=total_latency_ms,
                mlflow_run_name=mlflow_run_name,
            )

        logger.info(
            "Dense batch complete | source=%s | strategy=%s | queries=%d | "
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
        """MLflow mein batch retrieval metrics log karo."""
        if not results:
            return

        avg_latency = total_latency_ms / len(results)
        avg_results = sum(r.result_count for r in results) / len(results)
        avg_top_score = sum(
            r.extra.get("top_similarity", 0.0) for r in results
        ) / len(results)

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
                "avg_results_count": round(avg_results, 2),
                "avg_top_similarity":round(avg_top_score, 4),
            })


# ── Convenience Functions ─────────────────────────────────────────────────────


def retrieve_all_strategies(
    query: str,
    query_embedding: list[float],
    source: str,
    n_results: int = 5,
    store: SkyLexVectorStore | None = None,
) -> dict[str, RetrievalResult]:
    """
    Ek source ke saari candidate strategies pe dense retrieval karo.

    RAGAS evaluation mein same query ko alag chunking strategies se
    compare karne ke liye use hoga.

    Args:
        query           : Query text
        query_embedding : Pre-computed query vector
        source          : FAA_CFR, FAA_AD, etc.
        n_results       : Top-k per strategy
        store           : Optional pre-initialized store

    Returns:
        Dict mapping strategy → RetrievalResult
    """
    retriever = DenseRetriever(store=store)
    strategies = CANDIDATE_STRATEGIES.get(source, [])
    results: dict[str, RetrievalResult] = {}

    with tqdm(
        total=len(strategies),
        desc=f"Dense | All Strategies | {source}",
        unit="strategy",
        colour="cyan",
    ) as pbar:
        for strategy in strategies:
            results[strategy] = retriever.retrieve(
                query=query,
                query_embedding=query_embedding,
                source=source,
                strategy=strategy,
                n_results=n_results,
            )
            pbar.update(1)

    return results