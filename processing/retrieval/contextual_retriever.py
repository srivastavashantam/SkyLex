"""
processing/retrieval/contextual_retriever.py

R5 — Contextual Retriever (Anthropic 2024 approach — metadata-based).

=============================================================================
THE CORE PROBLEM: CHUNK CONTEXT LOSS
=============================================================================

Chunking ke waqt ek common problem hoti hai — chunks apna parent context
kho dete hain. Example:

  Original Document Title: "§ 121.135 — Contents of Manual"

  Chunk after splitting:
    "(a) Each manual required by this subpart must contain a table of
     contents and must be easy to revise."

  Agar koi query kare "§ 121.135 manual contents" — dense retriever
  is chunk ko miss kar sakta hai kyunki chunk text mein "§ 121.135"
  likha hi nahi hai. Context loss ho gaya chunking ke waqt.

=============================================================================
THE SOLUTION: CONTEXT INJECTION AT RETRIEVAL TIME
=============================================================================

Anthropic ka original approach (2024):
  GPT-3.5 se har chunk ke liye ek context summary generate karo aur
  chunk ke saath inject karo before embedding.

  Problem: Har chunk ke liye ek LLM call = 72,827 LLM calls = expensive.

Humara approach — Metadata-based Context Injection (Zero extra API cost):
  ChromaDB mein stored metadata use karo — source, doc_id, chunk_index,
  total_chunks, strategy. In fields se ek meaningful prefix banao.

  BEFORE (plain chunk text):
    "(a) Each manual required by this subpart must contain a table of
     contents and must be easy to revise."

  AFTER (context-injected):
    "[FAA_CFR | doc_id: faa_cfr_part121_135 | Chunk 3/12 | recursive]
     (a) Each manual required by this subpart must contain a table of
     contents and must be easy to revise."

  Yeh prefix retrieval ke waqt query ke saath match karta hai — no LLM needed.

=============================================================================
WHY THIS HELPS RETRIEVAL
=============================================================================

  Query: "§ 121.135 manual requirements"

  Without context: chunk text mein "§ 121.135" nahi hai → possible miss
  With context:    prefix mein "FAA_CFR" aur "faa_cfr_part121_135" hai
                   → better match probability

  Additionally: "Chunk 3/12" batata hai ki document mein yeh chunk kahan
  hai — LLM ko positional context milta hai answer generate karte waqt.

Monitoring:
  - tqdm: Batch retrieval progress
  - MLflow: Latency, context injection stats
  - LangSmith: @traceable — full trace with context-injected documents
"""

from __future__ import annotations

import time
from typing import Any

import mlflow
from langsmith import traceable
from tqdm import tqdm

from monitoring.logger import get_logger
from processing.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from processing.vector_store import CANDIDATE_STRATEGIES, SkyLexVectorStore

logger = get_logger(__name__)


# ── Context Builder ───────────────────────────────────────────────────────────


def build_context_prefix(metadata: dict[str, Any]) -> str:
    """
    Chunk ke metadata se ek human-readable context prefix banao.

    Yeh prefix chunk text ke saath concatenate hoga before returning to LLM.
    Dense retrieval already ho chuka hota hai — yeh prefix LLM ke liye context
    provide karta hai, retrieval ke liye nahi.

    Format: "[SOURCE | doc_id | Chunk X/Y | strategy]"

    Args:
        metadata: ChromaDB mein stored chunk metadata dict.
                  Expected keys: source, doc_id, chunk_index, total_chunks, strategy.

    Returns:
        Formatted context prefix string.

    Examples:
        metadata = {
            "source": "FAA_CFR",
            "doc_id": "faa_cfr_part121",
            "chunk_index": 3,
            "total_chunks": 12,
            "strategy": "recursive"
        }
        → "[FAA_CFR | faa_cfr_part121 | Chunk 3/12 | recursive]"

        Agar metadata empty hai:
        → "" (empty string — koi prefix nahi)
    """
    if not metadata:
        return ""

    source = metadata.get("source", "")
    doc_id = metadata.get("doc_id", "")
    chunk_index = metadata.get("chunk_index", "")
    total_chunks = metadata.get("total_chunks", "")
    strategy = metadata.get("strategy", "")

    # Sirf available fields use karo — missing fields skip karo.
    parts: list[str] = []

    if source:
        parts.append(source)
    if doc_id:
        # doc_id mein underscores hote hain — readable banana ke liye replace karo.
        parts.append(f"doc: {doc_id}")
    if chunk_index != "" and total_chunks != "":
        parts.append(f"Chunk {chunk_index}/{total_chunks}")
    if strategy:
        parts.append(f"strategy: {strategy}")

    if not parts:
        return ""

    return f"[{' | '.join(parts)}]\n"


def inject_context(document: str, metadata: dict[str, Any]) -> str:
    """
    Document text mein context prefix inject karo.

    LLM ko yeh enriched text milega — prefix se document ka origin clear hota hai
    aur LLM better grounded answers generate kar sakta hai.

    Args:
        document : Original chunk text.
        metadata : Chunk ka metadata dict from ChromaDB.

    Returns:
        Context prefix + original document text.
        Agar prefix empty hai toh original document as-is return hota hai.
    """
    prefix = build_context_prefix(metadata)

    if not prefix:
        return document

    return prefix + document


# ── Contextual Retriever ──────────────────────────────────────────────────────


class ContextualRetriever:
    """
    R5 — Contextual Retriever.

    Dense retrieval karo, phir har retrieved chunk mein metadata-based
    context prefix inject karo before returning to LLM.

    Retrieval quality (ChromaDB scores) same rehti hai Dense (R1) jaisi —
    yeh technique retrieval improve nahi karti, LLM ke liye context improve karti hai.

    Isliye RAGAS mein:
      - Context Recall/Precision: Dense (R1) jaisa hoga
      - Faithfulness/Answer Relevancy: Improve hoga — LLM ko better context milega

    Usage:
        retriever = ContextualRetriever()
        result = retriever.retrieve(
            query="§ 121.135 manual requirements",
            query_embedding=[0.023, ...],
            source="FAA_CFR",
            strategy="recursive",
            n_results=5,
        )
        # result.documents mein context-injected texts honge
    """

    rag_variant: str = "contextual"

    def __init__(self, store: SkyLexVectorStore | None = None) -> None:
        self._store = store or SkyLexVectorStore()
        # Internally DenseRetriever use karta hai — same ChromaDB search.
        self._dense = DenseRetriever(store=self._store)

    @traceable(
        name="contextual_retrieve",
        project_name="skylex",
    )
    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        source: str,
        strategy: str,
        n_results: int = 5,
    ) -> RetrievalResult:
        """
        Dense retrieval karo, phir har chunk mein context inject karo.

        Process:
          Step 1: Dense retrieval — ChromaDB cosine similarity search.
          Step 2: Har retrieved chunk ke metadata se context prefix banao.
          Step 3: Prefix + original chunk text = context-enriched document.
          Step 4: Enriched documents return karo.

        Args:
            query           : Original query text.
            query_embedding : Pre-computed query vector (1536-dim).
            source          : FAA_CFR, FAA_AD, etc.
            strategy        : recursive, hierarchical, etc.
            n_results       : Kitne chunks chahiye.

        Returns:
            RetrievalResult jisme documents context-injected hain.
            extra mein injection stats hain — kitne chunks mein prefix add hua.
        """
        start_time = time.perf_counter()

        # Step 1: Dense retrieval — same as R1.
        dense_result = self._dense.retrieve(
            query=query,
            query_embedding=query_embedding,
            source=source,
            strategy=strategy,
            n_results=n_results,
        )

        if not dense_result.documents:
            return dense_result

        # Step 2 + 3: Har chunk mein context inject karo.
        # Metadata already dense retrieval mein fetch ho chuka hai.
        injected_docs: list[str] = []
        injection_count = 0

        for doc, meta in zip(dense_result.documents, dense_result.metadatas):
            prefix = build_context_prefix(meta)
            if prefix:
                injected_docs.append(prefix + doc)
                injection_count += 1
            else:
                # Metadata nahi hai — original text as-is rakho.
                injected_docs.append(doc)

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            "Contextual retrieve | source=%s | strategy=%s | "
            "chunks=%d | context_injected=%d | latency=%.1fms",
            source, strategy,
            len(dense_result.documents), injection_count, latency_ms,
        )

        return RetrievalResult(
            query=query,
            source=source,
            strategy=strategy,
            rag_variant=self.rag_variant,
            chunk_ids=dense_result.chunk_ids,
            documents=injected_docs,           # Context-enriched documents.
            metadatas=dense_result.metadatas,
            scores=dense_result.scores,
            latency_ms=round(latency_ms, 2),
            extra={
                "injection_count":     injection_count,
                "total_chunks":        len(dense_result.documents),
                "injection_rate":      round(
                    injection_count / len(dense_result.documents), 4
                ) if dense_result.documents else 0.0,
                "top_similarity":      dense_result.scores[0] if dense_result.scores else 0.0,
            },
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
        Multiple queries ke liye batch contextual retrieval.

        tqdm progress bar + MLflow metrics logging included.

        Args:
            queries          : List of query strings.
            query_embeddings : Pre-computed embeddings — queries ke saath 1:1 mapping.
            source           : FAA_CFR, FAA_AD, etc.
            strategy         : recursive, hierarchical, etc.
            n_results        : Top-k per query.
            mlflow_run_name  : MLflow nested run name.

        Returns:
            List of RetrievalResult with context-injected documents.
        """
        if len(queries) != len(query_embeddings):
            raise ValueError(
                f"queries ({len(queries)}) aur query_embeddings "
                f"({len(query_embeddings)}) ki length equal honi chahiye."
            )

        results: list[RetrievalResult] = []
        total_latency_ms = 0.0
        total_injected = 0

        # Cyan colour — contextual retriever ke liye distinct.
        with tqdm(
            total=len(queries),
            desc=f"Contextual Retrieve | {source} × {strategy}",
            unit="query",
            colour="cyan",
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
                total_injected += result.extra.get("injection_count", 0)

                pbar.update(1)
                pbar.set_postfix({
                    "latency":   f"{result.latency_ms:.1f}ms",
                    "injected":  result.extra.get("injection_count", 0),
                })

        if mlflow_run_name:
            self._log_batch_metrics(
                results=results,
                source=source,
                strategy=strategy,
                total_latency_ms=total_latency_ms,
                total_injected=total_injected,
                mlflow_run_name=mlflow_run_name,
            )

        logger.info(
            "Contextual batch complete | source=%s | strategy=%s | "
            "queries=%d | total_injected=%d | avg_latency=%.1fms",
            source, strategy, len(queries), total_injected,
            total_latency_ms / len(queries) if queries else 0,
        )

        return results

    def _log_batch_metrics(
        self,
        results: list[RetrievalResult],
        source: str,
        strategy: str,
        total_latency_ms: float,
        total_injected: int,
        mlflow_run_name: str,
    ) -> None:
        """
        MLflow mein contextual retrieval batch metrics log karo.

        Key metric: avg_injection_rate — kitne retrieved chunks mein
        context successfully inject hua. Low rate = metadata missing tha,
        high rate = metadata properly stored tha embedding time pe.
        """
        if not results:
            return

        avg_latency = total_latency_ms / len(results)
        avg_injection_rate = sum(
            r.extra.get("injection_rate", 0.0) for r in results
        ) / len(results)

        with mlflow.start_run(run_name=mlflow_run_name, nested=True):
            mlflow.log_params({
                "rag_variant":  self.rag_variant,
                "source":       source,
                "strategy":     strategy,
                "query_count":  len(results),
            })
            mlflow.log_metrics({
                "avg_latency_ms":      round(avg_latency, 2),
                "total_latency_ms":    round(total_latency_ms, 2),
                "total_injected":      total_injected,
                "avg_injection_rate":  round(avg_injection_rate, 4),
            })


# ── Convenience Function ──────────────────────────────────────────────────────


def retrieve_all_strategies(
    query: str,
    query_embedding: list[float],
    source: str,
    n_results: int = 5,
    store: SkyLexVectorStore | None = None,
) -> dict[str, RetrievalResult]:
    """
    Ek source ke saari candidate strategies pe contextual retrieval karo.

    RAGAS evaluation mein same query ko alag chunking strategies se
    compare karne ke liye use hoga — R5 variant ke liye.

    Args:
        query           : Query text.
        query_embedding : Pre-computed query vector.
        source          : FAA_CFR, FAA_AD, etc.
        n_results       : Top-k per strategy.
        store           : Optional pre-initialized store.

    Returns:
        Dict mapping strategy → RetrievalResult (context-injected).
    """
    retriever = ContextualRetriever(store=store)
    strategies = CANDIDATE_STRATEGIES.get(source, [])
    results: dict[str, RetrievalResult] = {}

    with tqdm(
        total=len(strategies),
        desc=f"Contextual | All Strategies | {source}",
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