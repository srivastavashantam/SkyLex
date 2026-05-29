"""
processing/retrieval/metadata_retriever.py

R7 — Metadata Filtered Retriever.

=============================================================================
THE CORE IDEA: PRECISION THROUGH FILTERING
=============================================================================

Standard dense/hybrid retrieval saare chunks mein search karta hai —
even if query clearly ek specific source ya document se related hai.

Problem example:
  Query: "What are alcohol testing requirements for pilots?"

  Without filtering — 5 sources mein search:
    FAA_CFR:  "§ 91.17 — Alcohol or drugs..." (relevant)
    DGCA_CAR: "CAR Section 7 — Alcohol testing..." (relevant)
    FAA_AC:   "Advisory on substance testing..." (partially relevant)
    SKYBRARY: "Pilot impairment article..." (marginally relevant)
    FAA_AD:   "Airworthiness directive..." (likely irrelevant)

  With source filter (DGCA_CAR):
    Only DGCA_CAR chunks → precise, relevant, no cross-source noise

=============================================================================
WHEN METADATA FILTERING IS MOST USEFUL
=============================================================================

1. SOURCE-SPECIFIC QUERIES:
   User explicitly wants info from one source.
   "Under DGCA regulations, what are the..." → filter source=DGCA_CAR
   "According to FAA advisory circulars..." → filter source=FAA_AC

2. DOCUMENT-SPECIFIC QUERIES:
   User wants info from a specific regulation/document.
   "What does § 121.135 say about..." → filter doc_id contains "121_135"

3. STRATEGY COMPARISON (RAGAS evaluation):
   Same query, same source, different chunking strategies.
   Evaluate which chunking strategy gives better retrieval.

4. AGENTIC PIPELINE (Phase 5):
   Query classifier identifies source → metadata filter applied automatically.
   User asks about Indian aviation → classifier routes to DGCA_CAR filter.

=============================================================================
CHROMADB FILTER SYNTAX — Quick Reference
=============================================================================

  Single field match:
    where={"source": "DGCA_CAR"}
    where={"strategy": "recursive"}
    where={"doc_id": "faa_cfr_part121"}

  Multiple conditions (AND):
    where={"$and": [{"source": "FAA_CFR"}, {"strategy": "semantic"}]}

  Multiple conditions (OR):
    where={"$or": [{"source": "FAA_CFR"}, {"source": "DGCA_CAR"}]}

  Note: ChromaDB metadata values must be str, int, float, or bool.
        Lists aur nested dicts filter mein support nahi hote.

=============================================================================
MONITORING
=============================================================================

  - tqdm: Batch retrieval progress
  - MLflow: Filter stats, latency, result counts per filter combination
  - LangSmith: @traceable — trace mein filter parameters visible honge
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mlflow
from langsmith import traceable
from tqdm import tqdm

from monitoring.logger import get_logger
from processing.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from processing.retrieval.hybrid_retriever import HybridRetriever
from processing.vector_store import CANDIDATE_STRATEGIES, SkyLexVectorStore

logger = get_logger(__name__)


# ── Filter Presets ────────────────────────────────────────────────────────────

# Common filter combinations — RAGAS evaluation aur agentic pipeline mein use honge.
# Yeh presets manually define kiye hain — most common query patterns ke liye.
# Agentic pipeline mein query classifier in presets mein se select karega.
FILTER_PRESETS: dict[str, dict[str, Any]] = {
    # Source-level filters — kaunse data source se answer chahiye
    "faa_cfr_only":   {"source": "FAA_CFR"},
    "faa_ad_only":    {"source": "FAA_AD"},
    "faa_ac_only":    {"source": "FAA_AC"},
    "dgca_only":      {"source": "DGCA_CAR"},
    "skybrary_only":  {"source": "SKYBRARY"},

    # Strategy-level filters — RAGAS evaluation ke liye useful
    "recursive_only":          {"strategy": "recursive"},
    "hierarchical_only":       {"strategy": "hierarchical"},
    "semantic_only":           {"strategy": "semantic"},
    "double_pass_only":        {"strategy": "double_pass"},
    "improved_semantic_only":  {"strategy": "improved_semantic"},

    # Combined filters — source + strategy specific
    "faa_cfr_recursive":       {"$and": [{"source": "FAA_CFR"},  {"strategy": "recursive"}]},
    "faa_cfr_semantic":        {"$and": [{"source": "FAA_CFR"},  {"strategy": "semantic"}]},
    "faa_ad_hierarchical":     {"$and": [{"source": "FAA_AD"},   {"strategy": "hierarchical"}]},
    "dgca_double_pass":        {"$and": [{"source": "DGCA_CAR"}, {"strategy": "double_pass"}]},
}


# ── Metadata Filtered Retriever ───────────────────────────────────────────────


class MetadataFilteredRetriever:
    """
    R7 — Metadata Filtered Retriever.

    Dense ya Hybrid retrieval karo lekin ChromaDB metadata filter apply karo
    taaki sirf relevant source/strategy/document ke chunks search hon.

    Internally DenseRetriever ya HybridRetriever use karta hai —
    filtering ChromaDB ke `where` parameter se hoti hai jo collection-level
    metadata pe exact match karta hai.

    Two retrieval modes:
      mode="dense"  — R7a: Filtered Dense (fast, semantic only)
      mode="hybrid" — R7b: Filtered Hybrid RRF (best quality with filter)

    Usage:
        retriever = MetadataFilteredRetriever(mode="hybrid")

        # Source-specific query
        result = retriever.retrieve(
            query="alcohol testing requirements for pilots",
            query_embedding=[0.023, ...],
            source="DGCA_CAR",
            strategy="double_pass",
            metadata_filter={"source": "DGCA_CAR"},
            n_results=5,
        )

        # Preset use karo
        result = retriever.retrieve_with_preset(
            query="...",
            query_embedding=[...],
            source="DGCA_CAR",
            strategy="double_pass",
            preset_name="dgca_only",
        )
    """

    def __init__(
        self,
        mode: str = "hybrid",
        store: SkyLexVectorStore | None = None,
    ) -> None:
        """
        Args:
            mode  : "dense" ya "hybrid" — kaunsa base retriever use karna hai.
                    "dense"  = faster, semantic only, good for most queries.
                    "hybrid" = slower, dense + BM25 + RRF, better for exact terms.
            store : SkyLexVectorStore instance — None dene par fresh banega.
        """
        if mode not in ("dense", "hybrid"):
            raise ValueError(f"mode must be 'dense' or 'hybrid', got '{mode}'")

        self._mode = mode
        self._store = store or SkyLexVectorStore()
        self.rag_variant = f"metadata_filtered_{mode}"

        # Base retriever initialize karo mode ke hisaab se.
        if mode == "dense":
            self._retriever: DenseRetriever | HybridRetriever = DenseRetriever(
                store=self._store
            )
        else:
            self._retriever = HybridRetriever(store=self._store)

    @traceable(
        name="metadata_filtered_retrieve",
        project_name="skylex",
    )
    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        source: str,
        strategy: str,
        metadata_filter: dict[str, Any] | None = None,
        n_results: int = 5,
    ) -> RetrievalResult:
        """
        Metadata filter ke saath retrieval karo.

        Agar metadata_filter None hai toh automatically source-level filter
        apply hota hai — sirf us source ke chunks search honge.
        Yeh default behavior ensure karta hai ki cross-source noise nahi aati.

        Args:
            query           : Original query text.
            query_embedding : Pre-computed query vector (1536-dim).
            source          : FAA_CFR, FAA_AD, etc.
            strategy        : recursive, hierarchical, etc.
            metadata_filter : ChromaDB where clause dict.
                              None dene par {"source": source} use hoga.
                              Custom filter dene par woh use hoga as-is.
            n_results       : Kitne chunks chahiye.

        Returns:
            RetrievalResult with filtered chunks.
            extra mein filter info hogi — kaunsa filter apply hua.
        """
        start_time = time.perf_counter()

        # Default filter: agar koi filter nahi diya toh source-level filter use karo.
        # Yeh ensure karta hai ki retrieval always source-aware hoti hai.
        effective_filter = metadata_filter or {"source": source}

        logger.debug(
            "Metadata filtered retrieve | source=%s | strategy=%s | "
            "filter=%s | mode=%s",
            source, strategy, effective_filter, self._mode,
        )

        # Base retriever call karo — filter pass karo.
        if self._mode == "dense":
            # DenseRetriever ka where parameter use karo.
            base_result = self._retriever.retrieve(  # type: ignore[union-attr]
                query=query,
                query_embedding=query_embedding,
                source=source,
                strategy=strategy,
                n_results=n_results,
                where=effective_filter,
            )
        else:
            # HybridRetriever ke liye filter dense part mein pass karo.
            # BM25 filtering ChromaDB ke bahar hoti hai — post-filter apply karenge.
            base_result = self._retriever.retrieve(  # type: ignore[union-attr]
                query=query,
                query_embedding=query_embedding,
                source=source,
                strategy=strategy,
                n_results=n_results,
            )
            # Hybrid result pe metadata post-filter apply karo.
            base_result = self._apply_post_filter(base_result, effective_filter)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # RetrievalResult update karo — rag_variant aur filter info add karo.
        return RetrievalResult(
            query=base_result.query,
            source=base_result.source,
            strategy=base_result.strategy,
            rag_variant=self.rag_variant,
            chunk_ids=base_result.chunk_ids,
            documents=base_result.documents,
            metadatas=base_result.metadatas,
            scores=base_result.scores,
            latency_ms=round(latency_ms, 2),
            extra={
                **base_result.extra,
                "applied_filter":   effective_filter,
                "base_rag_variant": base_result.rag_variant,
                "filter_mode":      self._mode,
                "results_after_filter": len(base_result.chunk_ids),
            },
        )

    def retrieve_with_preset(
        self,
        query: str,
        query_embedding: list[float],
        source: str,
        strategy: str,
        preset_name: str,
        n_results: int = 5,
    ) -> RetrievalResult:
        """
        Pre-defined filter preset use karke retrieval karo.

        FILTER_PRESETS dict mein defined presets use karo —
        common query patterns ke liye shortcut.

        Args:
            preset_name : FILTER_PRESETS mein se ek key.
                          Available presets: faa_cfr_only, dgca_only,
                          recursive_only, faa_cfr_semantic, etc.

        Returns:
            RetrievalResult with preset filter applied.

        Raises:
            ValueError: Agar preset_name valid nahi hai.
        """
        if preset_name not in FILTER_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(FILTER_PRESETS.keys())}"
            )

        return self.retrieve(
            query=query,
            query_embedding=query_embedding,
            source=source,
            strategy=strategy,
            metadata_filter=FILTER_PRESETS[preset_name],
            n_results=n_results,
        )

    def _apply_post_filter(
        self,
        result: RetrievalResult,
        metadata_filter: dict[str, Any],
    ) -> RetrievalResult:
        """
        Hybrid retrieval ke results pe metadata post-filter apply karo.

        Hybrid retrieval mein BM25 ChromaDB ke bahar kaam karta hai —
        isliye BM25 results pe ChromaDB filter apply nahi hota.
        Post-filter manually metadata check karta hai aur non-matching chunks hata deta hai.

        Simple equality filters handle karta hai — complex $and/$or filters
        ke liye sirf top-level keys check karta hai.

        Args:
            result          : Hybrid retrieval ka output.
            metadata_filter : ChromaDB-style filter dict.

        Returns:
            RetrievalResult with non-matching chunks removed.
        """
        # Complex filters ($and, $or) ke liye simple key matching use karo.
        # Production mein full ChromaDB filter parser implement karna hoga.
        simple_filters: dict[str, Any] = {}

        for key, value in metadata_filter.items():
            if not key.startswith("$"):
                # Simple key-value filter — direct equality check.
                simple_filters[key] = str(value)
            elif key == "$and":
                # $and filter — saari conditions extract karo.
                for condition in value:
                    for k, v in condition.items():
                        simple_filters[k] = str(v)

        if not simple_filters:
            return result

        # Filter apply karo — matching chunks rakho.
        filtered_ids, filtered_docs, filtered_meta, filtered_scores = [], [], [], []

        for chunk_id, doc, meta, score in zip(
            result.chunk_ids,
            result.documents,
            result.metadatas,
            result.scores,
        ):
            # Har filter condition check karo.
            match = all(
                str(meta.get(k, "")) == str(v)
                for k, v in simple_filters.items()
            )
            if match:
                filtered_ids.append(chunk_id)
                filtered_docs.append(doc)
                filtered_meta.append(meta)
                filtered_scores.append(score)

        return RetrievalResult(
            query=result.query,
            source=result.source,
            strategy=result.strategy,
            rag_variant=result.rag_variant,
            chunk_ids=filtered_ids,
            documents=filtered_docs,
            metadatas=filtered_meta,
            scores=filtered_scores,
            latency_ms=result.latency_ms,
            extra=result.extra,
        )

    def retrieve_batch(
        self,
        queries: list[str],
        query_embeddings: list[list[float]],
        source: str,
        strategy: str,
        metadata_filter: dict[str, Any] | None = None,
        n_results: int = 5,
        mlflow_run_name: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Multiple queries ke liye batch metadata filtered retrieval.

        tqdm progress bar + MLflow metrics included.
        MLflow mein filter info bhi log hoti hai — analysis mein
        useful hai ki kaunse filters best results de rahe hain.

        Args:
            queries          : List of query strings.
            query_embeddings : Pre-computed embeddings — queries ke saath 1:1.
            source           : FAA_CFR, FAA_AD, etc.
            strategy         : recursive, hierarchical, etc.
            metadata_filter  : ChromaDB where clause.
            n_results        : Top-k per query.
            mlflow_run_name  : MLflow nested run name.

        Returns:
            List of filtered RetrievalResults.
        """
        if len(queries) != len(query_embeddings):
            raise ValueError(
                f"queries ({len(queries)}) aur query_embeddings "
                f"({len(query_embeddings)}) ki length equal honi chahiye."
            )

        effective_filter = metadata_filter or {"source": source}
        results: list[RetrievalResult] = []
        total_latency_ms = 0.0

        # Red colour — metadata filtered retriever ke liye distinct.
        with tqdm(
            total=len(queries),
            desc=f"Metadata Filtered ({self._mode}) | {source} × {strategy}",
            unit="query",
            colour="red",
        ) as pbar:
            for query, embedding in zip(queries, query_embeddings):
                result = self.retrieve(
                    query=query,
                    query_embedding=embedding,
                    source=source,
                    strategy=strategy,
                    metadata_filter=effective_filter,
                    n_results=n_results,
                )
                results.append(result)
                total_latency_ms += result.latency_ms

                pbar.update(1)
                pbar.set_postfix({
                    "latency":  f"{result.latency_ms:.1f}ms",
                    "results":  result.result_count,
                    "filter":   str(effective_filter)[:20],
                })

        if mlflow_run_name:
            self._log_batch_metrics(
                results=results,
                source=source,
                strategy=strategy,
                effective_filter=effective_filter,
                total_latency_ms=total_latency_ms,
                mlflow_run_name=mlflow_run_name,
            )

        logger.info(
            "Metadata filtered batch complete | source=%s | strategy=%s | "
            "filter=%s | queries=%d | avg_latency=%.1fms",
            source, strategy, effective_filter, len(queries),
            total_latency_ms / len(queries) if queries else 0,
        )

        return results

    def _log_batch_metrics(
        self,
        results: list[RetrievalResult],
        source: str,
        strategy: str,
        effective_filter: dict[str, Any],
        total_latency_ms: float,
        mlflow_run_name: str,
    ) -> None:
        """
        MLflow mein metadata filtered retrieval batch metrics log karo.

        Key metric: avg_results_after_filter — filter ke baad kitne chunks
        average mein bach rahe hain. Agar yeh bahut kam hai toh filter
        too strict hai — useful chunks bhi filter ho rahe hain.
        """
        if not results:
            return

        avg_latency = total_latency_ms / len(results)
        avg_results = sum(r.result_count for r in results) / len(results)

        with mlflow.start_run(run_name=mlflow_run_name, nested=True):
            mlflow.log_params({
                "rag_variant":    self.rag_variant,
                "source":         source,
                "strategy":       strategy,
                "query_count":    len(results),
                "filter_mode":    self._mode,
                "applied_filter": str(effective_filter),
            })
            mlflow.log_metrics({
                "avg_latency_ms":          round(avg_latency, 2),
                "total_latency_ms":        round(total_latency_ms, 2),
                "avg_results_after_filter":round(avg_results, 2),
            })