"""
processing/retrieval/parent_doc_retriever.py

R6 — Parent Document Retriever (Small-to-Big Retrieval).

=============================================================================
THE CORE PROBLEM: PRECISION vs CONTEXT TRADEOFF
=============================================================================

Chunking mein ek fundamental tension hai:

  SMALL CHUNKS (e.g., recursive, 1000 chars):
    ✅ Better retrieval precision — query exact chunk se match karta hai
    ❌ LLM ko kam context milta hai — answer generate karte waqt surrounding
       information nahi hoti

  LARGE CHUNKS (e.g., semantic, 3000+ chars):
    ✅ LLM ko zyada context milta hai — better answer generation
    ❌ Retrieval precision kam — ek bade chunk mein relevant part dhundhna mushkil,
       noise zyada hoti hai

  Ideal hoga: Retrieval ke liye chhote chunks use karo (precise match),
              LLM ke liye bade chunks return karo (rich context).

  Yahi "Small-to-Big" ya "Parent Document Retrieval" hai.

=============================================================================
OUR APPROACH — Using Existing Multi-Strategy Collections
=============================================================================

Humari unique situation:
  Humare paas already SAME documents ki MULTIPLE chunking strategies hain.
  FAA_CFR mein:
    - skylex_faa_cfr_recursive   → 16,291 chhote chunks (avg 845 chars)
    - skylex_faa_cfr_semantic    → 14,028 bade chunks  (avg 923 chars)
    - skylex_faa_cfr_double_pass → 21,704 chhote chunks (avg 633 chars)

  Hum is existing structure ko "parent-child" relationship ki tarah use kar sakte hain:
    CHILD  (retrieval ke liye): recursive chunks — chhote, precise
    PARENT (LLM ke liye):       semantic chunks  — bade, contextual

  Dono mein same doc_id hota hai — isliye child → parent mapping possible hai.

MAPPING LOGIC:
  1. Child collection (recursive) mein query se chhote chunks retrieve karo
  2. Retrieved chunks ke doc_id aur approximate position se parent chunks identify karo
  3. Parent collection (semantic) se corresponding large chunks fetch karo
  4. Large chunks LLM ko return karo

=============================================================================
CHILD → PARENT STRATEGY MAPPING (Phase 2 results se)
=============================================================================

  Source      | Child Strategy | Parent Strategy | Reasoning
  ------------|----------------|-----------------|---------------------------
  FAA_CFR     | recursive      | semantic        | Semantic = largest, most contextual
  FAA_AD      | hybrid         | hierarchical    | Hierarchical = full AD document
  FAA_AC      | recursive      | improved_semantic| ImprovedSemantic = better boundaries
  DGCA_CAR    | recursive      | improved_semantic| ImprovedSemantic = semantic boundaries
  SKYBRARY    | recursive      | improved_semantic| ImprovedSemantic = topic-aware

=============================================================================
MONITORING
=============================================================================

  - tqdm: Batch retrieval progress
  - MLflow: Child/parent latency breakdown, expansion ratio (parent size / child size)
  - LangSmith: @traceable — full trace showing child→parent expansion
"""

from __future__ import annotations

import time
from typing import Any

import mlflow
from langsmith import traceable
from tqdm import tqdm

from monitoring.logger import get_logger
from processing.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from processing.vector_store import SkyLexVectorStore

logger = get_logger(__name__)


# ── Child → Parent Strategy Mapping ──────────────────────────────────────────

# Har source ke liye: child strategy (retrieval ke liye) → parent strategy (LLM ke liye).
# Child = chhote precise chunks, Parent = bade contextual chunks.
# Yeh mapping Phase 2 chunking experiment results par based hai.
PARENT_CHILD_MAPPING: dict[str, dict[str, str]] = {
    "FAA_CFR": {
        "child":  "recursive",    # Avg 845 chars — precise retrieval
        "parent": "semantic",     # Avg 923 chars — richer context (ImprovedSemantic)
    },
    "FAA_AD": {
        "child":  "hybrid",       # Avg 829 chars — hybrid boundary detection
        "parent": "hierarchical", # Full AD document — 1 doc = 1 chunk (perfect context)
    },
    "FAA_AC": {
        "child":  "recursive",    # Avg 827 chars — precise splitting
        "parent": "improved_semantic",  # Avg 672 chars but semantic boundaries
    },
    "DGCA_CAR": {
        "child":  "recursive",    # Avg 781 chars — precise splitting
        "parent": "improved_semantic",  # Avg 971 chars — larger semantic chunks
    },
    "SKYBRARY": {
        "child":  "recursive",    # Avg 761 chars — precise splitting
        "parent": "improved_semantic",  # Avg 735 chars — topic-aware boundaries
    },
}


# ── Parent Document Retriever ─────────────────────────────────────────────────


class ParentDocumentRetriever:
    """
    R6 — Parent Document Retriever (Small-to-Big).

    Retrieval ke liye child (small) chunks use karta hai,
    LLM ke liye parent (large) chunks return karta hai.

    Yeh combination precision aur context dono achieve karta hai —
    jo single-strategy retrieval mein tradeoff hota hai.

    HOW doc_id MAPPING WORKS:
      Dono child aur parent collections mein har chunk ke metadata mein
      "doc_id" field hoti hai. Yeh field original source document ka
      unique identifier hai — chunking strategy se independent.

      Example:
        Child chunk:  {doc_id: "faa_cfr_14_cfr_121_135", chunk_index: 3, ...}
        Parent chunk: {doc_id: "faa_cfr_14_cfr_121_135", chunk_index: 1, ...}

      Same doc_id = same source document se aaye hain.
      Ek child ke doc_id se uske saare parent chunks find kar sakte hain.

    Usage:
        retriever = ParentDocumentRetriever()
        result = retriever.retrieve(
            query="§ 121.135 manual table of contents",
            query_embedding=[0.023, ...],
            source="FAA_CFR",
            n_results=5,
        )
        # result.documents mein parent (large) chunks honge
        # result.extra mein child chunks ka info hoga comparison ke liye
    """

    rag_variant: str = "parent_doc"

    def __init__(self, store: SkyLexVectorStore | None = None) -> None:
        """
        Args:
            store: SkyLexVectorStore instance.
                   None dene par fresh instance banega.
                   Testing ke liye inject karna useful hai.
        """
        self._store = store or SkyLexVectorStore()
        # Child retrieval ke liye DenseRetriever use karte hain.
        # BM25 ya Hybrid bhi use kar sakte the lekin Dense sufficient hai
        # kyunki parent expansion se context already improve ho raha hai.
        self._dense = DenseRetriever(store=self._store)

    @traceable(
        name="parent_doc_retrieve",
        project_name="skylex",
    )
    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        source: str,
        n_results: int = 5,
    ) -> RetrievalResult:
        """
        Child chunks se retrieve karo, phir corresponding parent chunks return karo.

        DETAILED PROCESS:
          Step 1 — Child Retrieval:
            Child strategy (e.g., recursive) ki collection mein dense search karo.
            n_results * 2 child chunks fetch karo — zyada candidates = better parent coverage.
            Kyun 2x? Kyunki multiple child chunks ek hi parent se aa sakte hain,
            isliye zyada child chunks lene se unique parents zyada milte hain.

          Step 2 — doc_id Extraction:
            Retrieved child chunks ke metadata se doc_id list nikalo.
            Duplicate doc_ids deduplicate karo — ek parent ek baar hi chahiye.

          Step 3 — Parent Chunk Fetch:
            Parent strategy (e.g., semantic) ki collection mein un doc_ids ke
            chunks fetch karo using ChromaDB metadata filtering.
            where={"doc_id": doc_id} filter use karta hai.

          Step 4 — Assemble Final Result:
            Parent chunks ka text aur metadata return karo.
            extra mein child retrieval stats preserve karo analysis ke liye.

        Args:
            query           : Original query text.
            query_embedding : Pre-computed query vector (1536-dim).
            source          : FAA_CFR, FAA_AD, etc.
                              Parent-child mapping is source se decide hogi.
            n_results       : Final mein kitne parent chunks chahiye.

        Returns:
            RetrievalResult with parent (large) chunks as documents.
            extra mein child retrieval info hai comparison ke liye.
        """
        start_time = time.perf_counter()

        # Source ke liye mapping check karo.
        if source not in PARENT_CHILD_MAPPING:
            logger.warning(
                "No parent-child mapping found for source=%s — "
                "falling back to dense retrieval with recursive strategy",
                source,
            )
            return self._dense.retrieve(
                query=query,
                query_embedding=query_embedding,
                source=source,
                strategy="recursive",
                n_results=n_results,
            )

        mapping = PARENT_CHILD_MAPPING[source]
        child_strategy = mapping["child"]
        parent_strategy = mapping["parent"]

        # ── Step 1: Child Retrieval ───────────────────────────────────────────
        # n_results * 2 child chunks fetch karo — zyada candidates better hai
        # kyunki multiple child chunks ek hi parent se aa sakte hain.
        child_fetch_n = n_results * 2

        child_result = self._dense.retrieve(
            query=query,
            query_embedding=query_embedding,
            source=source,
            strategy=child_strategy,
            n_results=child_fetch_n,
        )

        child_retrieval_latency = time.perf_counter() - start_time

        if not child_result.chunk_ids:
            logger.warning(
                "Child retrieval returned no results | source=%s | strategy=%s",
                source, child_strategy,
            )
            return child_result

        # ── Step 2: doc_id Extraction ─────────────────────────────────────────
        # Child chunks ke metadata se doc_ids nikalo.
        # Duplicate doc_ids deduplicate karo — set use karo aur order preserve karo.
        seen_doc_ids: set[str] = set()
        ordered_doc_ids: list[str] = []

        for meta in child_result.metadatas:
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                ordered_doc_ids.append(doc_id)

        if not ordered_doc_ids:
            logger.warning(
                "No doc_ids found in child chunk metadata | source=%s",
                source,
            )
            return child_result

        # ── Step 3: Parent Chunk Fetch ────────────────────────────────────────
        # Parent strategy collection mein un doc_ids ke chunks fetch karo.
        parent_chunks, parent_meta = self._fetch_parent_chunks(
            doc_ids=ordered_doc_ids[:n_results],  # Top n_results unique doc_ids
            source=source,
            parent_strategy=parent_strategy,
            max_chunks_per_doc=2,  # Ek doc se max 2 parent chunks — noise avoid
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        # ── Step 4: Assemble Result ───────────────────────────────────────────
        # Parent chunk IDs — metadata se extract karo.
        parent_chunk_ids = [
            m.get("chunk_id", f"parent_{i}") for i, m in enumerate(parent_meta)
        ]

        # Scores: parent chunks ke liye child scores proxy use karte hain.
        # Direct parent scores nahi hain kyunki parent pe query nahi chali.
        # Child ke top score ko parent ke liye bhi use karte hain.
        proxy_scores = child_result.scores[:len(parent_chunks)]
        # Agar parent chunks zyada hain toh remaining ke liye last child score use karo.
        while len(proxy_scores) < len(parent_chunks):
            proxy_scores.append(proxy_scores[-1] if proxy_scores else 0.0)

        logger.debug(
            "Parent doc retrieve | source=%s | child_strategy=%s | "
            "parent_strategy=%s | child_chunks=%d | unique_docs=%d | "
            "parent_chunks=%d | latency=%.1fms",
            source, child_strategy, parent_strategy,
            len(child_result.chunk_ids), len(ordered_doc_ids),
            len(parent_chunks), total_latency_ms,
        )

        return RetrievalResult(
            query=query,
            source=source,
            strategy=parent_strategy,   # Parent strategy batao — LLM ko context milega
            rag_variant=self.rag_variant,
            chunk_ids=parent_chunk_ids,
            documents=parent_chunks,
            metadatas=parent_meta,
            scores=proxy_scores,
            latency_ms=round(total_latency_ms, 2),
            extra={
                # Child retrieval stats — analysis ke liye useful
                "child_strategy":         child_strategy,
                "parent_strategy":        parent_strategy,
                "child_chunks_retrieved": len(child_result.chunk_ids),
                "unique_doc_ids":         len(ordered_doc_ids),
                "parent_chunks_returned": len(parent_chunks),
                "child_retrieval_ms":     round(child_retrieval_latency * 1000, 2),
                # Expansion ratio: parent chunk avg size / child chunk avg size
                # High ratio = zyada context expansion hua
                "expansion_ratio":        round(
                    sum(len(d) for d in parent_chunks) /
                    max(sum(len(d) for d in child_result.documents), 1),
                    2,
                ),
            },
        )

    def _fetch_parent_chunks(
        self,
        doc_ids: list[str],
        source: str,
        parent_strategy: str,
        max_chunks_per_doc: int = 2,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Parent strategy collection mein given doc_ids ke chunks fetch karo.

        WHY max_chunks_per_doc = 2:
          Ek bade document ke saare parent chunks return karna noise create karta hai.
          Ek doc se max 2 chunks rakho — enough context milta hai bina overwhelming kiye.
          FAA AC documents mein 100+ parent chunks ho sakte hain ek doc mein —
          sab return karna LLM ke liye counterproductive hoga.

        Args:
            doc_ids           : Child retrieval se mile unique document IDs.
            source            : FAA_CFR, FAA_AD, etc.
            parent_strategy   : Parent collection ki chunking strategy.
            max_chunks_per_doc: Ek document se maximum kitne parent chunks return karo.

        Returns:
            Tuple of (documents, metadatas) — parent chunks.
        """
        parent_collection = self._store._collections.get((source, parent_strategy))

        if parent_collection is None:
            logger.warning(
                "Parent collection not found | source=%s | strategy=%s",
                source, parent_strategy,
            )
            return [], []

        all_docs: list[str] = []
        all_meta: list[dict[str, Any]] = []

        for doc_id in doc_ids:
            try:
                # ChromaDB metadata filter: sirf is doc_id ke chunks fetch karo.
                # where clause exact string match karta hai.
                fetch_result = parent_collection.get(
                    where={"doc_id": doc_id},
                    include=["documents", "metadatas"],  # type: ignore[arg-type]
                    limit=max_chunks_per_doc,
                )

                docs: list[str] = fetch_result.get("documents") or []
                meta: list[dict[str, Any]] = fetch_result.get("metadatas") or [] # type: ignore[assignment]


                # chunk_index se sort karo — document order maintain karo.
                # Agar chunk_index nahi hai toh as-is order rakho.
                paired = list(zip(docs, meta))
                paired.sort(
                    key=lambda x: int(x[1].get("chunk_index", 0))
                    if str(x[1].get("chunk_index", "0")).isdigit()
                    else 0
                )

                for doc, m in paired[:max_chunks_per_doc]:
                    all_docs.append(doc)
                    all_meta.append(m)

            except Exception as e:
                logger.warning(
                    "Failed to fetch parent chunks for doc_id=%s | error=%s",
                    doc_id, e,
                )
                continue

        return all_docs, all_meta

    def retrieve_batch(
        self,
        queries: list[str],
        query_embeddings: list[list[float]],
        source: str,
        n_results: int = 5,
        mlflow_run_name: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Multiple queries ke liye batch parent document retrieval.

        tqdm progress bar dikhata hai — white colour parent doc ke liye.
        MLflow mein aggregate metrics log hote hain including expansion ratio
        jo batata hai kitna context expansion hua child se parent tak.

        Args:
            queries          : List of query strings.
            query_embeddings : Pre-computed embeddings — queries ke saath 1:1 mapping.
            source           : FAA_CFR, FAA_AD, etc.
            n_results        : Final top-n parent chunks per query.
            mlflow_run_name  : MLflow nested run name — None matlab log nahi karo.

        Returns:
            List of RetrievalResult with parent chunks as documents.
        """
        if len(queries) != len(query_embeddings):
            raise ValueError(
                f"queries ({len(queries)}) aur query_embeddings "
                f"({len(query_embeddings)}) ki length equal honi chahiye."
            )

        results: list[RetrievalResult] = []
        total_latency_ms = 0.0
        total_expansion = 0.0

        # White colour — parent doc retriever ke liye distinct from others.
        with tqdm(
            total=len(queries),
            desc=f"Parent Doc Retrieve | {source}",
            unit="query",
            colour="white",
        ) as pbar:
            for query, embedding in zip(queries, query_embeddings):
                result = self.retrieve(
                    query=query,
                    query_embedding=embedding,
                    source=source,
                    n_results=n_results,
                )
                results.append(result)
                total_latency_ms += result.latency_ms
                total_expansion += result.extra.get("expansion_ratio", 1.0)

                pbar.update(1)
                pbar.set_postfix({
                    "latency":    f"{result.latency_ms:.1f}ms",
                    "expansion":  f"{result.extra.get('expansion_ratio', 1.0):.2f}x",
                    "parents":    result.extra.get("parent_chunks_returned", 0),
                })

        if mlflow_run_name:
            self._log_batch_metrics(
                results=results,
                source=source,
                total_latency_ms=total_latency_ms,
                avg_expansion=total_expansion / len(results) if results else 1.0,
                mlflow_run_name=mlflow_run_name,
            )

        logger.info(
            "Parent doc batch complete | source=%s | queries=%d | "
            "avg_expansion=%.2fx | avg_latency=%.1fms",
            source, len(queries),
            total_expansion / len(results) if results else 1.0,
            total_latency_ms / len(queries) if queries else 0,
        )

        return results

    def _log_batch_metrics(
        self,
        results: list[RetrievalResult],
        source: str,
        total_latency_ms: float,
        avg_expansion: float,
        mlflow_run_name: str,
    ) -> None:
        """
        MLflow mein parent document retrieval batch metrics log karo.

        Key metric: avg_expansion_ratio — parent chunk kitna bada hai child se.
        Example: ratio=1.5 means parent chunks average mein 50% larger hain child se.
        High ratio = zyada context expansion = potentially better LLM answers.
        """
        if not results:
            return

        avg_latency = total_latency_ms / len(results)
        avg_parent_count = sum(
            r.extra.get("parent_chunks_returned", 0) for r in results
        ) / len(results)

        with mlflow.start_run(run_name=mlflow_run_name, nested=True):
            mlflow.log_params({
                "rag_variant":  self.rag_variant,
                "source":       source,
                "query_count":  len(results),
                "child_strategy":  PARENT_CHILD_MAPPING.get(source, {}).get("child", ""),
                "parent_strategy": PARENT_CHILD_MAPPING.get(source, {}).get("parent", ""),
            })
            mlflow.log_metrics({
                "avg_latency_ms":     round(avg_latency, 2),
                "total_latency_ms":   round(total_latency_ms, 2),
                "avg_expansion_ratio":round(avg_expansion, 4),
                "avg_parent_count":   round(avg_parent_count, 2),
            })