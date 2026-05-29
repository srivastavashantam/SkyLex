"""
processing/retrieval/retrieval_factory.py

Retrieval Factory — Unified interface for all 7 RAG variants.

=============================================================================
WHY THIS FILE EXISTS
=============================================================================

SkyLex mein 7 alag retriever classes hain — har ek ka alag interface hai,
alag initialization requirements hain, aur alag calling conventions hain.

RAGAS evaluation mein 7 variants × 15 collections × 10 queries = 1,050
evaluations chalani hain. Bina factory ke:
  - Har evaluation loop mein 50+ lines ka if/elif boilerplate
  - SkyLexVectorStore 1,050 baar initialize hota (~2 sec × 1,050 = 35 min waste)
  - CrossEncoderReranker model 1,050 baar load hota (~3 sec × 1,050 = 52 min waste)
  - Caller ko har retriever ka different function signature yaad rakhna padta

Factory yeh sab solve karta hai:
  - Ek uniform retrieve() function — variant name pass karo, result lo
  - Store aur reranker ek baar initialize, poore session mein reuse
  - Built-in validation — galat variant pe clear error message
  - MLflow aur LangSmith tracking automatically included

=============================================================================
SUPPORTED RAG VARIANTS
=============================================================================

  "dense"              → R1: ChromaDB cosine similarity search only
  "bm25"               → R2: BM25 keyword search only
  "hybrid_rrf"         → R3: Dense + BM25 + Reciprocal Rank Fusion
  "hybrid_reranked"    → R4: R3 + Cross-Encoder reranking (highest quality)
  "contextual"         → R5: Dense + metadata context prefix injection
  "parent_doc"         → R6: Small chunk retrieval → large parent chunk return
  "metadata_filtered"  → R7: Dense/Hybrid with ChromaDB metadata filtering

=============================================================================
USAGE
=============================================================================

  # Initialize once — store aur reranker cache ho jaate hain
  factory = RetrievalFactory()

  # Koi bhi variant — same interface
  result = factory.retrieve(
      variant="hybrid_reranked",
      query="§ 121.135 manual requirements",
      query_embedding=[0.023, -0.156, ...],
      source="FAA_CFR",
      strategy="recursive",
      n_results=5,
  )

  # Batch retrieval — tqdm + MLflow included
  results = factory.retrieve_batch(
      variant="hybrid_rrf",
      queries=["query1", "query2", ...],
      query_embeddings=[[...], [...]],
      source="FAA_CFR",
      strategy="recursive",
  )

  # Available variants check karo
  print(factory.available_variants)

=============================================================================
MONITORING
=============================================================================

  - tqdm: Batch retrieval progress (variant-specific colours)
  - MLflow: Per-variant latency, scores, result counts
  - LangSmith: @traceable on all underlying retrievers
"""

from __future__ import annotations

import time
from typing import Any

import mlflow
from tqdm import tqdm

from monitoring.logger import get_logger
from processing.retrieval.contextual_retriever import ContextualRetriever
from processing.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from processing.retrieval.hybrid_retriever import HybridRetriever
from processing.retrieval.metadata_retriever import MetadataFilteredRetriever
from processing.retrieval.parent_doc_retriever import ParentDocumentRetriever
from processing.retrieval.reranker import CrossEncoderReranker
from processing.retrieval.sparse_retriever import SparseRetriever
from processing.vector_store import CANDIDATE_STRATEGIES, SkyLexVectorStore

logger = get_logger(__name__)

# Saare supported RAG variant names — validation aur iteration ke liye.
# RAGAS experiment runner yeh list iterate karega.
ALL_VARIANTS: list[str] = [
    "dense",
    "bm25",
    "hybrid_rrf",
    "hybrid_reranked",
    "contextual",
    "parent_doc",
    "metadata_filtered",
]

# Variants jo query_embedding nahi lete — sirf BM25 based hain.
# Factory inhe handle karta hai — embedding pass nahi karta inhe.
_NO_EMBEDDING_VARIANTS: frozenset[str] = frozenset({"bm25"})

# Variants jo strategy parameter nahi lete — apni internal mapping use karte hain.
# Parent doc retriever ka apna child→parent mapping hai.
_NO_STRATEGY_VARIANTS: frozenset[str] = frozenset({"parent_doc"})


# ── Retrieval Factory ─────────────────────────────────────────────────────────


class RetrievalFactory:
    """
    Unified factory for all 7 SkyLex RAG variants.

    Lazy initialization pattern use karta hai — retrievers tabhi initialize
    hote hain jab pehli baar use hote hain. Yeh startup time bachata hai
    agar sirf kuch variants use karne hain.

    Thread safety:
      Yeh class thread-safe nahi hai — single-threaded RAGAS evaluation
      ke liye designed hai. Parallel evaluation ke liye alag factory
      instances banao.
    """

    def __init__(
        self,
        store: SkyLexVectorStore | None = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        metadata_filter_mode: str = "hybrid",
    ) -> None:
        """
        Args:
            store                : SkyLexVectorStore instance.
                                   None dene par fresh instance banega.
                                   Ek store inject karne se saare retrievers
                                   same store share karte hain — memory efficient.
            reranker_model       : HuggingFace cross-encoder model name.
                                   Default: ms-marco-MiniLM-L-6-v2
            metadata_filter_mode : "dense" ya "hybrid" — R7 ke liye base retriever.
        """
        # Store ek baar initialize karo — saare retrievers share karenge.
        # Yeh 15 ChromaDB collections load karta hai — ~2 seconds.
        self._store = store or SkyLexVectorStore()
        self._reranker_model = reranker_model
        self._metadata_filter_mode = metadata_filter_mode

        # Lazy-initialized retrievers — pehli use pe initialize honge.
        # Underscore prefix = private, factory ke bahar directly access mat karo.
        self._dense: DenseRetriever | None = None
        self._sparse: SparseRetriever | None = None
        self._hybrid: HybridRetriever | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._contextual: ContextualRetriever | None = None
        self._parent_doc: ParentDocumentRetriever | None = None
        self._metadata_filtered: MetadataFilteredRetriever | None = None

        logger.info(
            "RetrievalFactory initialized | variants=%d | store=ready",
            len(ALL_VARIANTS),
        )

    # ── Lazy Getters — Retrievers Tab Initialize Hote Hain Jab Pehli Baar Use Hon ──

    @property
    def dense(self) -> DenseRetriever:
        """Dense retriever — pehli call pe initialize hoga."""
        if self._dense is None:
            logger.debug("Initializing DenseRetriever...")
            self._dense = DenseRetriever(store=self._store)
        return self._dense

    @property
    def sparse(self) -> SparseRetriever:
        """BM25 sparse retriever — BM25 indexes disk se load honge."""
        if self._sparse is None:
            logger.debug("Initializing SparseRetriever...")
            self._sparse = SparseRetriever()
        return self._sparse

    @property
    def hybrid(self) -> HybridRetriever:
        """Hybrid RRF retriever — dense aur sparse dono use karta hai."""
        if self._hybrid is None:
            logger.debug("Initializing HybridRetriever...")
            self._hybrid = HybridRetriever(store=self._store)
        return self._hybrid

    @property
    def reranker(self) -> CrossEncoderReranker:
        """
        Cross-encoder reranker — pehli call pe model download/load hoga (~80MB).
        Subsequent calls fast honge — model memory mein cached rahega.
        """
        if self._reranker is None:
            logger.info(
                "Initializing CrossEncoderReranker — model load ho raha hai: %s",
                self._reranker_model,
            )
            self._reranker = CrossEncoderReranker(model_name=self._reranker_model)
        return self._reranker

    @property
    def contextual(self) -> ContextualRetriever:
        """Contextual retriever — metadata-based context injection ke saath dense."""
        if self._contextual is None:
            logger.debug("Initializing ContextualRetriever...")
            self._contextual = ContextualRetriever(store=self._store)
        return self._contextual

    @property
    def parent_doc(self) -> ParentDocumentRetriever:
        """Parent document retriever — small chunk retrieve, large chunk return."""
        if self._parent_doc is None:
            logger.debug("Initializing ParentDocumentRetriever...")
            self._parent_doc = ParentDocumentRetriever(store=self._store)
        return self._parent_doc

    @property
    def metadata_filtered(self) -> MetadataFilteredRetriever:
        """Metadata filtered retriever — source/strategy/doc level filtering."""
        if self._metadata_filtered is None:
            logger.debug(
                "Initializing MetadataFilteredRetriever | mode=%s",
                self._metadata_filter_mode,
            )
            self._metadata_filtered = MetadataFilteredRetriever(
                mode=self._metadata_filter_mode,
                store=self._store,
            )
        return self._metadata_filtered

    # ── Available Variants ────────────────────────────────────────────────────

    @property
    def available_variants(self) -> list[str]:
        """Saare supported RAG variant names — iteration aur validation ke liye."""
        return ALL_VARIANTS.copy()

    # ── Core Retrieve Method ──────────────────────────────────────────────────

    def retrieve(
        self,
        variant: str,
        query: str,
        query_embedding: list[float],
        source: str,
        strategy: str,
        n_results: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        rerank_top_n: int | None = None,
    ) -> RetrievalResult:
        """
        Koi bhi RAG variant se retrieval karo — uniform interface.

        Internally correct retriever call karta hai based on variant name.
        Caller ko individual retriever classes ke signatures yaad nahi rakhne.

        Args:
            variant          : RAG variant name — ALL_VARIANTS mein se ek.
                               "dense", "bm25", "hybrid_rrf", "hybrid_reranked",
                               "contextual", "parent_doc", "metadata_filtered"
            query            : Original query text.
            query_embedding  : Pre-computed query vector (1536-dim).
                               BM25 variant ke liye ignore hoga.
            source           : FAA_CFR, FAA_AD, etc.
            strategy         : recursive, hierarchical, etc.
                               parent_doc variant ke liye ignore hoga.
            n_results        : Kitne chunks chahiye (default: 5).
            metadata_filter  : metadata_filtered variant ke liye ChromaDB filter.
                               None dene par {"source": source} use hoga.
            rerank_top_n     : hybrid_reranked variant mein final top-n.
                               None dene par n_results use hoga.

        Returns:
            RetrievalResult — saare variants ke liye same structure.

        Raises:
            ValueError: Agar variant name valid nahi hai.
        """
        if variant not in ALL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available variants: {ALL_VARIANTS}"
            )

        logger.debug(
            "Factory retrieve | variant=%s | source=%s | strategy=%s | n=%d",
            variant, source, strategy, n_results,
        )

        # ── R1: Dense Only ────────────────────────────────────────────────────
        if variant == "dense":
            return self.dense.retrieve(
                query=query,
                query_embedding=query_embedding,
                source=source,
                strategy=strategy,
                n_results=n_results,
            )

        # ── R2: BM25 Sparse Only ──────────────────────────────────────────────
        elif variant == "bm25":
            # BM25 ke liye original texts chahiye — store pass karo.
            return self.sparse.retrieve_with_original_texts(
                query=query,
                source=source,
                strategy=strategy,
                n_results=n_results,
                store=self._store,
            )

        # ── R3: Hybrid RRF ────────────────────────────────────────────────────
        elif variant == "hybrid_rrf":
            return self.hybrid.retrieve(
                query=query,
                query_embedding=query_embedding,
                source=source,
                strategy=strategy,
                n_results=n_results,
            )

        # ── R4: Hybrid + Cross-Encoder Reranking ──────────────────────────────
        elif variant == "hybrid_reranked":
            # Step 1: Hybrid RRF se zyada candidates fetch karo reranking ke liye.
            # n_results * 5 = enough candidates for reranker to work with.
            # Reranker ke paas zyada options = better final ranking.
            candidate_n = n_results * 5

            hybrid_result = self.hybrid.retrieve(
                query=query,
                query_embedding=query_embedding,
                source=source,
                strategy=strategy,
                n_results=candidate_n,
            )

            # Step 2: Cross-encoder se rerank karo — top_n final chunks return.
            final_top_n = rerank_top_n or n_results
            return self.reranker.rerank(
                query=query,
                retrieval_result=hybrid_result,
                top_n=final_top_n,
            )

        # ── R5: Contextual Retrieval ──────────────────────────────────────────
        elif variant == "contextual":
            return self.contextual.retrieve(
                query=query,
                query_embedding=query_embedding,
                source=source,
                strategy=strategy,
                n_results=n_results,
            )

        # ── R6: Parent Document Retrieval ─────────────────────────────────────
        elif variant == "parent_doc":
            # Parent doc retriever apna internal child→parent mapping use karta hai.
            # Strategy parameter pass nahi karte — retriever khud decide karta hai.
            return self.parent_doc.retrieve(
                query=query,
                query_embedding=query_embedding,
                source=source,
                n_results=n_results,
            )

        # ── R7: Metadata Filtered ─────────────────────────────────────────────
        elif variant == "metadata_filtered":
            return self.metadata_filtered.retrieve(
                query=query,
                query_embedding=query_embedding,
                source=source,
                strategy=strategy,
                metadata_filter=metadata_filter,
                n_results=n_results,
            )

        # Yahan kabhi nahi pohonchna chahiye — upar validation already hai.
        raise RuntimeError(f"Unhandled variant: {variant}")

    # ── Batch Retrieve ────────────────────────────────────────────────────────

    def retrieve_batch(
        self,
        variant: str,
        queries: list[str],
        query_embeddings: list[list[float]],
        source: str,
        strategy: str,
        n_results: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        mlflow_run_name: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Multiple queries ke liye batch retrieval — koi bhi variant.

        tqdm progress bar included — variant name aur source/strategy dikhata hai.
        MLflow mein aggregate metrics log hote hain.

        Args:
            variant          : RAG variant name.
            queries          : List of query strings.
            query_embeddings : Pre-computed embeddings — queries ke saath 1:1.
            source           : FAA_CFR, FAA_AD, etc.
            strategy         : recursive, hierarchical, etc.
            n_results        : Top-k per query.
            metadata_filter  : metadata_filtered variant ke liye filter.
            mlflow_run_name  : MLflow nested run name — None matlab log nahi.

        Returns:
            List of RetrievalResults — queries ke saath 1:1 mapping.
        """
        if variant not in ALL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {ALL_VARIANTS}"
            )

        if len(queries) != len(query_embeddings):
            raise ValueError(
                f"queries ({len(queries)}) aur query_embeddings "
                f"({len(query_embeddings)}) ki length equal honi chahiye."
            )

        results: list[RetrievalResult] = []
        total_latency_ms = 0.0
        start_time = time.perf_counter()

        with tqdm(
            total=len(queries),
            desc=f"{variant} | {source} × {strategy}",
            unit="query",
            colour="blue",
        ) as pbar:
            for query, embedding in zip(queries, query_embeddings):
                result = self.retrieve(
                    variant=variant,
                    query=query,
                    query_embedding=embedding,
                    source=source,
                    strategy=strategy,
                    n_results=n_results,
                    metadata_filter=metadata_filter,
                )
                results.append(result)
                total_latency_ms += result.latency_ms

                pbar.update(1)
                pbar.set_postfix({
                    "latency": f"{result.latency_ms:.1f}ms",
                    "results": result.result_count,
                })

        total_time = time.perf_counter() - start_time

        if mlflow_run_name:
            self._log_batch_metrics(
                variant=variant,
                results=results,
                source=source,
                strategy=strategy,
                total_latency_ms=total_latency_ms,
                wall_time_s=total_time,
                mlflow_run_name=mlflow_run_name,
            )

        logger.info(
            "Factory batch complete | variant=%s | source=%s | strategy=%s | "
            "queries=%d | avg_latency=%.1fms | wall_time=%.1fs",
            variant, source, strategy, len(queries),
            total_latency_ms / len(queries) if queries else 0,
            total_time,
        )

        return results

    def _log_batch_metrics(
        self,
        variant: str,
        results: list[RetrievalResult],
        source: str,
        strategy: str,
        total_latency_ms: float,
        wall_time_s: float,
        mlflow_run_name: str,
    ) -> None:
        """
        MLflow mein factory batch metrics log karo.

        Wall time vs total_latency_ms difference batata hai ki overhead
        kitna hai — initialization, tqdm, etc. ka combined time.
        """
        if not results:
            return

        avg_latency = total_latency_ms / len(results)
        avg_results = sum(r.result_count for r in results) / len(results)

        with mlflow.start_run(run_name=mlflow_run_name, nested=True):
            mlflow.log_params({
                "rag_variant":  variant,
                "source":       source,
                "strategy":     strategy,
                "query_count":  len(results),
            })
            mlflow.log_metrics({
                "avg_latency_ms":  round(avg_latency, 2),
                "total_latency_ms":round(total_latency_ms, 2),
                "wall_time_s":     round(wall_time_s, 2),
                "avg_result_count":round(avg_results, 2),
            })

    # ── Full Experiment Run ───────────────────────────────────────────────────

    def run_all_variants(
        self,
        query: str,
        query_embedding: list[float],
        source: str,
        strategy: str,
        n_results: int = 5,
    ) -> dict[str, RetrievalResult]:
        """
        Ek query pe saare 7 variants run karo — comparison ke liye.

        RAGAS evaluation mein same query ke results compare karne ke liye
        useful hai — kaunsa variant best context retrieve karta hai.

        Args:
            query           : Query text.
            query_embedding : Pre-computed embedding.
            source          : FAA_CFR, FAA_AD, etc.
            strategy        : recursive, hierarchical, etc.
            n_results       : Top-k per variant.

        Returns:
            Dict mapping variant_name → RetrievalResult.
            Saare 7 variants ke results ek dict mein.
        """
        results: dict[str, RetrievalResult] = {}

        with tqdm(
            total=len(ALL_VARIANTS),
            desc=f"All Variants | {source} × {strategy}",
            unit="variant",
            colour="blue",
        ) as pbar:
            for variant in ALL_VARIANTS:
                try:
                    results[variant] = self.retrieve(
                        variant=variant,
                        query=query,
                        query_embedding=query_embedding,
                        source=source,
                        strategy=strategy,
                        n_results=n_results,
                    )
                except Exception as e:
                    logger.error(
                        "Variant failed | variant=%s | source=%s | error=%s",
                        variant, source, e,
                    )
                pbar.update(1)
                pbar.set_postfix({"variant": variant})

        return results