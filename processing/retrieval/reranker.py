"""
processing/retrieval/reranker.py

R4 Component — Cross-Encoder Reranker.

Retrieval pipeline mein yeh second-pass scoring karta hai.
Pehle Hybrid RRF (R3) se top-50 candidate chunks aate hain,
phir cross-encoder har chunk ko query ke context mein score karta hai.

=============================================================================
WHY RERANKING — The Core Problem with First-Pass Retrieval
=============================================================================

Dense aur BM25 dono "bi-encoder" approach use karte hain:
  - Query ko alag embed/score karo
  - Document ko alag embed/score karo
  - Similarity calculate karo

Yeh fast hai (similarity pre-computed ho sakti hai) lekin ek fundamental
limitation hai: query aur document ke beech ki INTERACTION capture nahi hoti.

Example:
  Query:    "When is alcohol testing mandatory for pilots?"
  Chunk A:  "Pilots must not consume alcohol 8 hours before duty" → score 0.82
  Chunk B:  "Random alcohol testing is mandatory under DGCA CAR Section 7.1
             for all commercial pilots before each flight duty period" → score 0.79

  Dense retriever ne Chunk A ko upar rakha kyunki "alcohol" aur "pilots"
  dono words hain. Lekin Chunk B actually query ka better answer hai —
  "mandatory testing" aur "when" directly address karta hai.

  Cross-encoder query + document dono ko SAATH process karta hai:
    Input: "[QUERY] When is alcohol testing mandatory... [SEP] [DOC] Random
            alcohol testing is mandatory under DGCA CAR Section 7.1..."
  
  Model directly query-document relevance judge karta hai → Chunk B wins.

=============================================================================
MODEL — cross-encoder/ms-marco-MiniLM-L-6-v2
=============================================================================

  - MS MARCO dataset pe trained — 8.8M query-passage pairs
  - MiniLM-L6 architecture — 6 transformer layers, fast inference
  - CPU pe ~200ms for 50 chunks reranking
  - Free, local, no API dependency
  - First time use pe HuggingFace se automatically download hoga (~80MB)

=============================================================================
PIPELINE POSITION
=============================================================================

  Step 1: Hybrid RRF → top-50 candidates (fast, approximate)
  Step 2: Cross-Encoder → rerank top-50 (slow, precise)
  Step 3: Return top-5 to LLM (best quality)

  Kyun 50 candidates pehle:
    Cross-encoder ke liye har query-document pair ek forward pass hai.
    72,827 chunks pe directly chalana = ~14,565 forward passes per query = impractical.
    50 candidates pe = 50 forward passes = ~200ms = acceptable.

Monitoring:
  - tqdm: Reranking batch progress
  - MLflow: Reranking latency, score distributions, improvement over first-pass
  - LangSmith: @traceable — full reranking trace with input/output
"""

from __future__ import annotations

import time
from typing import Any

import mlflow
from langsmith import traceable
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from monitoring.logger import get_logger
from processing.retrieval.dense_retriever import RetrievalResult

logger = get_logger(__name__)

# HuggingFace model name — first use pe ~80MB download hoga automatically.
# MS MARCO pe trained — passage retrieval ke liye specifically designed.
_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cross-encoder score threshold — is se neeche ke chunks low quality maane jaate hain.
# MS MARCO model ka score range approximately -10 to +10 hota hai.
# 0.0 ek reasonable threshold hai — negative score matlab chunk irrelevant hai.
_MIN_RELEVANCE_SCORE: float = 0.0


# ── Reranker ──────────────────────────────────────────────────────────────────


class CrossEncoderReranker:
    """
    Cross-Encoder based reranker — second-pass precision scoring.

    Pehli baar instantiate karne pe model load hoga (~2-3 seconds).
    Subsequent calls fast honge — model memory mein cached rahta hai.

    Singleton pattern recommend karta hoon — ek instance poore application
    mein reuse karo, baar baar load mat karo.

    Usage:
        reranker = CrossEncoderReranker()

        # Hybrid RRF ka result pass karo
        reranked = reranker.rerank(
            query="§ 121.135 fuel reserve requirements",
            retrieval_result=hybrid_result,
            top_n=5,
        )
    """

    rag_variant: str = "hybrid_rrf_reranked"

    def __init__(
        self,
        model_name: str = _CROSS_ENCODER_MODEL,
        min_relevance_score: float = _MIN_RELEVANCE_SCORE,
    ) -> None:
        """
        Args:
            model_name          : HuggingFace cross-encoder model.
                                  Default: ms-marco-MiniLM-L-6-v2
            min_relevance_score : Is threshold se neeche ke chunks filter honge.
                                  -10 se +10 range mein hota hai MS MARCO model ka.
        """
        logger.info(
            "Loading cross-encoder model: %s — first time pe HuggingFace se "
            "download hoga (~80MB)", model_name
        )
        # CPU pe load karo — GPU optional hai, CPU pe bhi fast enough hai
        # 50 chunks ke reranking ke liye (~200ms).
        self._model = CrossEncoder(model_name, max_length=512)
        self._min_relevance_score = min_relevance_score

        logger.info("Cross-encoder model loaded successfully: %s", model_name)

    @traceable(
        name="cross_encoder_rerank",
        project_name="skylex",
    )
    def rerank(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        top_n: int = 5,
    ) -> RetrievalResult:
        """
        Retrieval result ko cross-encoder se rerank karo.

        Process:
          1. Har (query, chunk) pair ke liye cross-encoder score calculate karo.
             Cross-encoder dono ko SAATH process karta hai — direct interaction.
          2. Score ke basis pe chunks sort karo — descending order.
          3. top_n chunks return karo with updated scores.

        Score interpretation (MS MARCO model):
          > 5.0  : Highly relevant — direct answer
          1-5    : Relevant — related content
          -1 to 1: Marginally relevant
          < -1   : Likely irrelevant

        Args:
            query            : Original user query — same jo retrieval mein use hua.
            retrieval_result : Hybrid RRF ya kisi bhi retriever ka output.
                               Documents yahan se liye jaate hain reranking ke liye.
            top_n            : Final mein kitne chunks return karne hain.

        Returns:
            RetrievalResult with:
              - chunks sorted by cross-encoder score (descending)
              - scores updated to cross-encoder scores
              - extra mein original order aur score comparison
        """
        start_time = time.perf_counter()

        documents = retrieval_result.documents

        # Edge case: agar koi document nahi hai toh original return karo.
        if not documents:
            logger.warning(
                "Reranker called with empty documents | query='%s'", query[:50]
            )
            return retrieval_result

        # Cross-encoder ke liye (query, document) pairs banao.
        # Model yeh pairs ek saath process karta hai — yahi cross-encoder ka magic hai.
        query_doc_pairs = [(query, doc) for doc in documents]

        logger.debug(
            "Cross-encoder scoring %d query-document pairs | query='%s'",
            len(query_doc_pairs), query[:50]
        )

        # Scores calculate karo — numpy array return hota hai.
        # show_progress_bar=False kyunki hum apna tqdm manage karte hain.
        ce_scores = self._model.predict(
            query_doc_pairs,
            show_progress_bar=False,
        )

        # Original index ke saath scores zip karo — order track karne ke liye.
        scored_chunks = list(zip(ce_scores, range(len(documents))))

        # Descending order mein sort karo — highest score pehle.
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # top_n rakho.
        safe_top_n = min(top_n, len(scored_chunks))
        top_scored = scored_chunks[:safe_top_n]

        # Naye order mein chunks assemble karo.
        reranked_indices = [idx for _, idx in top_scored]
        reranked_scores = [round(float(score), 4) for score, _ in top_scored]

        reranked_chunks = [retrieval_result.chunk_ids[i] for i in reranked_indices]
        reranked_docs = [retrieval_result.documents[i] for i in reranked_indices]
        reranked_meta = [retrieval_result.metadatas[i] for i in reranked_indices]

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Score improvement track karo — pehle kya tha, rerank ke baad kya hua.
        original_top_chunk = retrieval_result.chunk_ids[0] if retrieval_result.chunk_ids else ""
        reranked_top_chunk = reranked_chunks[0] if reranked_chunks else ""
        order_changed = original_top_chunk != reranked_top_chunk

        logger.debug(
            "Reranking complete | input=%d chunks | output=%d chunks | "
            "top_ce_score=%.4f | order_changed=%s | latency=%.1fms",
            len(documents), len(reranked_chunks),
            reranked_scores[0] if reranked_scores else 0.0,
            order_changed, latency_ms,
        )

        return RetrievalResult(
            query=query,
            source=retrieval_result.source,
            strategy=retrieval_result.strategy,
            rag_variant=self.rag_variant,
            chunk_ids=reranked_chunks,
            documents=reranked_docs,
            metadatas=reranked_meta,
            scores=reranked_scores,
            latency_ms=round(
                retrieval_result.latency_ms + latency_ms, 2
            ),
            extra={
                # First-pass retrieval ki info preserve karo comparison ke liye.
                "first_pass_variant":    retrieval_result.rag_variant,
                "first_pass_top_score":  retrieval_result.scores[0] if retrieval_result.scores else 0.0,
                "ce_top_score":          reranked_scores[0] if reranked_scores else 0.0,
                "ce_bottom_score":       reranked_scores[-1] if reranked_scores else 0.0,
                "order_changed":         order_changed,
                "candidates_reranked":   len(documents),
                "rerank_latency_ms":     round(latency_ms, 2),
                "retrieval_latency_ms":  retrieval_result.latency_ms,
            },
        )

    def rerank_batch(
        self,
        queries: list[str],
        retrieval_results: list[RetrievalResult],
        top_n: int = 5,
        mlflow_run_name: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Multiple queries ke liye batch reranking.

        tqdm progress bar dikhata hai kitne queries rerank ho gaye.
        MLflow mein aggregate metrics log hote hain.

        Args:
            queries           : Query strings — retrieval_results ke saath 1:1 mapping.
            retrieval_results : First-pass retrieval results jo rerank karne hain.
            top_n             : Final top-n chunks per query.
            mlflow_run_name   : MLflow nested run name — None matlab log nahi karo.

        Returns:
            List of reranked RetrievalResults.
        """
        if len(queries) != len(retrieval_results):
            raise ValueError(
                f"queries ({len(queries)}) aur retrieval_results "
                f"({len(retrieval_results)}) ki length equal honi chahiye."
            )

        results: list[RetrievalResult] = []
        total_rerank_latency_ms = 0.0
        order_changed_count = 0

        # tqdm — magenta colour reranker ke liye (distinct from other retrievers).
        with tqdm(
            total=len(queries),
            desc=f"Cross-Encoder Rerank | {retrieval_results[0].source if retrieval_results else ''}",
            unit="query",
            colour="magenta",
        ) as pbar:
            for query, first_pass_result in zip(queries, retrieval_results):
                reranked = self.rerank(
                    query=query,
                    retrieval_result=first_pass_result,
                    top_n=top_n,
                )
                results.append(reranked)

                rerank_lat = reranked.extra.get("rerank_latency_ms", 0.0)
                total_rerank_latency_ms += rerank_lat

                if reranked.extra.get("order_changed", False):
                    order_changed_count += 1

                pbar.update(1)
                pbar.set_postfix({
                    "ce_score":     f"{reranked.extra.get('ce_top_score', 0):.2f}",
                    "reordered":    order_changed_count,
                    "latency":      f"{rerank_lat:.0f}ms",
                })

        if mlflow_run_name:
            self._log_batch_metrics(
                results=results,
                total_rerank_latency_ms=total_rerank_latency_ms,
                order_changed_count=order_changed_count,
                mlflow_run_name=mlflow_run_name,
            )

        logger.info(
            "Reranking batch complete | queries=%d | order_changed=%d/%d | "
            "avg_rerank_latency=%.1fms",
            len(queries), order_changed_count, len(queries),
            total_rerank_latency_ms / len(queries) if queries else 0,
        )

        return results

    def _log_batch_metrics(
        self,
        results: list[RetrievalResult],
        total_rerank_latency_ms: float,
        order_changed_count: int,
        mlflow_run_name: str,
    ) -> None:
        """
        MLflow mein reranking batch ke aggregate metrics log karo.

        Key metric: order_changed_rate — kitne queries mein reranking ne
        top result change kiya. High rate = first-pass retrieval mein significant
        imprecision thi, reranker ne value add ki.
        """
        if not results:
            return

        avg_rerank_latency = total_rerank_latency_ms / len(results)
        avg_ce_top_score = sum(
            r.extra.get("ce_top_score", 0.0) for r in results
        ) / len(results)
        order_changed_rate = order_changed_count / len(results)

        with mlflow.start_run(run_name=mlflow_run_name, nested=True):
            mlflow.log_params({
                "rag_variant":   self.rag_variant,
                "model":         _CROSS_ENCODER_MODEL,
                "query_count":   len(results),
            })
            mlflow.log_metrics({
                "avg_rerank_latency_ms": round(avg_rerank_latency, 2),
                "total_rerank_latency":  round(total_rerank_latency_ms, 2),
                "avg_ce_top_score":      round(avg_ce_top_score, 4),
                # order_changed_rate batata hai kitne queries mein reranker ne
                # first-pass ka top result badla. 0.0 = reranker useless tha,
                # 1.0 = har query mein order change hua.
                "order_changed_rate":    round(order_changed_rate, 4),
                "order_changed_count":   order_changed_count,
            })