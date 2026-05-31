"""
evaluation/ragas_evaluator.py

SkyLex — Production RAG Evaluator: IR Metrics + LLM-as-Judge.

Tier 1 — IR Metrics (zero LLM calls, instant):
  precision@5  : Top 5 mein se kitne relevant doc_ids match karte hain
  recall@5     : Total relevant docs mein se kitne retrieve hue
  mrr@5        : Mean Reciprocal Rank — pehla relevant chunk kis position pe
  ndcg@5       : Normalized Discounted Cumulative Gain — ranking quality

Tier 2 — LLM-as-Judge (1 GPT-4o-mini call per query per variant):
  context_relevance : Retrieved chunks question ke liye kitne relevant (0-1)
  groundedness      : Answer chunks se kitna supported hai (0-1)

Ground truth source: golden_queries.py mein har query ka relevant_doc_ids field.

Total API calls: 25 queries × 7 variants × 1 = 175 calls (~5-8 min)
IR metrics: 0 API calls (~30 sec)

Usage:
  python evaluation/ragas_evaluator.py
  python evaluation/ragas_evaluator.py --variants dense bm25
  python evaluation/ragas_evaluator.py --dry-run
  python evaluation/ragas_evaluator.py --skip-llm-judge
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import numpy as np
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from tqdm import tqdm

from config.settings import settings
from evaluation.golden_queries import GOLDEN_QUERIES
from monitoring.logger import get_logger, setup_logging
from processing.retrieval.retrieval_factory import ALL_VARIANTS, RetrievalFactory
from processing.vector_store import CANDIDATE_STRATEGIES, SkyLexVectorStore

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT        = "skylex_ragas_eval"
RESULTS_DIR              = Path("evaluation/results")
QUERIES_PER_SOURCE       = 5
SOURCES                  = ["FAA_CFR", "FAA_AD", "FAA_AC", "DGCA_CAR", "SKYBRARY"]
K                        = 5          # Top-K for IR metrics
MAX_CONSECUTIVE_FAILURES = 5
INTER_CALL_DELAY_SEC     = 0.3        # RPM throttle — 429 avoid karne ke liye


# ── Query Sampling ────────────────────────────────────────────────────────────


def sample_queries(seed: int = 42) -> list[dict[str, Any]]:
    """Stratified sample — har source se QUERIES_PER_SOURCE queries, fixed seed."""
    sampled: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for source in SOURCES:
        source_qs = [q for q in GOLDEN_QUERIES if q["source"] == source]
        k = min(QUERIES_PER_SOURCE, len(source_qs))
        sampled.extend(rng.sample(source_qs, k))
    logger.info(
        "Sampled %d queries | %d sources × %d each",
        len(sampled), len(SOURCES), QUERIES_PER_SOURCE,
    )
    return sampled


# ── Embedding ─────────────────────────────────────────────────────────────────


@traceable(name="embed_batch", project_name="skylex")
def embed_batch(texts: list[str], client: OpenAI) -> list[list[float]]:
    """
    Texts ko batch embed karo — ek call mein.
    LangSmith mein token count + latency visible hoga.
    """
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ── Fatal Checks ──────────────────────────────────────────────────────────────


def check_fatal_conditions(client: OpenAI) -> None:
    """
    Run se pehle critical conditions verify karo.
    Koi bhi fail → ValueError → process hard stop.
    """
    try:
        client.models.list()
        logger.info("✅ OpenAI API key valid")
    except Exception as e:
        raise ValueError(f"❌ FATAL: OpenAI API not reachable: {e}") from e


# ── IR Metrics ────────────────────────────────────────────────────────────────


def extract_doc_id(chunk_metadata: dict[str, Any]) -> str:
    """
    Retrieved chunk ke metadata se doc_id extract karo.
    doc_id field na ho toh source + title se fallback banao.
    """
    return str(
        chunk_metadata.get("doc_id")
        or chunk_metadata.get("id")
        or f"{chunk_metadata.get('source', '')}_{chunk_metadata.get('title', '')}"
    )


def compute_ir_metrics(
    retrieved_metadatas: list[dict[str, Any]],
    relevant_doc_ids: list[str],
    k: int = K,
) -> dict[str, float]:
    """
    IR metrics compute karo — zero LLM calls.

    Args:
        retrieved_metadatas : Retrieved chunks ke metadata (ordered by rank).
        relevant_doc_ids    : Ground truth — kaunse doc_ids relevant hain.
        k                   : Top-K cutoff.

    Returns:
        precision@k, recall@k, mrr@k, ndcg@k
    """
    if not relevant_doc_ids:
        return {
            "precision@k": 0.0,
            "recall@k":    0.0,
            "mrr@k":       0.0,
            "ndcg@k":      0.0,
        }

    relevant_set = set(relevant_doc_ids)
    retrieved_ids = [
        extract_doc_id(meta)
        for meta in retrieved_metadatas[:k]
    ]

    # Relevance vector — 1 agar relevant, 0 agar nahi
    relevance = [1 if doc_id in relevant_set else 0 for doc_id in retrieved_ids]

    # Precision@K
    precision = sum(relevance) / k if k > 0 else 0.0

    # Recall@K
    recall = sum(relevance) / len(relevant_set) if relevant_set else 0.0

    # MRR@K — pehla relevant result kitne position pe hai
    mrr = 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel == 1:
            mrr = 1.0 / rank
            break

    # NDCG@K
    dcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(relevance, start=1)
    )
    # Ideal DCG — agar saare relevant docs top pe hote
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_relevance, start=1)
    )
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "precision@k": round(precision, 4),
        "recall@k":    round(recall, 4),
        "mrr@k":       round(mrr, 4),
        "ndcg@k":      round(ndcg, 4),
    }


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────


JUDGE_SYSTEM_PROMPT = """You are an expert aviation regulatory evaluator.
You will be given a question, retrieved context chunks, and you must evaluate:

1. context_relevance (0.0 to 1.0):
   How relevant are the retrieved chunks to answering the question?
   0.0 = completely irrelevant, 1.0 = perfectly relevant

2. groundedness (0.0 to 1.0):
   If someone were to answer the question using ONLY these chunks,
   how well-grounded/supported would that answer be?
   0.0 = no support, 1.0 = fully supported

Respond ONLY with valid JSON, no explanation:
{"context_relevance": <float>, "groundedness": <float>}"""


@traceable(name="llm_judge", project_name="skylex")
def llm_judge(
    question: str,
    context_chunks: list[str],
    client: OpenAI,
) -> dict[str, float]:
    """
    LLM-as-Judge — 1 GPT-4o-mini call per query.
    LangSmith mein tokens, cost, latency track hoga.
    Returns context_relevance aur groundedness (0-1).
    """
    context_text = "\n\n---\n\n".join(context_chunks[:3]) if context_chunks else "No context."

    user_prompt = f"Question: {question}\n\nRetrieved Context:\n{context_text}"

    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                max_tokens=60,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            parsed: dict[str, float] = json.loads(raw)
            return {
                "context_relevance": float(parsed.get("context_relevance", 0.0)),
                "groundedness":      float(parsed.get("groundedness", 0.0)),
            }
        except json.JSONDecodeError:
            logger.warning("LLM judge JSON parse failed | attempt=%d", attempt + 1)
            return {"context_relevance": 0.0, "groundedness": 0.0}
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            if is_rate_limit and attempt < 3:
                wait = 5.0 * (2 ** attempt)
                logger.warning("Rate limit | attempt=%d | waiting=%.1fs", attempt + 1, wait)
                time.sleep(wait)
            else:
                logger.warning("LLM judge failed after %d attempts: %s", attempt + 1, e)
                return {"context_relevance": 0.0, "groundedness": 0.0}

    return {"context_relevance": 0.0, "groundedness": 0.0}


# ── Core Evaluation Loop ──────────────────────────────────────────────────────


def run_variant_evaluation(
    variant: str,
    queries: list[dict[str, Any]],
    query_embeddings: list[list[float]],
    factory: RetrievalFactory,
    client: OpenAI,
    skip_llm_judge: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Ek variant ke liye saare queries evaluate karo.
    IR metrics + LLM-as-Judge dono compute karo.
    consecutive_failures >= MAX pe hard stop.
    """
    logger.info("Evaluating variant: %s | queries: %d", variant, len(queries))

    per_query:   list[dict[str, Any]] = []
    ir_scores:   list[dict[str, float]] = []
    judge_scores: list[dict[str, float]] = []
    consecutive_failures = 0

    for query_dict, q_embedding in tqdm(
        zip(queries, query_embeddings),
        total=len(queries),
        desc=f"  {variant}",
        unit="query",
    ):
        question         = query_dict["question"]
        ground_truth     = query_dict["ground_truth"]
        source           = query_dict["source"]
        query_id         = query_dict["query_id"]
        relevant_doc_ids = query_dict.get("relevant_doc_ids", [])

        if dry_run:
            dummy_ir    = {"precision@k": 0.0, "recall@k": 0.0, "mrr@k": 0.0, "ndcg@k": 0.0}
            dummy_judge = {"context_relevance": 0.0, "groundedness": 0.0}
            ir_scores.append(dummy_ir)
            judge_scores.append(dummy_judge)
            per_query.append({
                "query_id": query_id, **dummy_ir, **dummy_judge,
            })
            continue

        # Source ki primary strategy
        strategies = CANDIDATE_STRATEGIES.get(source, ["recursive"])
        strategy   = strategies[0] if strategies else "recursive"

        # ── Retrieve ──────────────────────────────────────────────────────────
        try:
            result = factory.retrieve(
                variant=variant,
                query=question,
                query_embedding=q_embedding,
                source=source,
                strategy=strategy,
                n_results=K,
            )
            chunks:    list[str]            = result.documents  if result.documents  else []
            metadatas: list[dict[str, Any]] = result.metadatas  if result.metadatas  else []
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            logger.warning(
                "Retrieval failed | variant=%s | failures=%d/%d | %s",
                variant, consecutive_failures, MAX_CONSECUTIVE_FAILURES, e,
            )
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"FATAL: {consecutive_failures} consecutive retrieval failures "
                    f"for variant '{variant}'. ChromaDB/BM25 check karo."
                )
            chunks, metadatas = [], []

        # ── IR Metrics (zero LLM) ─────────────────────────────────────────────
        ir = compute_ir_metrics(
            retrieved_metadatas=metadatas,
            relevant_doc_ids=relevant_doc_ids,
            k=K,
        )
        ir_scores.append(ir)

        # ── LLM-as-Judge (1 call per query) ──────────────────────────────────
        if not skip_llm_judge:
            time.sleep(INTER_CALL_DELAY_SEC)  # RPM throttle
            judge = llm_judge(question, chunks, client)
        else:
            judge = {"context_relevance": 0.0, "groundedness": 0.0}
        judge_scores.append(judge)

        per_query.append({
            "query_id":        query_id,
            "source":          source,
            "question":        question,
            "ground_truth":    ground_truth,
            "relevant_doc_ids":relevant_doc_ids,
            "retrieved_chunks":chunks[:2],  # Sirf first 2 save karo — space bachao
            **ir,
            **judge,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _mean_key(scores: list[dict[str, float]], key: str) -> float:
        vals = [s[key] for s in scores if key in s]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "per_query":          per_query,
        "precision@k":        _mean_key(ir_scores,    "precision@k"),
        "recall@k":           _mean_key(ir_scores,    "recall@k"),
        "mrr@k":              _mean_key(ir_scores,    "mrr@k"),
        "ndcg@k":             _mean_key(ir_scores,    "ndcg@k"),
        "context_relevance":  _mean_key(judge_scores, "context_relevance"),
        "groundedness":       _mean_key(judge_scores, "groundedness"),
    }


# ── MLflow Logging ────────────────────────────────────────────────────────────


def log_to_mlflow(
    variant: str,
    result: dict[str, Any],
    query_count: int,
    elapsed_sec: float,
) -> None:
    """MLflow nested run mein variant scores log karo."""
    with mlflow.start_run(run_name=f"variant_{variant}", nested=True):
        mlflow.log_params({
            "variant":     variant,
            "query_count": query_count,
            "k":           K,
        })
        mlflow.log_metrics({
            "precision_at_k":    float(result["precision@k"]),
            "recall_at_k":       float(result["recall@k"]),
            "mrr_at_k":          float(result["mrr@k"]),
            "ndcg_at_k":         float(result["ndcg@k"]),
            "context_relevance": float(result["context_relevance"]),
            "groundedness":      float(result["groundedness"]),
            "elapsed_seconds":   float(elapsed_sec),
        })


# ── Results Save ──────────────────────────────────────────────────────────────


def save_results_json(payload: dict[str, Any], output_path: Path) -> None:
    """Results JSON mein save karo."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Results saved: %s", output_path)


# ── Summary Table ─────────────────────────────────────────────────────────────


def print_comparison_table(
    all_results: dict[str, dict[str, Any]],
    skip_llm_judge: bool,
) -> None:
    """7 variants × 6 metrics comparison table."""
    ir_metrics    = ["precision@k", "recall@k", "mrr@k", "ndcg@k"]
    judge_metrics = ["context_relevance", "groundedness"]
    metrics       = ir_metrics + ([] if skip_llm_judge else judge_metrics)

    col_w = 18
    total_w = 25 + col_w * len(metrics)

    print("\n" + "=" * total_w)
    print(f"SKYLEX RAG EVALUATION RESULTS  (K={K})")
    print("=" * total_w)
    print(f"{'Variant':<25}" + "".join(f"{m:<{col_w}}" for m in metrics))
    print("-" * total_w)

    for variant, result in all_results.items():
        row = f"{variant:<25}"
        for m in metrics:
            row += f"{result.get(m, 0.0):<{col_w}.4f}"
        print(row)

    print("=" * total_w)

    if all_results:
        best = max(
            all_results.items(),
            key=lambda x: sum(x[1].get(m, 0.0) for m in metrics),
        )
        avg = sum(best[1].get(m, 0.0) for m in metrics) / len(metrics)
        print(f"\n🏆 Best: {best[0]} (avg={avg:.4f})\n")


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def run_evaluation(
    variants: list[str],
    skip_llm_judge: bool,
    dry_run: bool,
) -> None:
    """Full evaluation pipeline — IR Metrics + LLM-as-Judge."""

    setup_logging()
    queries = sample_queries(seed=42)

    logger.info("Initializing clients...")
    client  = wrap_openai(OpenAI(api_key=settings.openai_api_key))
    store   = SkyLexVectorStore()
    factory = RetrievalFactory(store=store)

    if not dry_run:
        check_fatal_conditions(client)

    # Queries embed karo — ek baar, saare variants ke liye reuse
    questions: list[str] = [q["question"] for q in queries]
    if not dry_run:
        logger.info("Embedding %d queries...", len(questions))
        q_embeddings = embed_batch(questions, client)
    else:
        q_embeddings = [[0.0] * 1536] * len(questions)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    all_results: dict[str, dict[str, Any]] = {}
    total_start = time.time()

    with mlflow.start_run(run_name="skylex_eval_ir_judge"):
        mlflow.log_params({
            "variants":       ",".join(variants),
            "total_queries":  len(queries),
            "k":              K,
            "skip_llm_judge": skip_llm_judge,
            "dry_run":        dry_run,
        })

        for variant in variants:
            logger.info("=" * 50)
            logger.info("Starting variant: %s", variant)
            t0 = time.time()

            try:
                result = run_variant_evaluation(
                    variant=variant,
                    queries=queries,
                    query_embeddings=q_embeddings,
                    factory=factory,
                    client=client,
                    skip_llm_judge=skip_llm_judge,
                    dry_run=dry_run,
                )
            except RuntimeError as e:
                logger.error("FATAL for variant %s: %s — skipping", variant, e)
                all_results[variant] = {
                    m: 0.0 for m in
                    ["precision@k", "recall@k", "mrr@k", "ndcg@k",
                     "context_relevance", "groundedness"]
                }
                continue

            elapsed               = time.time() - t0
            all_results[variant]  = result
            log_to_mlflow(variant, result, len(queries), elapsed)

            logger.info(
                "Variant %s done | p@k=%.3f | r@k=%.3f | mrr=%.3f | "
                "ndcg=%.3f | rel=%.3f | grnd=%.3f | %.1fs",
                variant,
                result["precision@k"],
                result["recall@k"],
                result["mrr@k"],
                result["ndcg@k"],
                result["context_relevance"],
                result["groundedness"],
                elapsed,
            )

        total_elapsed = time.time() - total_start
        mlflow.log_metric("total_elapsed_seconds", float(total_elapsed))

    timestamp    = time.strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"eval_results_{timestamp}.json"
    save_results_json(
        {"results": all_results, "k": K, "timestamp": timestamp},
        results_path,
    )

    print_comparison_table(all_results, skip_llm_judge)
    logger.info("Total time: %.1fs | Results: %s", total_elapsed, results_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SkyLex RAG Evaluator — IR Metrics + LLM-as-Judge"
    )
    parser.add_argument(
        "--variants", nargs="+", default=list(ALL_VARIANTS),
        choices=list(ALL_VARIANTS),
        help="RAG variants to evaluate (default: all 7)",
    )
    parser.add_argument(
        "--skip-llm-judge", action="store_true",
        help="Sirf IR metrics — zero LLM calls (rate limit issue pe use karo)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip all API calls — pipeline structure test only",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        variants=args.variants,
        skip_llm_judge=args.skip_llm_judge,
        dry_run=args.dry_run,
    )