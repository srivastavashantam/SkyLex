"""
evaluation/ragas_evaluator.py

SkyLex — RAGAS-based RAG Evaluation Pipeline.

Kya karta hai:
  75 golden queries ko 7 RAG variants pe run karta hai.
  Har variant ke liye 4 RAGAS metrics calculate karta hai:
    - faithfulness        : Answer context se grounded hai?
    - answer_relevancy    : Answer question ke relevant hai?
    - context_precision   : Retrieved chunks precise hain?
    - context_recall      : Relevant chunks retrieve hue?
  Results MLflow experiment 'skylex_ragas_eval' mein logged hote hain.

Usage:
  python evaluation/ragas_evaluator.py --variants dense bm25 hybrid_rrf
  python evaluation/ragas_evaluator.py --sources FAA_CFR DGCA_CAR
  python evaluation/ragas_evaluator.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root ko sys.path mein add karo — script kahi se bhi run ho sake
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from tqdm import tqdm

from config.settings import settings
from evaluation.golden_queries import GOLDEN_QUERIES
from monitoring.logger import get_logger, setup_logging
from processing.retrieval.retrieval_factory import ALL_VARIANTS, RetrievalFactory
from processing.vector_store import CANDIDATE_STRATEGIES, SkyLexVectorStore

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT = "skylex_ragas_eval"
RESULTS_DIR       = Path("evaluation/results")
RAGAS_METRICS     = [faithfulness, answer_relevancy, context_precision, context_recall]

# ── Embedding ─────────────────────────────────────────────────────────────────


def embed_queries(questions: list[str], client: OpenAI) -> list[list[float]]:
    """Batch mein queries embed karo — ek baar, saare variants ke liye reuse."""
    logger.info("Embedding %d queries...", len(questions))
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=questions,
    )
    return [item.embedding for item in response.data]


# ── Answer Generation ─────────────────────────────────────────────────────────


def generate_answer(question: str, context_chunks: list[str], client: OpenAI) -> str:
    """Retrieved chunks se GPT-4o-mini ke saath answer generate karo."""
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context retrieved."
    system_prompt = (
        "You are an aviation regulatory expert. "
        "Answer the question using ONLY the provided context. "
        "If the context does not contain enough information, say so clearly. "
        "Be precise and cite specific regulations or section numbers when available."
    )
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=500,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content or "No answer generated."
    except Exception as e:
        logger.warning("Answer generation failed: %s", e)
        return "Answer generation failed."


# ── Core Evaluation Loop ──────────────────────────────────────────────────────


def run_variant_evaluation(
    variant: str,
    queries: list[dict[str, Any]],
    query_embeddings: list[list[float]],
    factory: RetrievalFactory,
    client: OpenAI,
    dry_run: bool = False,
) -> dict[str, list[Any]]:
    """
    Ek RAG variant ke liye saare queries evaluate karo.
    Returns RAGAS-compatible dict with question, answer, contexts, ground_truth.
    """
    logger.info("Evaluating variant: %s | queries: %d", variant, len(queries))

    ragas_data: dict[str, list[Any]] = {
        "question":     [],
        "answer":       [],
        "contexts":     [],
        "ground_truth": [],
    }

    for query_dict, embedding in tqdm(
        zip(queries, query_embeddings),
        total=len(queries),
        desc=f"  {variant}",
        unit="query",
    ):
        question     = query_dict["question"]
        ground_truth = query_dict["ground_truth"]
        source       = query_dict["source"]

        if dry_run:
            ragas_data["question"].append(question)
            ragas_data["answer"].append("DRY RUN")
            ragas_data["contexts"].append(["DRY RUN context"])
            ragas_data["ground_truth"].append(ground_truth)
            continue

        # Source ke primary strategy use karo
        strategies = CANDIDATE_STRATEGIES.get(source, ["recursive"])
        strategy   = strategies[0] if strategies else "recursive"

        # Retrieve
        try:
            result = factory.retrieve(
                variant=variant,
                query=question,
                query_embedding=embedding,
                source=source,
                strategy=strategy,
                n_results=5,
            )
            context_chunks: list[str] = result.documents if result.documents else []
        except Exception as e:
            logger.warning("Retrieval failed | variant=%s | error=%s", variant, e)
            context_chunks = []

        answer = generate_answer(question, context_chunks, client)

        ragas_data["question"].append(question)
        ragas_data["answer"].append(answer)
        ragas_data["contexts"].append(context_chunks if context_chunks else [""])
        ragas_data["ground_truth"].append(ground_truth)

    return ragas_data


# ── RAGAS Scoring ─────────────────────────────────────────────────────────────


def compute_ragas_scores(ragas_data: dict[str, list[Any]]) -> dict[str, float]:
    """RAGAS Dataset banao aur 4 metrics compute karo."""
    dataset = Dataset.from_dict({
        "question":     ragas_data["question"],
        "answer":       ragas_data["answer"],
        "contexts":     ragas_data["contexts"],
        "ground_truth": ragas_data["ground_truth"],
    })
    try:
        result = evaluate(dataset=dataset, metrics=RAGAS_METRICS)
        # result[metric] returns list[float] (per-row scores) — mean lena hoga
        def _mean(val: Any) -> float:
            if isinstance(val, list):
                return float(np.mean(val)) if val else 0.0
            return float(val)
        return {
            "faithfulness":      _mean(result["faithfulness"]),
            "answer_relevancy":  _mean(result["answer_relevancy"]),
            "context_precision": _mean(result["context_precision"]),
            "context_recall":    _mean(result["context_recall"]),
        }
    except Exception as e:
        logger.error("RAGAS compute failed: %s", e)
        return {m: 0.0 for m in ["faithfulness", "answer_relevancy",
                                  "context_precision", "context_recall"]}


# ── MLflow Logging ────────────────────────────────────────────────────────────


def log_to_mlflow(
    variant: str,
    scores: dict[str, float],
    query_count: int,
    elapsed_sec: float,
    sources_evaluated: list[str],
) -> None:
    """MLflow nested run mein variant results log karo."""
    with mlflow.start_run(run_name=f"variant_{variant}", nested=True):
        mlflow.log_params({
            "variant":          variant,
            "query_count":      query_count,
            "sources":          ",".join(sources_evaluated),
            "embedding_model":  settings.openai_embedding_model,
            "generation_model": settings.openai_model,
        })
        # Explicit float cast — Pylance list[float] type error avoid karo
        mlflow.log_metrics({
            "faithfulness":      float(scores["faithfulness"]),
            "answer_relevancy":  float(scores["answer_relevancy"]),
            "context_precision": float(scores["context_precision"]),
            "context_recall":    float(scores["context_recall"]),
            "elapsed_seconds":   float(elapsed_sec),
        })


# ── Results Save ──────────────────────────────────────────────────────────────


def save_results_json(payload: dict[str, Any], output_path: Path) -> None:
    """Results JSON file mein save karo."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Results saved: %s", output_path)


# ── Summary Table ─────────────────────────────────────────────────────────────


def print_comparison_table(all_scores: dict[str, dict[str, float]]) -> None:
    """7 variants × 4 metrics comparison table print karo."""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n" + "=" * 100)
    print("SKYLEX RAGAS EVALUATION RESULTS")
    print("=" * 100)
    print(f"{'Variant':<22}" + "".join(f"{m:<20}" for m in metrics))
    print("-" * 100)
    for variant, scores in all_scores.items():
        print(f"{variant:<22}" + "".join(f"{scores.get(m, 0.0):<20.4f}" for m in metrics))
    print("=" * 100)
    if all_scores:
        best = max(all_scores.items(), key=lambda x: sum(x[1].values()))
        avg  = sum(best[1].values()) / len(best[1])
        print(f"\n🏆 Best overall: {best[0]} (avg score: {avg:.4f})\n")


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def run_evaluation(
    variants: list[str],
    sources: list[str] | None,
    dry_run: bool,
) -> None:
    """Full RAGAS evaluation pipeline run karo."""

    setup_logging()

    # Query filter
    queries: list[dict[str, Any]]
    if sources:
        queries = [q for q in GOLDEN_QUERIES if q["source"] in sources]
        logger.info("Sources filtered: %s | queries: %d", sources, len(queries))
    else:
        queries = list(GOLDEN_QUERIES)
        logger.info("Using all %d golden queries", len(queries))

    # Clients — ek baar initialize, saare variants ke liye reuse
    logger.info("Initializing clients...")
    client  = OpenAI(api_key=settings.openai_api_key)
    store   = SkyLexVectorStore()
    factory = RetrievalFactory(store=store)   # store inject karo — no 'store' param in retrieve()

    # Embed once — reuse across all 7 variants
    questions: list[str] = [q["question"] for q in queries]
    embeddings: list[list[float]]
    if not dry_run:
        embeddings = embed_queries(questions, client)
    else:
        embeddings = [[0.0] * 1536] * len(questions)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    all_scores: dict[str, dict[str, float]] = {}
    all_raw: dict[str, dict[str, list[Any]]] = {}

    total_start = time.time()

    with mlflow.start_run(run_name="skylex_ragas_full_eval"):
        mlflow.log_params({
            "variants":      ",".join(variants),
            "total_queries": len(queries),
            "dry_run":       dry_run,
        })

        for variant in variants:
            logger.info("=" * 50)
            logger.info("Starting variant: %s", variant)
            t0 = time.time()

            ragas_data = run_variant_evaluation(
                variant=variant,
                queries=queries,
                query_embeddings=embeddings,
                factory=factory,
                client=client,
                dry_run=dry_run,
            )

            scores: dict[str, float]
            if not dry_run:
                scores = compute_ragas_scores(ragas_data)
            else:
                scores = {m: 0.0 for m in
                          ["faithfulness", "answer_relevancy",
                           "context_precision", "context_recall"]}

            elapsed        = time.time() - t0
            all_scores[variant] = scores
            all_raw[variant]    = ragas_data

            sources_eval = list({q["source"] for q in queries})
            log_to_mlflow(variant, scores, len(queries), elapsed, sources_eval)
            logger.info("Variant %s done | scores=%s | %.1fs", variant, scores, elapsed)

        total_elapsed = time.time() - total_start
        mlflow.log_metric("total_elapsed_seconds", float(total_elapsed))

    # Save results
    timestamp    = time.strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"ragas_results_{timestamp}.json"
    save_results_json(
        {"scores": all_scores, "raw_data": all_raw, "timestamp": timestamp},
        results_path,
    )

    print_comparison_table(all_scores)
    logger.info("Total time: %.1fs | Results: %s", total_elapsed, results_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SkyLex RAGAS Evaluation")
    parser.add_argument(
        "--variants", nargs="+", default=list(ALL_VARIANTS),
        choices=list(ALL_VARIANTS),
        help="RAG variants to evaluate (default: all 7)",
    )
    parser.add_argument(
        "--sources", nargs="+", default=None,
        choices=["FAA_CFR", "FAA_AD", "FAA_AC", "DGCA_CAR", "SKYBRARY"],
        help="Filter queries by source (default: all 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip API calls — test pipeline structure only",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        variants=args.variants,
        sources=args.sources,
        dry_run=args.dry_run,
    )