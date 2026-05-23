"""
processing/embedding_pipeline.py

SkyLex Embedding Pipeline — Chunk + Embed + Store.

Har source × candidate strategy combination ke liye:
  1. Raw JSON documents load karo (data/raw/{source}/)
  2. Us strategy se chunk karo
  3. Chunks ko OpenAI text-embedding-3-small se embed karo (batch=20)
  4. Specific source×strategy ChromaDB collection mein store karo
  5. MLflow mein metrics log karo

Kyun har strategy ke liye alag embedding:
  - RAGAS evaluation mein same golden queries alag strategy collections pe chalti hain
  - Retrieval quality compare karke final strategy decide hoti hai
  - Yeh Phase 3 ka core experiment hai

Candidate strategies per source (Phase 2 top performers):
  FAA_CFR  → semantic, recursive, double_pass
  FAA_AD   → hierarchical, improved_hybrid, hybrid
  FAA_AC   → double_pass, recursive, improved_semantic
  DGCA_CAR → double_pass, recursive, improved_semantic
  SKYBRARY → recursive, improved_semantic, improved_recursive

Total runs: 15 (5 sources × 3 strategies each)

FAA_CFR semantic mein hard cap fix apply hoga —
ImprovedSemanticChunker use hoga jo max 6000 chars per chunk enforce karta hai.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from langsmith import traceable
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from config.settings import settings
from monitoring.logger import get_logger
from processing.base_chunker import ChunkedDocument
from processing.strategies import (
    DoublePassChunker,
    HierarchicalChunker,
    HybridChunker,
    ImprovedHybridChunker,
    ImprovedRecursiveChunker,
    ImprovedSemanticChunker,
    RecursiveChunker,
    SemanticChunker,
)
from processing.vector_store import CANDIDATE_STRATEGIES, SkyLexVectorStore

# Environment variables aur LangSmith telemetry authentication tokens initialize karne k liye explicit load call.
load_dotenv()

# Pipeline lifecycle events, embedding batches aur MLflow tracking events ko console par trace karne ke liye.
logger = get_logger(__name__)

# Direct OpenAI client — actual token usage LangSmith mein track karne ke liye.
# LangChain ke embed_documents() wrapper mein usage metadata chup jata hai, isliye explicit native API client instantiate kiya hai.
_openai_client = OpenAI(api_key=settings.openai_api_key)


# ── Constants ─────────────────────────────────────────────────────────────────

# Raw JSON files ka base root directory path pointer assignment.
_RAW_DATA_DIR: Path = Path("data/raw")

# OpenAI API limits aur network payload congestion ko prevent karne ke liye fixed safe boundary cap.
_EMBED_BATCH_SIZE: int = 20

# System source indicators se actual file system folder names ki routing memory map definitions.
_SOURCE_TO_FOLDER: dict[str, str] = {
    "FAA_CFR":  "faa_cfr",
    "FAA_AD":   "faa_ad",
    "FAA_AC":   "faa_ac",
    "DGCA_CAR": "dgca_car",
    "SKYBRARY": "skybrary",
}

# Master dashboard telemetry index name taaki Phase 3 runs Phase 2 pipeline matrices ke saath clash na karein.
_MLFLOW_EXPERIMENT: str = "skylex_phase3_embedding"


# ── Strategy Factory ──────────────────────────────────────────────────────────


def _get_chunker(source: str, strategy: str) -> Any:
    """
    Source + strategy ke liye correct chunker instance return karo.

    FAA_CFR + semantic special case:
      Original SemanticChunker ka max chunk 35,941 chars tha —
      embedding model limit (8191 tokens ≈ 32,764 chars) cross karta hai.
      ImprovedSemanticChunker use karenge jo hard cap 6000 chars enforce karta hai.
    """
    # Core conditional bypass logic: Baseline semantic engine crash shield. FAA_CFR + semantic ko safe limits par force kiya hai.
    if source == "FAA_CFR" and strategy == "semantic":
        return ImprovedSemanticChunker(
            breakpoint_threshold_amount=85.0,
            hard_cap_chars=6000,
        )

    # Master registry map jo strictly evaluation specific algorithms ki configuration boundaries hold karta hai.
    chunker_map: dict[str, Any] = {
        "recursive":          RecursiveChunker(chunk_size=1000, chunk_overlap=150),
        "improved_recursive": ImprovedRecursiveChunker(chunk_size=1000, chunk_overlap=200),
        "hierarchical":       HierarchicalChunker(max_chunk_size=3000),
        "hybrid":             HybridChunker(max_chunk_size=1500, chunk_overlap=150),
        "improved_hybrid":    ImprovedHybridChunker(max_chunk_size=2000, chunk_overlap=200),
        "semantic":           SemanticChunker(breakpoint_threshold_amount=95.0),
        "improved_semantic":  ImprovedSemanticChunker(
                                  breakpoint_threshold_amount=85.0,
                                  hard_cap_chars=6000,
                              ),
        "double_pass":        DoublePassChunker(
                                  coarse_threshold=90.0,
                                  fine_chunk_size=1000,
                              ),
    }

    # Configuration validation guardrail agar unauthorized strings input pass ho jayein.
    if strategy not in chunker_map:
        raise ValueError(
            f"Unknown strategy '{strategy}' for source '{source}'."
        )

    # Initialize ki gayi strategy class direct execution reference memory path par wapas return hogi.
    return chunker_map[strategy]


# ── Data Loader ───────────────────────────────────────────────────────────────


def _load_documents(source: str) -> list[dict[str, Any]]:
    """
    Source ke raw JSON folder se saare documents load karo.
    hash_registry.json aur _meta.json exclude karo.
    Duplicate doc_ids deduplicate — pehli occurrence rakho.
    """
    # Mapping table pointer ko resolve karke actual disk path URL target structure extract kiya gaya.
    folder = _RAW_DATA_DIR / _SOURCE_TO_FOLDER[source]

    # Pre-execution environment sanity check directory folder read confirmation bypass tracking step.
    if not folder.exists():
        logger.warning(
            "Raw data folder not found | source=%s | path=%s", source, folder
        )
        return []

    # Valid data items container initialization block aur memory hash routing tracks for duplicate filtration checks.
    all_docs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Directory layout sub-trees recursive search matching JSON identifiers filter tracking logic implementation loop.
    for json_file in sorted(folder.rglob("*.json")):
        # Registry history tracking files or structural properties map bypassing rules checklist path sequence.
        if json_file.name == "hash_registry.json":
            continue
        if json_file.name.endswith("_meta.json"):
            continue

        try:
            # File read operational bytes streams parsing conversion routines validation syntax blocks.
            with json_file.open(encoding="utf-8") as f:
                data: Any = json.load(f)

            # Dictionary parameter structural validation conditions check data payload dictionary key extraction verifier constraints.
            if not isinstance(data, dict) or "documents" not in data:
                continue

            # Sequential unpacking loops inner level dictionary mapping items identity code parameter duplicate protection cache validations context.
            for doc in data["documents"]:
                doc_id: str = doc.get("doc_id", "")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)

        except (json.JSONDecodeError, OSError) as e:
            # Safe catch exceptions boundaries fault tolerance runtime logs info screen display tracking operations checks.
            logger.warning("Could not load %s: %s", json_file, e)

    # Execution telemetry log message strings terminal report completion data records length counter print parameters options block.
    logger.info("Documents loaded | source=%s | count=%d", source, len(all_docs))
    return all_docs


# ── Embedding ─────────────────────────────────────────────────────────────────


from langsmith.run_helpers import get_current_run_tree

@traceable(
    name="embed_chunks",
    project_name="skylex",
)
def _embed_in_batches(
    texts: list[str],
    embedder: OpenAIEmbeddings,
) -> list[list[float]]:
    """
    Chunk texts ko batch mein embed karo.
    Direct OpenAI client use karo — actual token usage LangSmith mein track hoga.
    """
    all_embeddings: list[list[float]] = []
    total_tokens = 0
    total_batches = (len(texts) + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE

    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]

        response = _openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
        )

        total_tokens += response.usage.total_tokens

        batch_embeddings: list[list[float]] = [
            item.embedding
            for item in sorted(response.data, key=lambda x: x.index)
        ]
        all_embeddings.extend(batch_embeddings)

        current_batch = i // _EMBED_BATCH_SIZE + 1
        if current_batch % 10 == 0 or current_batch == total_batches:
            logger.debug(
                "Embedding batch %d/%d | chunks done: %d/%d | tokens: %d",
                current_batch, total_batches,
                min(i + _EMBED_BATCH_SIZE, len(texts)),
                len(texts), total_tokens,
            )

    actual_cost = (total_tokens / 1_000_000) * 0.02

    # LangSmith ko actual token usage explicitly pass karo.
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.end(
            outputs={"chunk_count": len(texts)},
            metadata={
                "actual_tokens": total_tokens,
                "actual_cost_usd": actual_cost,
                "embedding_model": settings.openai_embedding_model,
            },
        )

    logger.info(
        "Embedding complete | chunks=%d | actual_tokens=%d | actual_cost=$%.6f",
        len(texts), total_tokens, actual_cost,
    )

    return all_embeddings


# ── Cost Estimator ────────────────────────────────────────────────────────────


def _estimate_tokens(texts: list[str]) -> int:
    """Rough token estimate — avg 4 chars per token. Pre-embedding cost preview ke liye."""
    # Comprehension loop lengths total calculation factors conversion formula bypass standard tiktoken mapping overheads.
    return sum(len(t) for t in texts) // 4


def _estimate_cost_usd(token_count: int) -> float:
    """text-embedding-3-small: $0.02 per 1M tokens."""
    # Fast fractional division calculations configuration metric coefficients pricing multiplier setups variables variables options maps values models framework definitions trace options models processing definitions data code flow.
    return (token_count / 1_000_000) * 0.02


# ── Single Run Pipeline ───────────────────────────────────────────────────────


def run_single(
    source: str,
    strategy: str,
    store: SkyLexVectorStore,
    embedder: OpenAIEmbeddings,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """
    Ek source × strategy combination ke liye complete pipeline run karo.

    Args:
        source        : FAA_CFR, FAA_AD, etc.
        strategy      : recursive, hierarchical, etc.
        store         : Initialized SkyLexVectorStore
        embedder      : OpenAI embeddings client (LangChain wrapper)
        force_reindex : True → existing collection wipe karke fresh index karo

    Returns:
        Metrics dict — MLflow logging ke liye.
    """
    # Composite string names generation templates formatting execution values variables definitions parameter trace path indicators.
    run_label = f"{source} × {strategy}"
    logger.info("Pipeline start | %s", run_label)
    # Hardware resolution precise delta duration chronometer mark execution pipeline baseline initialization tracker path code blocks configuration logic trace tracking.
    pipeline_start = time.perf_counter()

    # Force reindex — existing collection data delete karo.
    # Safe operation verification condition evaluations clear directives parameters triggers erasure methods call tracking line metrics parameters option constraints checks.
    if force_reindex:
        logger.warning("Force reindex | %s — deleting existing data", run_label)
        store.delete_collection(source, strategy)

    # ── Step 1: Documents load ────────────────────────────────────────────────
    # Core ingestion routine execution values array return context dictionary loading components parameters logic mapping validation layout check code flow operations parameters check lines conditions variable block traces.
    docs = _load_documents(source)
    if not docs:
        logger.error("No documents found | %s — skipping", run_label)
        return {
            "source": source, "strategy": strategy,
            "status": "skipped", "doc_count": 0,
        }

    # ── Step 2: Chunking ──────────────────────────────────────────────────────
    # Selected factory mapping rules parsing initialization instance method pointers execution properties data elements structures loop path execution configurations lookup validation parameter string properties.
    chunker = _get_chunker(source, strategy)
    logger.info(
        "Chunking | %s | chunker=%s | docs=%d",
        run_label, chunker.strategy_name, len(docs),
    )

    # Algorithmic transformation intervals duration mathematical calculations delta metrics float assignments value properties mapping configurations setups logic flow variables fields.
    chunk_start = time.perf_counter()
    chunks: list[ChunkedDocument] = chunker.chunk(docs, deduplicate=True)
    chunk_time = time.perf_counter() - chunk_start

    # Console updates terminal notifications string trace messages text info view dashboard elements options tracking parameters properties details display checks map loop parameters mapping properties validation context text execution reporting summary.
    logger.info(
        "Chunking done | %s | chunks=%d | time=%.2fs",
        run_label, len(chunks), chunk_time,
    )

    if not chunks:
        logger.error("No chunks produced | %s — skipping", run_label)
        return {
            "source": source, "strategy": strategy,
            "status": "no_chunks", "doc_count": len(docs),
        }

    # ── Step 3: Metadata prepare ──────────────────────────────────────────────
    # Extraction indexing loops list array comprehensions text content sequences maps variables dictionary packaging format primitive types structure configurations setups loop data dictionary parameter.
    chunk_ids = [c.chunk_id for c in chunks]
    texts     = [c.content for c in chunks]
    metadatas = [
        {
            "source":       c.source,
            "doc_id":       c.doc_id,
            "chunk_index":  c.chunk_index,
            "total_chunks": c.total_chunks,
            "strategy":     c.strategy,
            # ChromaDB sirf primitive types accept karta hai — str cast zaroori hai.
            # Inline parameter string conversion mapping iteration unpack metadata details payload configuration mapping loop value check structures context block lines values fields validation mapping parameters configurations format setups list logic.
            **{k: str(v) for k, v in c.metadata.items()},
        }
        for c in chunks
    ]

    # ── Step 4: Pre-embedding cost estimate log karo ──────────────────────────
    # Predictive statistical functions calls parameters pass arithmetic formulas outputs results variables definitions assignment float mapping parameters context limits tracking.
    token_estimate = _estimate_tokens(texts)
    cost_estimate  = _estimate_cost_usd(token_estimate)
    logger.info(
        "Embedding estimate | %s | tokens~=%d | cost~=$%.4f",
        run_label, token_estimate, cost_estimate,
    )

    # ── Step 5: Embed karo ───────────────────────────────────────────────────
    # embedder parameter signature mein hai lekin internally _openai_client use hota hai.
    # Yeh LangSmith @traceable ke saath actual token tracking enable karta hai.
    # Vector arrays computation remote API endpoints triggers delta duration performance execution loops mapping structures variables.
    embed_start = time.perf_counter()
    embeddings  = _embed_in_batches(texts, embedder)
    embed_time  = time.perf_counter() - embed_start

    # Notice updates print text parameters string values numbers terminal options reporting layouts parameters details framework parameters format variables mappings code lines metrics indicators setups text string lines.
    logger.info(
        "Embedding done | %s | chunks=%d | time=%.2fs",
        run_label, len(embeddings), embed_time,
    )

    # ── Step 6: ChromaDB mein store karo ─────────────────────────────────────
    # Method payload routing local database transactions array storage mapping index code configurations data pointers parameters validation properties options values loop configuration mapping index properties details updates variables framework flow properties.
    store_start   = time.perf_counter()
    chunks_stored = store.add_chunks(
        chunk_ids  = chunk_ids,
        documents  = texts,
        embeddings = embeddings,
        metadatas  = metadatas,
        source     = source,
        strategy   = strategy,
    )
    store_time = time.perf_counter() - store_start
    total_time = time.perf_counter() - pipeline_start

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Global tracking evaluations dataset dictionary configuration memory map parameters output packaging schema context values data loop configurations mappings values matrix layout data values framework lists models maps definitions parameters logic variables array return.
    metrics: dict[str, Any] = {
        "source":          source,
        "strategy":        strategy,
        "status":          "success",
        "doc_count":       len(docs),
        "chunk_count":     len(chunks),
        "chunks_stored":   chunks_stored,
        "token_estimate":  token_estimate,
        "cost_usd":        cost_estimate,
        "chunk_time_s":    round(chunk_time, 3),
        "embed_time_s":    round(embed_time, 3),
        "store_time_s":    round(store_time, 3),
        "total_time_s":    round(total_time, 3),
        "avg_chunk_chars": round(sum(len(t) for t in texts) / len(texts), 1),
    }

    # Consolidated summaries text traces visibility print terminal views details mapping properties configurations values mappings loops check tracking code block logic.
    logger.info("Pipeline complete | %s | %s", run_label, metrics)
    return metrics


# ── Full Pipeline ─────────────────────────────────────────────────────────────


def run_full_pipeline(
    sources: list[str] | None = None,
    strategies_override: dict[str, list[str]] | None = None,
    force_reindex: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Saare source × strategy combinations ke liye embedding pipeline run karo.
    MLflow mein har combination ka alag nested run log hoga.

    Args:
        sources             : Specific sources — None matlab saare 5 sources
        strategies_override : Per-source strategy list override
                              e.g. {"FAA_CFR": ["recursive"]} — sirf recursive chalao
        force_reindex       : True → sab fresh se index karo

    Returns:
        Dict mapping "source__strategy" → metrics
    """
    # Fallback configuration default list options validation paths routing map data pointer parameters value initialization assignment.
    target_sources = sources or list(CANDIDATE_STRATEGIES.keys())

    # Embedder — ek baar initialize, saare runs ke liye reuse.
    # _embed_in_batches internally _openai_client use karta hai actual tracking ke liye.
    # Client module instantiation architecture parameters passing settings metadata validation rules path execution variables check data limits metrics pointer tracking model parameters data loop rules.
    embedder = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=SecretStr(settings.openai_api_key),  # type: ignore[call-arg]
    )

    # Local persistent directory storage initialization baseline database mappings logic trace parameters properties check limits configuration map variables options validation.
    store = SkyLexVectorStore()

    # Remote telemetry server tracker paths configurations routing index values setup parameter directory values definitions tracking memory limits pointer parameter models trace mapping structures loops variables tracking paths options mapping loops variables context parameters code.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    # Aggregated performance totals accumulators matrices variables records parameters definition layout dictionary setups memory assignments mapping block logic tracking configuration.
    all_metrics: dict[str, dict[str, Any]] = {}
    total_cost   = 0.0
    total_chunks = 0

    # Master pipeline session control environment manager initialization parameters context scope loop trace string mappings index framework variable options context loops mapping structure.
    with mlflow.start_run(run_name="skylex_phase3_embedding"):

        # Parent run — global pipeline params log karo.
        # Direct telemetry dictionary variables tracking uploads backend server maps variables metadata parameters metrics trace.
        mlflow.log_params({
            "embedding_model":  settings.openai_embedding_model,
            "embed_batch_size": _EMBED_BATCH_SIZE,
            "sources":          ",".join(target_sources),
            "force_reindex":    force_reindex,
        })

        # Main orchestration loop sequence elements iterations context mapping arrays definitions configurations mapping loop code logic tracking path values loops mapping checks tracking.
        for source in target_sources:
            # Source ke liye strategies determine karo.
            # Local evaluation parameter lookup criteria validation condition loops dynamic choices assignments parameters options block context values logic variables tracking array logic tracking framework loops options path trace values fields check variables parameters definitions list logic configurations parameters.
            strategies = (
                strategies_override.get(source, CANDIDATE_STRATEGIES[source])
                if strategies_override
                else CANDIDATE_STRATEGIES[source]
            )

            # Strategy subsets processing sequence combinations mappings strings parameters loops configuration definitions variable data context framework.
            for strategy in strategies:
                run_key = f"{source}__{strategy}"

                # Har source × strategy ka alag nested MLflow run.
                # Dynamic isolated child session manager initialization parameters context variables parameters setup layout pointer index variables tracking framework configurations parameters.
                with mlflow.start_run(
                    run_name=f"embed_{source.lower()}_{strategy}",
                    nested=True,
                ):
                    # Subroutine logic metrics results dictionary execution passing parameters arguments configurations loop context arrays mapping trace validation.
                    metrics = run_single(
                        source        = source,
                        strategy      = strategy,
                        store         = store,
                        embedder      = embedder,
                        force_reindex = force_reindex,
                    )

                    # Matrix assignments updates dictionary array properties mapping mapping paths lists values definitions parameters.
                    all_metrics[run_key] = metrics

                    # Error handling logic skip configuration check parameters flag mapping context execution tracker data path.
                    if metrics["status"] == "success":
                        mlflow.log_metrics({
                            "chunk_count":     metrics["chunk_count"],
                            "token_estimate":  metrics["token_estimate"],
                            "cost_usd":        metrics["cost_usd"],
                            "total_time_s":    metrics["total_time_s"],
                            "avg_chunk_chars": metrics["avg_chunk_chars"],
                        })
                        mlflow.log_params({
                            "source":    source,
                            "strategy":  strategy,
                            "doc_count": metrics["doc_count"],
                        })

                        # Numerical totals tracking increment calculation math operations context parameters configuration updates value definitions variable parameters checks loop code trace properties variables tracking trace.
                        total_cost   += metrics["cost_usd"]
                        total_chunks += metrics["chunk_count"]

        # Parent run — totals log karo.
        # Consolidated matrices upload transmission parameter parameters map data setup context values loops metrics path loop code layer framework tracking structures updates variables lists loop fields options mapping variables mappings index properties parameters checks.
        mlflow.log_metrics({
            "total_chunks":   total_chunks,
            "total_cost_usd": total_cost,
        })

    # Terminal summary print logging text message updates parameter details format values tracing parameters execution report lines print framework logic mapping values variables parameters.
    logger.info(
        "Full pipeline complete | total_chunks=%d | total_cost=$%.4f",
        total_chunks, total_cost,
    )

    # Database reporting method execution call configurations layouts display context mappings formats parameters string configurations matrix views.
    store.print_stats()
    return all_metrics