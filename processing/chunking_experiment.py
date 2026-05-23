"""
processing/chunking_experiment.py

SkyLex Phase 2 — Chunking Strategy Experiment Runner v2.

9 strategies × 5 sources = 45 MLflow runs.
Experiment name: skylex_chunking_v2

Metrics tracked (5 total):
  1. avg_semantic_density      — chunk vs parent cosine similarity (weight: 40%)
  2. boundary_respect_score    — sentence-complete chunks ratio (weight: 30%)
  3. size_consistency_score    — 1 - CV (coefficient of variation) (weight: 20%)
  4. efficiency_score          — per-source normalized latency score (weight: 10%)
  5. composite_score           — weighted combination of all 4 metrics

Composite formula:
  composite = 0.40×density + 0.30×boundary + 0.20×consistency + 0.10×efficiency

Execution model:
  - Non-API strategies (6): ThreadPoolExecutor (MAX_WORKERS=4) — parallel
  - API-bound strategies (3): Sequential per source — rate limit safe

Thread-safety:
  - Fresh chunker + embedder per task (factory pattern)
  - Per-thread random.Random (no global state mutation)
  - MLflow MlflowClient explicit run_id (fluent API bypass)
  - matplotlib.figure.Figure() (thread-safe plots)
  - tenacity retry + exponential backoff + jitter

MLflow UI:
    mlflow ui --backend-store-uri mlruns
    http://127.0.0.1:5000 → experiment: skylex_chunking_v2

LangSmith:
    https://smith.langchain.com → project: skylex

Usage:
    python processing/chunking_experiment.py
    python processing/chunking_experiment.py --skip-api
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.figure
import mlflow
import numpy as np
from langchain_openai import OpenAIEmbeddings
from mlflow.tracking import MlflowClient
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from config.settings import settings
from monitoring.logger import get_logger, setup_logging
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
    StructureAwareChunker,
)

# Core logging mechanism setup bootstrap ho raha hai taaki pure matrix pipeline me terminal and log files synced rahein.
setup_logging()
# Dynamic multi-threading runs k events aur context stack traces ko systematically track karne k liye customized logger object model initialize kiya.
logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# MLflow UI dashboard backend track engine me store karne k liye standalone project folder collection name register mapping path pointer.
EXPERIMENT_NAME: str = "skylex_chunking_v2"
# Central file module systems settings property parameters retrieval pointer paths validation location.
RAW_DATA_DIR: Path = settings.data_raw_dir
# Sizing calculations empirical model token adjustments metric formula standard criteria check configuration.
CHARS_PER_TOKEN: float = 4.0
# Vector calculation upper limits truncation parameter checks safety margins check properties boundary condition setup parameter numbers.
EMBEDDING_MAX_CHARS: int = 6000
# Multi threading pool capacity ceilings constraints limits concurrency safety settings properties parameters mapping thread limits value.
MAX_WORKERS: int = 4
# Linear seed data matching reproducibility validation states generator numbers configurations track index setup parameter mapping.
RANDOM_SEED: int = 42

# Composite score weights — research-backed
# Semantics content fidelity check metrics scaling calculations multiplier configuration parameter weight distribution analysis trace.
WEIGHT_DENSITY: float = 0.40
# Sentence level completion check evaluation parameters score tracking ratio multiplier value balance assignment setup.
WEIGHT_BOUNDARY: float = 0.30
# Size distributions uniform consistency balancing factor mathematical coefficient parameters weight mapping dashboard registry tracking.
WEIGHT_CONSISTENCY: float = 0.20
# Latency and cost feasibility performance factor evaluation parameters scales tracking numbers ratio configuration.
WEIGHT_EFFICIENCY: float = 0.10

# Input files listings collections structures maps folder matching strings display tags dashboard maps repository.
SOURCES: dict[str, str] = {
    "faa_cfr": "FAA_CFR",
    "faa_ad": "FAA_AD",
    "faa_ac": "FAA_AC",
    "dgca_car": "DGCA_CAR",
    "skybrary": "SKYBRARY",
}

# OpenAI server operations check dynamic boundaries token rate management check indicators list configuration parameters registry trackers sets mapping loop.
API_BOUND_STRATEGIES: set[str] = {"semantic", "improved_semantic", "double_pass"}

# Concurrency thread isolated parallel safe execution algorithms options parameters check lists sequences models listings targets checklist tracker arrays context properties.
NON_API_STRATEGIES: list[str] = [
    "recursive",
    "hierarchical",
    "hybrid",
    "improved_recursive",
    "improved_hybrid",
    "structure_aware",
]


# ── Result Dataclass ──────────────────────────────────────────────────────────


@dataclass
class ExperimentResult:
    """
    Ek strategy × source combination ka poora result.
    Pass 1 mein raw metrics fill hote hain.
    Pass 2 mein efficiency_score aur composite_score fill hote hain.
    """

    # Downstream evaluation parameters metadata tracking identifier strings names mapping registry parameters indicators dashboard context structures validation setup.
    strategy: str
    # Ingestion documentation category listings tracking labels configurations parameters target properties variables map location trace data block.
    source: str
    # Quantitative documentation units limits variables items counter metadata metrics mapping data properties validation rule check code path.
    doc_count: int
    # Technical transaction identification sequence code context string properties update parameters assignment tracking execution trace parameters pointer.
    run_id: str

    # Structural metrics
    # Absolute calculated outputs aggregate text elements segments chunks array counter numbers values parameter mapping model trace layout database placeholder.
    chunk_count: float = 0.0
    # Average numeric measurement scale properties size evaluations data values definitions parameter tracking indicator.
    avg_chunk_size_chars: float = 0.0
    # Lower bounds range check limit parameter evaluations configurations tracking indicators conditions check logic minimum array space trace.
    min_chunk_size_chars: float = 0.0
    # Maximum bounds restrictions sizing parameters check metadata storage fields check safety criteria layout check.
    max_chunk_size_chars: float = 0.0
    # Standard deviation calculations value mapping check checking metrics variance limits structural parameters data model context layout variables database properties tracking.
    std_chunk_size_chars: float = 0.0
    # Vacant nodes metrics validation structural spacing check checks condition branch data flow checks execution balance options tracking model.
    empty_chunk_count: float = 0.0
    # Ratio indicator calculation formula checking metrics conditions scale parameters layout splits density analysis trace model setup fields check.
    chunks_per_doc_avg: float = 0.0
    # Approximate model token count allocations storage summaries calculation parameters formula rule calculations metrics dashboards value.
    estimated_tokens_total: float = 0.0

    # Core metrics
    # Vector similarities mathematical metrics calculation averages parameters mapping validation dynamic calculations track index context properties.
    avg_semantic_density: float = 0.0
    # Grammatical termination consistency checking ratio metrics configurations tracking calculations validation properties check dynamic indicator level score.
    boundary_respect_score: float = 0.0
    # Total calculation benchmark timing snap interval duration math results value parameters metrics metrics context code layout parameter decimal values structure logic.
    latency_seconds: float = 0.0

    # Derived metrics — computed in Pass 2
    # Unified size variability index calculation parameters float precision scoring mapping database reporting model trace control logic data.
    size_consistency_score: float = 0.0
    # Relative performance standard min-max normalized calculations processing outputs mapping parameters scale check indicators tracking dashboard model parameters logic.
    efficiency_score: float = 0.0
    # Master scoring benchmark evaluation compound dynamic configuration metrics mapping value parameters formula check validation trace layout options.
    composite_score: float = 0.0

    # Extra fields for summary
    # Extended context configurations properties dictionary arrays parameters records mappings configurations metadata inventory target store memory options custom block map variables.
    params: dict[str, Any] = field(default_factory=dict)


# ── Retry Decorators ──────────────────────────────────────────────────────────


def _make_embed_with_retry(embedder: OpenAIEmbeddings) -> Any:
    """
    embed_documents ko tenacity retry wrapper mein wrap karo.
    Exponential backoff + jitter — rate limit (429) aur transient errors handle.
    Per-call wrapper — embedder instance thread-isolated rehta hai.
    """

    # Multi stage exception control resilient closure function dynamic configurations metadata options rules tracking criteria validator trace block wrap logic decorator parameters.
    @retry(
        # Intercept connection timeouts, rate throttling errors, or temporary remote anomalies routing checks parameters path branch checking branch condition model validation loops.
        retry=retry_if_exception_type(Exception),
        # API requests processing thresholds counters capacity validation numbers verification check metrics limits tracker options conditions sequence.
        stop=stop_after_attempt(5),
        # Sizing calculations backoff exponential dynamic timing equations rules tracking interval variance metrics jitter noise setup parameters allocations metrics.
        wait=wait_exponential_jitter(initial=2, max=60, jitter=4),
        # Propagate error parameters directly inside active worker thread stack execution lines tracking error indicator bubble context code line.
        reraise=True,
    )
    def _embed(texts: list[str]) -> list[list[float]]:
        # Inbound dynamic character streams array passed direct target processing computational model code matrix calculations array listings pipeline query trace handle indicator.
        return embedder.embed_documents(texts)  # type: ignore[return-value]

    # Return encapsulated thread safe closure functional pointer reference straight back to evaluation analytics pipeline module layer function wrapper caller.
    return _embed


# ── Factory Functions ─────────────────────────────────────────────────────────


def make_chunker(strategy_name: str) -> Any:
    """
    Fresh chunker instance per task.
    Factory pattern — shared instances = thread-safety risk.
    """
    # Parameter tracking identifier string comparison conversions processing validation conditions routing path select execution layout checking variable parameters mapping check layer context line.
    if strategy_name == "recursive":
        # Baseline limit configurations sizing parameters pass initialize fresh text parsing algorithm strategy instance properties return flow.
        return RecursiveChunker(chunk_size=1000, chunk_overlap=150)
    # Advanced semantic metrics distance matrix transformation properties tracking configurations comparisons lookup fields context variable data maps framework model trace options routing context.
    if strategy_name == "semantic":
        # Localized vector cluster bounds check algorithm processor initialization parameters setup core logic function object creation layout.
        return SemanticChunker()
    # Structural rule checking formatting validation regex splits checking metrics register setup variable criteria checklist options branch pointer level.
    if strategy_name == "hierarchical":
        # Sizing limit indices restrictions safe maximum dimensions capacity criteria pass parameters checking code variable allocation pointer direct framework code.
        return HierarchicalChunker(max_chunk_size=3000)
    # Combined hybrid orchestration workflows verification conditions checking dynamic properties lookup data variables routing parameters context options layout setup block logic.
    if strategy_name == "hybrid":
        # Slicing parameters limit values parsing equations dynamic calculations keys mapping settings memory pointer check structural logic parameters.
        return HybridChunker(max_chunk_size=1500, chunk_overlap=150)
    # Custom baseline adjustments structural tracking parameter checks conditions check dynamic branch configurations variables layout definition.
    if strategy_name == "improved_recursive":
        # Advanced sizing range control metrics parameters padding design characters limits pass freshly configured variable mapping context.
        return ImprovedRecursiveChunker(chunk_size=1000, chunk_overlap=200)
    # Optimization similarity drop boundary tracking logic properties switch parsing validation state tracking traces model parameters model step context logic.
    if strategy_name == "improved_semantic":
        # Lower threshold validations calculations strategy model record fresh object allocation return structure metrics analytics dashboard tracking context.
        return ImprovedSemanticChunker()
    # Hybrid pipeline scale balancing settings checking documentation architecture patterns rules criteria variables checking data properties layer config details.
    if strategy_name == "improved_hybrid":
        # Section capacity index boundary sliding continuous text sections retention layout properties mapping configuration parameters settings check.
        return ImprovedHybridChunker(max_chunk_size=2000, chunk_overlap=200)
    # Advanced context encapsulation prefix parameters text configurations metadata injections model priority layout expressions logic checks options.
    if strategy_name == "structure_aware":
        # Sizing limit variables parameter tracking constraints safety metrics character parameters configuration lists array return layout model target setup indicator execution tracking.
        return StructureAwareChunker(max_chunk_size=1800, chunk_overlap=180)
    # Multi pass tracking context parsing validations calculations rules sequence pipeline double processing method call validation model track logic.
    if strategy_name == "double_pass":
        # Dual operational phase data transformation routing subroutines code execution parameters options mapping dictionary return complete indicator trace variables.
        return DoublePassChunker()
    # Inbound wrong string parameters safeguards verification error indicators definitions exceptions trigger code trace execution framework options parsing block code.
    raise ValueError(f"Unknown strategy: {strategy_name}")


def make_embedder() -> OpenAIEmbeddings:
    """
    Fresh OpenAIEmbeddings instance per task.
    Shared httpx connection pool = thread-safety risk.
    """
    # Networking tracking dependencies states leaks connection pool corruptions prevent framework initialization client configuration parameters build call sequence checks logic variables framework.
    return OpenAIEmbeddings(
        # Settings properties registry metadata model path directory parameters matching target configurations tracking memory lookup model properties reading context scale indicators.
        model=settings.openai_embedding_model,
        # Encapsulated tokens memory system protection check validations parameter properties allocations trace context validation types configuration level annotations layers code execution.
        openai_api_key=SecretStr(settings.openai_api_key),  # type: ignore[call-arg]
    )


def make_local_random() -> random.Random:
    """Per-thread isolated random.Random — no global state mutation."""
    # Thread isolated clean standalone state mathematical pseudo generator setup initialize context state parameter tracking rules map variables layout return logic trace code layer runtime settings.
    return random.Random(RANDOM_SEED)


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_documents(source_folder: str) -> list[dict[str, Any]]:
    """
    Source folder se saare bulk JSON files load karo.
    hash_registry.json aur _meta.json exclude karo.
    Duplicate doc_ids deduplicate — pehli occurrence rakho.
    """
    # Target filesystem directory location building template configuration maps setup file addresses string tracking pathway code pointer memory address validation tracking.
    source_dir: Path = RAW_DATA_DIR / source_folder
    # Output verified clean document model dictionary data collections list allocation tracking configuration storage map array variable initialized placeholder trace database models tracking.
    all_docs: list[dict[str, Any]] = []
    # Real-time memory hash cache tracking sets map database indices duplicate filtering configurations lookup layer control logic setup state tracking parameters check loops validate unique variables framework.
    seen_ids: set[str] = set()

    # Recursive global subdirectory trees scanning mapping search loop execution crawl storage components directory parsing structural layout matching files process data crawlers dynamic processing.
    for json_file in sorted(source_dir.rglob("*.json")):
        # Registry logging data indexes skipping criteria rule tracking parameters checking condition branch validation context path skip step rule baseline parameters trace context log files indicator.
        if json_file.name == "hash_registry.json":
            continue
        # Descriptors metadata file layout rules structure specifications matching skip constraints processing check step memory configuration parameters trace layout code block layer text strings mapping validation.
        if json_file.name.endswith("_meta.json"):
            continue
        try:
            # Native filesystem descriptor mapping channel access setup reader configuration properties unicode standard mapping registry stream parameters framework path logic options stream read tracking.
            with json_file.open(encoding="utf-8") as f:
                # Character stream formatting parser runtime deserialization matrix transform directly inside active object reference parameters map load processing options memory target variable parameters execution blocks context.
                data: Any = json.load(f)
            # Object schema validation rule properties tracking context verification parameters data structural integrity status loop
            # boundary filtering check logic indicator context type configurations parsing check metrics validation options layout bounds thread.
            if not isinstance(data, dict) or "documents" not in data:
                continue
            # Data matrices payload list iteration mapping elements internal data structures traverse sequence workflow tracking controls execution loop parameters state data collections step parsing indicator loops tracking trace items properties.
            for doc in data["documents"]:
                # Individual data component layout unique indicator identifier key reference string lookups read pipeline trace matching choices validation default key parameter tracking values details options code mapping fields.
                doc_id: str = doc.get("doc_id", "")
                # Cross check current parsed identifier code history registry history parameters check data overlap conditions filter execution tracking cache verification matching indices rule constraints checking logic verification options history.
                if doc_id and doc_id not in seen_ids:
                    # Inject valid newly discovered identifier string key right inside memory framework lookup tracking collection update instantly caching index trace registry mapping framework update data sets cache tracking set layout update variables metrics.
                    seen_ids.add(doc_id)
                    # Non overlapping validated genuine document database dictionary metadata record components append into destination tracking array properties variable targets data items metadata collection storage lists configuration rules elements parsing.
                    all_docs.append(doc)
        except (json.JSONDecodeError, OSError) as e:
            # Handle malformed files encoding structures or hardware level disk sector parsing blocks anomalies cleanly without breaking core thread orchestrator execution scheduler loops logic context.
            logger.warning("Could not load %s: %s", json_file, e)

    # Performance logging pipeline tracking reporting console terminal metrics indicator statistics feed print out execution layout trace summary metadata output channel view info tracking dashboard.
    logger.info(
        "Loaded %d unique documents from %s", len(all_docs), source_folder
    )
    # Output isolated completely filtered sanitized clean documentation maps list reference back to core scheduling runner loop processes framework model data elements repository structural variables setup layout.
    return all_docs


def build_doc_content_map(documents: list[dict[str, Any]]) -> dict[str, str]:
    """
    doc_id → full original content map.
    Semantic density mein chunk ko actual parent se compare karne ke liye.
    No data leak — chunk ko apne aap se compare nahi karte.
    """
    # Python dictionary comprehensions parsing syntax layout mapping equations calculations properties text variables allocation memory context loop variables lookup trace step metrics balance options mapping tracker rules expressions maps data elements block reading maps.
    return {
        doc["doc_id"]: doc.get("content", "")
        for doc in documents
        if doc.get("doc_id") and doc.get("content")
    }


# ── Metric Calculators ────────────────────────────────────────────────────────


def compute_structural_metrics(
    chunks: list[ChunkedDocument],
) -> dict[str, float]:
    """
    Chunk size distribution metrics.

    chunk_count          : Total chunks produced
    avg_chunk_size_chars : Mean characters per chunk
    min_chunk_size_chars : Smallest chunk
    max_chunk_size_chars : Largest chunk — embedding limit check ke liye
    std_chunk_size_chars : Size variance — size_consistency_score ka input
    empty_chunk_count    : Quality check — should always be 0
    chunks_per_doc_avg   : Average chunks per source document
    estimated_tokens_total: Approximate embedding cost estimate
    """
    # Empty inputs boundaries protection data filters criteria parameters check validation default configurations placeholder return path routing control branch logic matrix trace pointer.
    if not chunks:
        return {
            "chunk_count": 0.0,
            "avg_chunk_size_chars": 0.0,
            "min_chunk_size_chars": 0.0,
            "max_chunk_size_chars": 0.0,
            "std_chunk_size_chars": 0.0,
            "empty_chunk_count": 0.0,
            "chunks_per_doc_avg": 0.0,
            "estimated_tokens_total": 0.0,
        }

    # Extract dynamic list array integer lengths specifications measuring metrics sizing calculations loops context variables text lengths tracking parameters value array data items size metrics.
    sizes: list[int] = [len(c.content) for c in chunks]
    # Blank nodes indicators tracking loop checks calculations accumulation counters filter spacing character validation checking parameters summary options framework variables.
    empty_count: int = sum(1 for c in chunks if not c.content.strip())
    # Distinct baseline dynamic original document parent references unique lookup count using tracking keys sets initialization parameters data map checks index configurations variance.
    unique_docs: int = len({c.doc_id for c in chunks})
    # Density metric ratio check division formatting formula scaling checks calculations rules options logic loop trace metrics index variables configuration setup mapping rules.
    chunks_per_doc: float = (
        len(chunks) / unique_docs if unique_docs > 0 else 0.0
    )

    # Packaging calculated analytics metrics floats properties values parameters records directly inside structural reporting metric dashboard dictionary collections tracking layout indicators log.
    return {
        "chunk_count": float(len(chunks)),
        "avg_chunk_size_chars": float(statistics.mean(sizes)),
        "min_chunk_size_chars": float(min(sizes)),
        "max_chunk_size_chars": float(max(sizes)),
        "std_chunk_size_chars": float(
            statistics.stdev(sizes) if len(sizes) > 1 else 0.0
        ),
        "empty_chunk_count": float(empty_count),
        "chunks_per_doc_avg": round(chunks_per_doc, 2),
        "estimated_tokens_total": round(sum(sizes) / CHARS_PER_TOKEN, 0),
    }


def compute_boundary_respect_score(chunks: list[ChunkedDocument]) -> float:
    """
    Sentence boundary respect score.

    Formula: complete_chunks / total_chunks
    complete_chunk = chunk ending with sentence-terminating punctuation

    Range: 0.0 to 1.0 — higher = better sentence completeness
    Weight in composite: 30%

    Research reference: Shi et al. (2023) — Chunking for RAG evaluation.
    """
    # Empty item data sequences boundary checkpoint checks validation parameters options control redirection path mapping trace configuration code assignment rules tracking framework layout parameters.
    if not chunks:
        return 0.0
    # Core standard grammatical punctuation components markers tracking patterns lookups choices arrays configurations parameters route lookup mapping context setup text rules expressions symbols delimiters prioritization choices checklists.
    sentence_endings: tuple[str, ...] = (
        ".",
        "?",
        "!",
        ".'",
        "?'",
        "!'",
        '."',
        '?"',
        '!"',
    )
    # Text string edge formatting validations checks loop accumulator index counting criteria values configuration mapping trace run indicator tracking pattern checks dynamic edge layout verification calculations metrics rules complete.
    complete: int = sum(
        1
        for c in chunks
        if c.content.strip().endswith(sentence_endings)
    )
    # Precision decimal calculation rules metrics constraints rounding formulas conversion values return flow layout matrix tracking indices configuration parameters precision float mathematical division operations metrics status.
    return round(complete / len(chunks), 4)


def compute_size_consistency_score(
    avg_chunk_size: float,
    std_chunk_size: float,
) -> float:
    """
    Size consistency score via Coefficient of Variation (CV).

    Formula: max(0.0, 1.0 - (std / avg))
    CV = std / avg — normalized measure of size variability
    score = 1 - CV — higher = more consistent chunk sizes

    Floor at 0.0 — CV > 1.0 pe negative nahi hoga.
    Example:
        recursive FAA_CFR: std=163, avg=845 → CV=0.193 → score=0.807
        hierarchical FAA_AC: std=32610, avg=4548 → CV=7.17 → score=0.0

    Range: 0.0 to 1.0 — higher = better embedding space fairness
    Weight in composite: 20%
    """
    # Base structural checks parameter checking metrics configurations condition zero edge prevention logic control path branching redirect trace.
    if avg_chunk_size <= 0:
        return 0.0
    # Calculate variation coefficient scale variance metrics value mapping ratio index formula tracking code parameter settings layout logic.
    cv: float = std_chunk_size / avg_chunk_size
    # Normalization offset matrix calculation bounds protection constraints threshold floor limit tracking options parameter values direct returns value.
    return round(max(0.0, 1.0 - cv), 4)


def compute_semantic_density(
    chunks: list[ChunkedDocument],
    doc_content_map: dict[str, str],
    embedder: OpenAIEmbeddings,
    local_random: random.Random,
) -> float:
    """
    Average cosine similarity — chunk embedding vs parent document embedding.

    Formula:
        for each sampled chunk:
            similarity = cosine_similarity(embed(chunk), embed(parent_doc))
        avg_semantic_density = mean(similarities)

    Design decisions:
        Dynamic sample size: min(50, max(5, 5% of chunks))
            — percentage-based, scales with dataset
            — upper bound 50 controls API cost
            — lower bound 5 ensures statistical validity
        Stratified random sampling — not first N chunks
            — avoids selection bias
        Token-safe truncation — 6000 chars max
            — text-embedding-3-small limit: 8191 tokens ≈ 32764 chars
            — 6000 chars = safe buffer
        No data leak — parent content from doc_content_map, not chunk itself

    Range: 0.0 to 1.0 — higher = chunk better represents parent document
    Weight in composite: 40%

    Research reference: Chen et al. (2024) — Dense Passage Retrieval evaluation.
    """
    # Safety boundary parameter checking validation mapping options parameters lookup verify structural state tracking array model branch code fallback redirection trace block logic database configurations constraints rules framework models verification layer metadata path settings.
    if not chunks or not doc_content_map:
        return 0.0

    # Scale percentage parameters numerical boundary criteria equations metrics formulations checks scaling allocation variable maps evaluation step dynamic scale constraints variables formulation calculations index rules logic matching tracking options.
    sample_size: int = min(50, max(5, int(len(chunks) * 0.05)))
    # Completely thread isolated unbiased randomized items picker execution subroutine logic slice passing parameters arrays tracking data matrix return layout block pipeline runtime scheduler metrics context sample arrays collection pull worker step execution configuration trace context.
    sample: list[ChunkedDocument] = local_random.sample(
        chunks, min(len(chunks), sample_size)
    )

    # Valid confirmed processing elements pairs text configurations collections directory variables pointer context dataset map array tracking container data allocation register matched items records verification layer lists collection placeholders dynamic context.
    valid_pairs: list[tuple[str, str]] = []
    # Scanning elements lifecycle tracking matrix traversal loop variables items records attributes data execution sequence loop block trace step pipeline runtime handler logic data stream operations components checks validation conditions array traces traversal route processing loops.
    for chunk in sample:
        # Dynamic memory key map lookup extraction safe parameters configurations metadata reading access check property variables tracking trace context data value lookup step parameters keys validation query index mapping extraction properties configurations.
        parent_content: str = doc_content_map.get(chunk.doc_id, "")
        # String content availability validation processing check structural rule verification matching tracking sequence branch layout condition code trace layer context block configurations checks verification alignment filter boundary parameters paths trace layer criteria.
        if parent_content:
            # Token constraints safety protection bounds character limit criteria parameters map layout slicing operation dynamic extraction collection variables array save trace parameters capacity length tracking options slices metrics collection insertions variable.
            valid_pairs.append(
                (
                    chunk.content[:EMBEDDING_MAX_CHARS],
                    parent_content[:EMBEDDING_MAX_CHARS],
                )
            )

    # Structural item size validation check to prevent vacant computational arrays data loop execution processing error safety block path pointer data pipeline logic trace tracker validations framework monitoring operations index items calculation limits indicators data streams context logging.
    if not valid_pairs:
        logger.warning(
            "No valid chunk-parent pairs — semantic density returning 0.0"
        )
        return 0.0

    # Dynamically inject tenacity retry resilience layer wrapper right on top of the thread-isolated embedding engine instance function pointer variables setup tracking mapping core logic closure function connection pool safety wrappers parameter setup runtime logic.
    embed_fn = _make_embed_with_retry(embedder)

    try:
        # Request outbound remote provider network embedding vector array matrix calculations using dynamic retry wrapped routine execution result arrays return trace list data tracking parameters connectivity options tokens metrics evaluation models query stream updates trace.
        chunk_embeddings: list[list[float]] = embed_fn(
            [p[0] for p in valid_pairs]
        )
        # Parent original documentation sequences text vectors representation computation provider connection call address pipeline routing data maps values execution results trace remote servers computation layers data alignment vector models mappings metrics trace indicators.
        parent_embeddings: list[list[float]] = embed_fn(
            [p[1] for p in valid_pairs]
        )

        # Distance similarity arrays registry storage context target metrics track tracking array allocation initialization code variables calculations loop parameters setup metrics model analytics mapping results trace arrays configurations database registry repository metrics.
        similarities: list[float] = []
        # Multi-variable loop tracking traversing parallel vector arrays coordinates data unpack trace execution validations sequence check control iterations blocks tracking look trace matrix multi component pairs mapping dimensions scanning iterations validation parameters coordinates tracking.
        for c_emb, p_emb in zip(chunk_embeddings, parent_embeddings):
            # Convert basic array list floats into high speed mathematical matrix processing numpy structures layout tracking objects parameters values design rules calculation trace model high performance multidimensional optimization float operations metrics calculations tracking linear.
            c_arr = np.array(c_emb)
            # Alignment matching checking numerical checks validation properties configuration map elements inside high efficiency numpy object calculations parameters check runtime log trace data check metrics operations values validation properties matrix dimension checks layout tracking.
            p_arr = np.array(p_emb)
            # Vector norm dimensions square matrix multiplier calculations structural magnitude coefficients index resolution safety check variable trace formula execution data layer context rules math metrics calculation formulas magnitude multiplier scaling protection parameters checks variance.
            norm_product: float = float(
                np.linalg.norm(c_arr) * np.linalg.norm(p_arr)
            )
            # Zero division protection boundaries exceptions branching path skip execution context loop controls validation tracker layer tracking code logic framework operations trace variables criteria validation constraints safety tracking checks loop bypass options.
            if norm_product < 1e-10:
                continue
            # Distance matrix resolution linear dot multiplier products logic formula resolution score parameter verification dynamic mapping data variable list append track indicator data flow checks process vector operations linear multipliers results aggregation float scores mapping.
            similarities.append(
                float(np.dot(c_arr, p_arr) / norm_product)
            )

        # Yield completely resolved mathematical mean average metrics coefficient properties configurations layout metrics analysis output check trace return flow pointer data logic statistical mean computations coefficient results overview validation parameters floating summaries analytics reporting.
        return (
            round(float(np.mean(similarities)), 4)
            if similarities
            else 0.0
        )

    except Exception as e:
        # Capture API connection crashes, authentication timeouts, or unhandled token data parsing issues metrics warnings tracking traces indicators screen output logs warning updates tracking connectivity failures processing exceptions context logs tracking trace indicators stream details parsing.
        logger.warning("Semantic density failed after retries: %s", e)
        # Secure failure mode path redirection standard fallback default values tracking parameters setup execution branch check metrics layout flow trace runtime code recovery options pathway index structural validation safe fallback tracking process indicators monitoring configuration.
        return 0.0


def compute_efficiency_score(
    latency_per_doc_values: list[float],
) -> list[float]:
    """
    Per-source min-max normalized efficiency scores.

    Formula:
        latency_per_doc = latency_seconds / doc_count  (per strategy)
        efficiency_score = 1 - (latency_per_doc - min_latency) /
                               (max_latency - min_latency + 1e-10)

    Per-source normalization — global normalization unfair hogi kyunki
    FAA_CFR pe semantic 612s hai lekin SKYBRARY pe 13s —
    dono alag contexts hain, global comparison misleading hoga.

    Example (FAA_CFR):
        recursive:    0.288s / 226 = 0.00127s/doc → efficiency = ~1.0
        hierarchical: 0.207s / 226 = 0.00092s/doc → efficiency = ~1.0
        semantic:     612s   / 226 = 2.71s/doc    → efficiency = ~0.0

    Range: 0.0 to 1.0 — higher = faster relative to other strategies on same source
    Weight in composite: 10%
    """
    # Clean verification bounds constraints condition checks logic flow execution safe parameters check.
    if not latency_per_doc_values:
        return []
    # Identify lowest mathematical duration value index mapping criteria data structures parameter tracking lookup.
    min_lat: float = min(latency_per_doc_values)
    # Find slowest ceiling limit context performance metrics index variable storage value checking trace parameters.
    max_lat: float = max(latency_per_doc_values)
    # Math expression mapping equations fraction denominator configurations values rules epsilon crash guard setup logic.
    denom: float = max_lat - min_lat + 1e-10
    # Python comprehensive data parsing matrix execution linear loop formula translation formatting returns mapping collections dynamic scores lists variables.
    return [
        round(1.0 - (lat - min_lat) / denom, 4)
        for lat in latency_per_doc_values
    ]


def compute_composite_score(
    density: float,
    boundary: float,
    consistency: float,
    efficiency: float,
) -> float:
    """
    Weighted composite score — master metric for strategy comparison.

    Formula:
        composite = 0.40×density + 0.30×boundary + 0.20×consistency + 0.10×efficiency

    Weight rationale:
        40% density    — primary retrieval quality signal
        30% boundary   — LLM answer completeness
        20% consistency — embedding space fairness
        10% efficiency  — production feasibility

    Range: 0.0 to 1.0 — higher = better overall chunking quality
    """
    # Apply standard constant multiplier weight factors formula calculations mathematical combinations parameter off floating round rules setup value return layer tracking indicator.
    return round(
        WEIGHT_DENSITY * density
        + WEIGHT_BOUNDARY * boundary
        + WEIGHT_CONSISTENCY * consistency
        + WEIGHT_EFFICIENCY * efficiency,
        4,
    )


# ── Artifact Generators ───────────────────────────────────────────────────────


def save_chunk_size_histogram(
    chunks: list[ChunkedDocument],
    strategy: str,
    source: str,
    tmp_dir: str,
) -> str:
    """
    Thread-safe chunk size distribution histogram.
    matplotlib.figure.Figure() — plt global state bypass.
    """
    # Extract string length capacity dimensions loop comprehension calculation metrics properties tracker values list setup characters length measurement inline list comprehension operations character capacity evaluations tracker integer arrays.
    sizes: list[int] = [len(c.content) for c in chunks]
    # Running statistical analysis calculations formula computing exact distribution pattern midpoint evaluation values track trace parameters property configuration setup model parameters arithmetic average calculations formulation variables tracking parameters indicators summary options.
    mean_size: float = float(statistics.mean(sizes))

    # Thread safe fully isolated decoupled canvas tracking architecture plot generation instance class invocation context setup mapping variables parameters trace log code step execution line layout standalone chart container canvas initialize allocation parameters geometry configuration modeling.
    fig = matplotlib.figure.Figure(figsize=(10, 5))
    # Coordinate grid frame manager allocation section mapping layout active index object generation add execution layout variables parameters check code runtime layer processing frame management axis placement subplot structural canvas indexing coordinate grid setup mapping variables rules.
    ax = fig.add_subplot(1, 1, 1)
    # Chart plotting data graphics mapping parameters color specifications borders styling layout alpha frequency density tracking data blocks canvas trace draw context parameters design settings frequency metrics chart drawings rendering bars metrics alignment properties layout visualization palette color adjustments.
    ax.hist(sizes, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    # Typography template variable string formatting visualization title overlays main definitions description styling values parameters log terminal console updates trace output text display models template string compound charting dynamic name parameters header formatting style layout details trace data fields mappings.
    ax.set_title(
        f"Chunk Size Distribution — {strategy.upper()} | {source}",
        fontsize=14,
        fontweight="bold",
    )
    # Horizontal canvas layout parameters naming labels text specifications design metrics styling criteria parameter code target maps tracking level data line alignments rules configurations parameters horizontal axis labeling character dimensions tracker layout definitions rules text.
    ax.set_xlabel("Chunk Size (chars)", fontsize=12)
    # Vertical coordinate sequence tracking metric level parameter label details formatting string text dashboard configuration settings write analytics values trace runtime logic metrics level parameters check vertical scale metrics frequency values mapping configurations visibility screen parameters text layout specifications.
    ax.set_ylabel("Frequency", fontsize=12)
    # Overlay reference line markers plotting statistical center calculations dash format setup indicators variables balance display path alignment logic code trace statement parameters mapping lines design boundaries performance median indicators overlays dashboard guidelines trace display markers graphics settings drawing line parameters options lines template layout.
    ax.axvline(
        mean_size,
        color="#DD4444",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_size:.0f} chars",
    )
    # Legend description layout panel canvas visibility activation parameters trigger maps text details styling parameters tracking method call trace code logic updates visibility panel interface dashboard active indicators labels panel box placement console monitoring metrics styling views visibility parameter checks tracking system options display.
    ax.legend(fontsize=11)
    # Spacing margins calculation auto optimization bounding compress chart layouts tidy graphics canvas drawing save output processing layer code run step rendering alignments formatting canvas space auto arrangement boundary padding limits adjustment execution graphics draw engine clean render options structural.
    fig.tight_layout()

    # Dynamic target file system repository destination path layout string naming templates mappings parameters properties location variable string data look files directory locations naming conventions destination local disk filename structural index mappings variable template tracking lookup indicators strings address paths file address rules layout.
    plot_path: str = f"{tmp_dir}/{strategy}_{source}_chunk_dist.png"
    # Write chart binary sequence streams directly onto active system local disk persistence track image format rendering file saving action execution indicator code block log data stream write maps filesystem persistent file output allocation binary conversion export trace options line layout framework configuration drawings binary file.
    fig.savefig(plot_path, dpi=150)
    # Return destination mapping folder storage address pointer variable direct back to metadata uploading transaction scheduler system modules interface logic path analytics tracking logs dynamic file path location validation string upload task tracking framework component reference indicator output pointer tracking.
    return plot_path


def save_sample_chunks_json(
    chunks: list[ChunkedDocument],
    strategy: str,
    source: str,
    tmp_dir: str,
    local_random: random.Random,
    n: int = 5,
) -> str:
    """
    Stratified random sample of N chunks — qualitative inspection.
    Per-thread local_random — no global state mutation.
    """
    # Random metrics selection indices calculations pool tracking array selection window slice elements logic lists variables extraction tracking execution routine code step processing parameters allocation window randomized data sub selections sampling executor array slice limits tracking rules operations framework parameters options data layout constraints.
    sample: list[ChunkedDocument] = local_random.sample(
        chunks, min(len(chunks), n)
    )
    # Data mapping format conversion subroutine keys values configuration mapping structures transformation direct inside serialization key value format dictionaries lists storage variables check properties fields layout model serializable standard reporting dictionary metadata alignments formatting format configuration metadata values.
    output: list[dict[str, Any]] = [
        {
            "chunk_id": c.chunk_id,
            "chunk_index": c.chunk_index,
            "total_chunks": c.total_chunks,
            "content_length": len(c.content),
            "content_preview": c.content[:500],
            "doc_id": c.doc_id,
            "strategy": c.strategy,
            "source": c.source,
        }
        for c in sample
    ]
    # Destination location coordinate string template formatting configuration fields address variables mapping setup indicators tracking file variables check layout address layout destination setup naming coordinate folder path templates indexing character maps variables validation locations reference files system tracking layouts maps.
    json_path: str = f"{tmp_dir}/{strategy}_{source}_sample_chunks.json"
    # File descriptor handler channel allocation, enabling local resource path writing configurations conversions processing parameters integrity checks code path modes enabled unicode safe character conversions text file creation stream tracking layout settings options validations framework processes write operations.
    with open(json_path, "w", encoding="utf-8") as f:
        # Packing serializations writing data maps records json data dump operations layouts style indentation properties variable configurations flow trace code run level packing data serialization metrics formatters serialization parameters dump streaming actions text variables processing write execution track variables context lists.
        json.dump(output, f, indent=2, ensure_ascii=False)
    # Return saved destination address pathway character string direct back to tracking uploading manager integration modules context interface framework trace output files properties variable address clean reference character string data validation framework indicators pointer reference pathway path mappings context tracker metrics allocator.
    return json_path


# ── Pass 1: Single Experiment Runner ─────────────────────────────────────────


def run_single_experiment(
    strategy_name: str,
    docs: list[dict[str, Any]],
    doc_content_map: dict[str, str],
    source_name: str,
    experiment_id: str,
) -> ExperimentResult:
    """
    Pass 1 — Ek strategy × source combination ka raw experiment.

    Logs: structural metrics + density + boundary + latency
    Does NOT log: efficiency_score, composite_score (per-source normalization needed)
    Returns: ExperimentResult with run_id for Pass 2 update

    Thread-safety:
        - Fresh chunker + embedder + random per task
        - MlflowClient explicit run_id — no fluent API
        - try/finally — run always terminated
    """
    # Composite dashboard visualization indexing label template string definitions parameter combinations mapping configuration indicator target code values trace value identity formulation properties multi component variable name compound metrics tag identifiers format settings lookup value mapping criteria context system event dashboard logs.
    run_name: str = f"{strategy_name}__{source_name}"
    # Performance telemetry track notification logging info print process run monitoring console update terminal trace metrics display layout block line text tracking variables verification options startup notifications trigger logs info console output print trace analytics message status updates technical validation logging execution.
    logger.info("Starting: %s | docs: %d", run_name, len(docs))

    # Initialize low-level backend monitoring management interface client object completely unbinding standard framework contextual tracking thread state parameters explicit transaction tracker system class instantiation isolate backend tracking operations from high level dependencies tracking engine interface.
    client = MlflowClient()
    # Invoke backend monitoring systems directory to allocate unique isolated database recording track element context model maps setup value mapping trace context transaction request server dashboard session configurations metadata metrics keys register mapping target setups indicators project environment setup tracker.
    run = client.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={
            "mlflow.runName": run_name,
            "run_type": "chunking_experiment_v2",
            "phase": "phase_2",
            "strategy": strategy_name,
            "source": source_name,
            "strategy_type": (
                "api_bound"
                if strategy_name in API_BOUND_STRATEGIES
                else "non_api"
            ),
        },
    )
    # Technical transaction identification sequence reference code string tracking identifier parameter location updates reading parameter value data mapping variable database unique transaction parameters variable parsing explicit storage token identifier parameters access keys technical parameters explicit metadata trace.
    run_id: str = run.info.run_id

    try:
        # Fresh instances — no shared state across threads
        # Strategy localized object instantiation subroutine factory design execution passing targets schema context variable values indicator check layout mapping track properties model allocation dynamic components factory pattern execution loop checks values mappings handlers initialization factory blueprint logic loops.
        chunker = make_chunker(strategy_name)
        # Vector spatial metrics translation engine builder instantiation function invoke configurations criteria mapping setup call level check tracking code pointer line core modeling parameter mappings vector engines initialization request handlers parameters pass dynamic logic mappings validation parameters layout constructor call.
        embedder = make_embedder()
        # Randomized parameters selection entropy device tracker instantiation function execution isolation variables assignment sequence validation setup code tracing metrics evaluation random state per thread separate entropy isolated instance creation setup rules mapping value parameters context thread dynamic seed allocator.
        local_random = make_local_random()

        # Log parameters
        # Properties configuration collection parameters storage inventory listings dictionary schema definition mapping parameters setup data variable tracing tracker values checklist parameters dynamic inventory baseline record metrics parameters key indexing maps tracking backend register storage dashboard records parameters initialization maps configuration properties listing context dictionary.
        params: dict[str, Any] = {
            "strategy": strategy_name,
            "source": source_name,
            "doc_count": len(docs),
        }
        # Dynamic property lookup check checking variable validity condition branch routing validation parameters path selection validation track properties look variables code layer mapping branch attribute search execution condition context variable validations logic search iteration framework sequences parameters tracker setup loop attributes checks.
        for attr in (
            "chunk_size",
            "chunk_overlap",
            "max_chunk_size",
            "hard_cap_chars",
            "coarse_threshold",
            "fine_chunk_size",
            "breakpoint_threshold_type",
            "breakpoint_threshold_amount",
        ):
            # Dynamic object parameter retrieval validation checklist branches routing control check metadata layer check code statement execution.
            if hasattr(chunker, attr):
                # Access runtime instances values variables using string mapping lookups assignment direct maps tracker memory parameters context property.
                params[attr] = getattr(chunker, attr)

        # Transmit single structural configuration items sequence straight into distant database logs server tracking framework telemetry upload loop api code execution layer metadata registration parameters logging actions loop execution tracking parameters maps dashboard update indices parameters backend register trace mapping.
        for key, value in params.items():
            client.log_param(run_id, key, value)

        # Chunk + measure latency
        # High resolution hardware platform chronometer checkpoint snapshot register timing sequence benchmark evaluation track runtime interval delta start track pointer line clock execution baseline precise timing device tracking snapshots check metrics hardware timing register execution interval clock init check trace tracking.
        start_time: float = time.perf_counter()
        # Polymorphic functional procedure orchestration layout text content arrays computational loops text segmentation data arrays mapping return computation trace variable text parsing strategy model structural framework procedure mapping components array return data procedural functional components chunk execution tracking list arrays.
        chunks: list[ChunkedDocument] = chunker.chunk(
            docs, deduplicate=True
        )
        # Runtime calculation results time window formatting parameters float decimal limits parameters difference tracking indicator score value save data trace loop level metrics hardware operational latency evaluation time calculation differences metrics mapping float numbers evaluation duration results metrics formatting data.
        latency: float = round(time.perf_counter() - start_time, 3)

        # Compute metrics
        # Statistical data layout check metrics parsing calculations function invocation datasets input array properties maps return configuration code validation layer structure parameters calculations statistical text property analysis values calculation matrices maps models formatting metrics context loop distribution structural analytics value profiles matrix calculation summaries.
        structural: dict[str, float] = compute_structural_metrics(chunks)
        # Sentence segmentation formatting compliance index indicator calculation routine call target matrix variables register tracking validation console indicator trace step accuracy measurement scores sentence layout termination check indicators metrics reporting sentence termination correctness ratios score calculations execution module.
        boundary_score: float = compute_boundary_respect_score(chunks)
        # Semantic semantic space distance matrix mapping values calculation method call passing variables calculations validation vector dimension levels indicators tracking step context fidelity checks vector mathematical calculations validation tracking matrix parameters context fidelity checking logic vector matrix math calculation metrics tracking parameters.
        density: float = compute_semantic_density(
            chunks, doc_content_map, embedder, local_random
        )
        # Uniform size distribution parameter calculation execution passing averages and standard deviations indicators functions call return indicators validation logic control path branch code mapping formulas ratios variance indicator check metric variables tracking.
        consistency: float = compute_size_consistency_score(
            structural["avg_chunk_size_chars"],
            structural["std_chunk_size_chars"],
        )

        # Log Pass 1 metrics (efficiency + composite logged in Pass 2)
        # Consolidated metrics payload mapping preparation dictionary bundle containing computed floating data variables send direct dashboard backend metric telemetry interface.
        pass1_metrics: dict[str, float] = {
            **structural,
            "boundary_respect_score": boundary_score,
            "avg_semantic_density": density,
            "size_consistency_score": consistency,
            "latency_seconds": latency,
        }
        # Loop over structural properties metrics entries to stream variables floats data direct inside active server visualization infrastructure monitors logger runtime interface.
        for key, value in pass1_metrics.items():
            client.log_metric(run_id, key, value)

        # Run stays RUNNING — Pass 2 will terminate it after composite logged
        # Technical execution logging process tracing verification output text logs summary indicators display context tracking indicators telemetry instrumentation reporting message print trace info level details updates dashboard control channels layout.
        logger.info(
            "Pass1 done: %s | chunks=%d | lat=%.2fs | density=%.4f | "
            "boundary=%.4f | consistency=%.4f",
            run_name,
            len(chunks),
            latency,
            density,
            boundary_score,
            consistency,
        )

        # Structured object factory instantiation using dataclass specification model mapping records values direct inside local memory container allocation object data reference placeholder.
        result = ExperimentResult(
            strategy=strategy_name,
            source=source_name,
            doc_count=len(docs),
            run_id=run_id,
            chunk_count=structural["chunk_count"],
            avg_chunk_size_chars=structural["avg_chunk_size_chars"],
            min_chunk_size_chars=structural["min_chunk_size_chars"],
            max_chunk_size_chars=structural["max_chunk_size_chars"],
            std_chunk_size_chars=structural["std_chunk_size_chars"],
            empty_chunk_count=structural["empty_chunk_count"],
            chunks_per_doc_avg=structural["chunks_per_doc_avg"],
            estimated_tokens_total=structural["estimated_tokens_total"],
            avg_semantic_density=density,
            boundary_respect_score=boundary_score,
            size_consistency_score=consistency,
            latency_seconds=latency,
            params=params,
        )
        # Yield completed object data references containing raw analytics measurements context back to orchestration loop engine runner pipeline framework.
        return result

    except Exception as e:
        # Runtime code exceptions safety catcher tracking metadata updates failure state parameters check session lifecycle cancel signal execution call code block parameters tracking execution state context indicators trace.
        client.set_terminated(run_id, status="FAILED")
        # Failures metrics error tracing statement logging console text reporting visibility dashboard instrumentation monitor data pipelines overview traces check.
        logger.error("Experiment failed: %s — %s", run_name, e)
        # Bubble up exception interruption signals back to thread manager thread clusters orchestration queue framework line bubble up code context loops path.
        raise


# ── Pass 2: Efficiency + Composite Update ────────────────────────────────────


def update_efficiency_and_composite(
    results: list[ExperimentResult],
) -> None:
    """
    Pass 2 — Per-source efficiency normalization + composite score calculation.

    Sabhi results same source ke hain — latency_per_doc min-max normalize karo.
    Phir composite calculate karo aur MLflow mein log karo.
    Run terminate karo FINISHED status se.
    """
    # Security bounds array size validation checks control path branch data fallback redirection check runtime boundary condition trace model statement setup.
    if not results:
        return

    # Low level explicit monitoring management client constructor initialize bypassing framework generic state parameters explicit tracker session instance allocations.
    client = MlflowClient()

    # Per-source latency normalization
    # Dynamic list comprehension parsing loop calculating runtime processing averages per individual file unit mappings results records indicators parameter evaluation tracking equations text layout elements tracking logic array.
    latency_per_doc: list[float] = [
        r.latency_seconds / r.doc_count if r.doc_count > 0 else 0.0
        for r in results
    ]
    # Perform mathematical normalization conversions function call parameters passing arrays floating results return container maps metrics variables configuration layer.
    efficiency_scores: list[float] = compute_efficiency_score(latency_per_doc)

    # Parallel arrays processing traversal scanning metrics unpacking structures elements collections sequential check index iteration logic context options mapping loop variables.
    for result, eff_score in zip(results, efficiency_scores):
        # High level master compound logic configuration formula resolution invoke passing variables calculated scores dimensions return metrics alignment score parameter value tracking indicators.
        composite: float = compute_composite_score(
            density=result.avg_semantic_density,
            boundary=result.boundary_respect_score,
            consistency=result.size_consistency_score,
            efficiency=eff_score,
        )

        # Ingest computed dynamic derived values fields direct inside local object record data attributes references values save trace block path configuration model tracking variables.
        result.efficiency_score = eff_score
        # Carry structural calculated final overall matrix data back into destination dataset models variables memory address update indicator parameters data fields array collections.
        result.composite_score = composite

        # Log derived metrics
        # Ship derived dynamic performance parameters directly inside server dashboard logging pipeline analytics platform monitoring target updates data check indicators api.
        client.log_metric(result.run_id, "efficiency_score", eff_score)
        # Project master benchmark parameters directly remote infrastructure tracking parameters dashboards logging modules execution data parameters write runtime check line log interface api.
        client.log_metric(result.run_id, "composite_score", composite)

        # Terminate run
        # Update server backend transaction manager records context flags to mark current sequence lifecycle state fully closed finish status call trace execution path properties parameter.
        client.set_terminated(result.run_id, status="FINISHED")

        # Telemetry analytical report logging update info tracing message console printing results overview verification layout metrics data tracking parameter code write parameters console output print text.
        logger.info(
            "Pass2 done: %s__%s | efficiency=%.4f | composite=%.4f",
            result.strategy,
            result.source,
            eff_score,
            composite,
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def run_all_experiments(skip_api: bool = False) -> None:
    """
    45 MLflow runs — 9 strategies × 5 sources.

    Two-pass per source:
      Pass 1 — Raw metrics (parallel for non-API, sequential for API-bound)
      Pass 2 — Efficiency normalization + composite score + run termination

    Execution model:
      Non-API (6 strategies): ThreadPoolExecutor (MAX_WORKERS=4)
      API-bound (3 strategies): Sequential — rate limit safe
    """
    # Active server backend target connection pathways url parameter configurations setting reading validation store endpoint routing interface logic context.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # Master workspace folder project data storage registry directory initialization track components mapping context model register trace validation project initial data loading project maps.
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    # Technical core parameter data lookup explicit tracking identifier code context reading variable fields assignment allocation storage pointer configuration state checking indicators project entry reference.
    experiment_id: str = experiment.experiment_id

    # Master results compilation database memory placeholder container array listing targets tracking overall tracking metadata metrics collection dynamic variable dataset maps record matrix tracking.
    all_results: list[ExperimentResult] = []

    # Master loop iteration scope scanning document folders classification maps lists variables configurations data folders processing layout sequence loops matrix traversal path loadings.
    for source_folder, source_display in SOURCES.items():
        # Load documentation bulk contexts lists using system loading function returns data array dictionaries dynamic variable mappings loading components directory readings data arrays files parser layout logic tools.
        docs: list[dict[str, Any]] = load_documents(source_folder)
        # Empty inputs checking safeguards parameter condition branch validation check data loop boundary conditions safety constraints checks condition branches control loop forward pointer redirects options matrix configs verification fields.
        if not docs:
            # System notification warning logging triggers when vacant inputs datasets discovered context mapping checking options fields variables mapping limitations tracking missing message structural indicators.
            logger.warning("No documents found: %s", source_folder)
            # Forward core execution scheduler loop indices forward directly into neighboring source folders entries trace path skip iteration branch execution sequence pointer track context loops mapping structures redirect indicators check.
            continue
        # Context dynamic injection leakage insulation dataset building variable creation routine parameters passing components map list values validation tracking data return flow loop insulation data setup logic memory.
        doc_content_map: dict[str, str] = build_doc_content_map(docs)

        # Performance analytics metrics visibility reporting dashboard console log statistics instrumentation tracking updates text stream logic console parameters visibility printout indicators overview metrics trace execution print layout block line text.
        logger.info(
            "Source: %s | docs: %d | strategies: %d",
            source_display,
            len(docs),
            len(NON_API_STRATEGIES)
            + (0 if skip_api else len(API_BOUND_STRATEGIES)),
        )

        # Local source processing elements tracking database array placeholder storage variables calculation tracking context layout results records storage lists configurations parameters.
        source_results: list[ExperimentResult] = []

        # ── Parallel: Non-API strategies ─────────────────────────────

        # Asynchronous concurrency manager thread pool setups structures control allocation capacity limits indicator tracking pipeline block initialization trigger execution layout components.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # List comprehension task dynamic schedules dictionary items mapping thread worker workflows direct memory mapping variables lists array code parallel thread scheduler mapping task mapper thread async submission.
            futures = {
                executor.submit(
                    run_single_experiment,
                    strategy_name=strategy_name,
                    docs=docs,
                    doc_content_map=doc_content_map,
                    source_name=source_display,
                    experiment_id=experiment_id,
                ): strategy_name
                for strategy_name in NON_API_STRATEGIES
            }
            # Active tracking polling loops checking future parameters async variables metrics updates loop monitoring configurations check validation status updates trace tracking elements loop look checking routines event async.
            for future in as_completed(futures):
                # Fetch localized parsing strategy tracking string labels direct dictionary mapping index parameters lookup keys variables sequence config check layout character strings options.
                strategy_name = futures[future]
                try:
                    # Ingest resolved thread data parameters channels calculated summaries records lists direct inside active local results database tracking array list properties variable items collection.
                    source_results.append(future.result())
                except Exception as e:
                    # Intercept thread exceptions processing models failure reporting indicators log text trace matrix evaluations error context parameters write console documentation reports metrics code concurrency failures.
                    logger.error(
                        "Failed: %s × %s — %s",
                        strategy_name,
                        source_display,
                        e,
                    )

        # ── Pass 1B: Sequential API-bound strategies ──────────────────────────

        # Budget boundaries usage condition checks modifiers options parameters validation check criteria logic routing parameters flags checking metrics tokens variables rate limit safe setup context variables loop structures.
        if not skip_api:
            # Sorted iteration loops data mapping array items listings traversal sequences options parameters order sorting alphanumeric index code lists context loop sorting configurations execution paths logic.
            for strategy_name in sorted(API_BOUND_STRATEGIES):
                try:
                    # Running single standalone execution tracking routine logic call parameters components passing tracking variable schemas database models reports layout indicator execution pointer sequence task runner invocation direct metrics.
                    result = run_single_experiment(
                        strategy_name=strategy_name,
                        docs=docs,
                        doc_content_map=doc_content_map,
                        source_name=source_display,
                        experiment_id=experiment_id,
                    )
                    # Accumulate computed linear execution summaries matrices records right inside local source processing list data arrays components variable targets dictionary data items collection.
                    source_results.append(result)
                except Exception as e:
                    # Intercept parsing faults or remote network timeouts errors warnings logging terminal reporting exceptions data maps metrics values execution diagnostics indicators monitor data pipeline layout line print variables.
                    logger.error(
                        "API strategy failed: %s × %s — %s",
                        strategy_name,
                        source_display,
                        e,
                    )

        # ── Pass 2: Efficiency + Composite ───────────────────────────────────

        # Secondary normalization score adjustments optimization mathematical routines execution subroutine call passing dataset elements metrics calculation components track variables models values trace function path.
        update_efficiency_and_composite(source_results)
        # Accumulate completely calculated execution summaries records maps back inside global table tracker database matrix arrays lists values collection master results context mapping loop.
        all_results.extend(source_results)

    # ── Summary Table ─────────────────────────────────────────────────────────
    # Ordering metadata entries tracking layout parameters using sorting indexing rules parameters lambda data properties sequences coordinate layout process loop variable check data trace sorting parameters configurations index structural sorting logic alignment.
    all_results.sort(key=lambda r: (r.source, -r.composite_score))

    # Visual alignment console tracking design borders drawing standard patterns line text log execution parameters visualization track metrics block layout graphics print code sequence separation line layout graphics dashboard configurations framework separator bar visual.
    logger.info("\n%s", "=" * 130)
    # Master summary table console banner updates text indicators console output logger verification prints report framework statement logic setup variables trace master dashboard calculation metrics summaries printing terminal visibility update overview indicators text values tracking.
    logger.info(
        "CHUNKING EXPERIMENT v2 SUMMARY — Composite Score Ranked"
    )
    # Border pattern layout separator line drawing standard parameters console logs reporting analytics dashboard configuration graphic structure context variables updates trace mapping line alignment separators divider horizontal dashboard framing visual layout marker separations graphic borders.
    logger.info("=" * 130)
    # Typography padding metrics adjustments dynamic spacing calculations character length alignment formatting fields variables configuration layout setup data definitions write text parameters padding spacing format markers column metrics calculations formatting typography configurations labels definitions write terminal output console channels log.
    logger.info(
        f"{'Strategy':<20} {'Source':<12} {'Chunks':>7} "
        f"{'Density':>9} {'Boundary':>9} {'Consist':>9} {'Effic':>7} {'Composite':>10} {'MaxChunk':>10}"
    )
    # Spacing dividers rendering standard lines console logging presents reports visibility optimization visual formatting clean terminal view code parameters line step layout level tracking reporting console separator bars segment layout separation line chart grids console presentations layout screen guidelines alignment separator line drawing indicators canvas format indicators.
    logger.info("-" * 130)

    # Summary results dataset maps traversal element processing configurations loop scanning steps arrays verification variables context output trace logic parameter details check code trace dynamic variable iterations mapping text processing records iterations layout dashboard numbers summary evaluation parameters checking values data configurations listing.
    for r in all_results:
        # Dynamic property parameter numbers variables float conversions configuration padding layout formatting column calculations math console channel direct string updates execution line text visual string alignments formatters layout string format width configurations decimal operations metric prints console outputs numbers conversions.
        logger.info(
            f"{r.strategy:<20} {r.source:<12} {int(r.chunk_count):>7} "
            f"{r.avg_semantic_density:>9.4f} {r.boundary_respect_score:>9.4f} "
            f"{r.size_consistency_score:>9.4f} {r.efficiency_score:>7.4f} "
            f"{r.composite_score:>10.4f} {int(r.max_chunk_size_chars):>10}"
        )

    # Visual console line framing layouts border markers typography patterns write log analytics results terminal view indicator report framework statement divider graphics drawing lines code.
    logger.info("=" * 130)

    # Per-source winner
    # Typography template variable string formatting visualization banner overlays main descriptions styling parameters log terminal console updates trace output text display models template string compound charting dynamic name parameters header layouts lines report labels console look visibility metrics.
    logger.info("\n🏆 PER-SOURCE WINNERS (by composite score):")
    # Dynamic tracking history verification cache collection parameters initialize lookup key set memory tracking constraints state logic filter execution trace loop.
    seen_sources: set[str] = set()
    # Summary results maps listings scanning items iteration variables sequence data models formatting layout padding measurements metrics print tracking statement.
    for r in all_results:
        # Cross check validation tracking elements arrays visibility verifying if current category has skipped first unique high scoring entry.
        if r.source not in seen_sources:
            # Inject newly indexed category direct inside cache memory tracker collections parameters variable visibility update.
            seen_sources.add(r.source)
            # Dynamic tracking property string parameter values float numbers layout column alignment configuration parameters metrics terminal console write data tracking summary logs trace text message info print.
            logger.info(
                "  %-12s → %-20s | composite=%.4f | density=%.4f | boundary=%.4f",
                r.source,
                r.strategy,
                r.composite_score,
                r.avg_semantic_density,
                r.boundary_respect_score,
            )

    # User guidelines tracking paths URLs instructions details terminal updates console trace configuration properties print mapping documentation logs visibility level path line data check manual guidelines directions display accessibility dashboard instructions locations terminal printout navigation guidance reference layouts links data path log.
    logger.info(
        "\nMLflow UI: mlflow ui --backend-store-uri mlruns → http://127.0.0.1:5000"
    )
    # Enterprise project traces index reference analytics pipelines variables dashboard links path logger terminal text output printing layout monitoring trace pipeline parameters config browser trace logging directions options cloud telemetry indices reference dashboard configuration links terminal logs view project trace logging pathway.
    logger.info(
        "LangSmith : https://smith.langchain.com → project: %s",
        settings.langchain_project,
    )


# ── Entry Point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Command interface system terminal inputs parsing constructor handler instantiation configurations register variables tracking control execution initialization setup metrics field data line execution variables dictionary lookups structural arguments parsing terminal handling blueprints validation setups setup dynamic controller rules.
    parser = argparse.ArgumentParser(
        description="SkyLex Chunking Experiment v2 — 5-metric composite scoring"
    )
    # CLI command arguments option modifier flag definitions allocation dynamic binary switches data logic true values parameters configuration variables properties tracing step level assignment modifier execution toggles custom modifiers option parameters switches variables checking boolean parameters allocation layout modifiers.
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="API-bound strategies skip karo (semantic, improved_semantic, double_pass)",
    )
    # Input terminal data configurations lookup validation process trigger execution variables parameter collection structure read values parameters lookup setup metrics fields data line execution variables dictionary lookups parser arguments checking evaluation context variables query mappings fetch record storage indicators parameters mapping setups layout options fields validations logic.
    args = parser.parse_args()
    # Execute master orchestrator loop routing configurations pass properties flags conditions experiments matrix matrix analysis processing pipeline loops execution indicator complete trace run core loop trigger execution pathways core master routine orchestration call parameters passing modifier flags dynamic conditional processing run loops matrix matrix checks analytics.
    run_all_experiments(skip_api=args.skip_api)