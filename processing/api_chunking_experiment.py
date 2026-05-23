"""
processing/api_chunking_experiment.py

SkyLex Phase 2 — API-Bound Chunking Strategy Experiment.

Strategies: semantic, improved_semantic, double_pass
Sources: FAA_CFR, FAA_AD, FAA_AC, DGCA_CAR, SKYBRARY

Runs in separate MLflow experiment (skylex_api_chunking)
taaki saare results ek jagah compare ho sakein.

Cost control:
  - Per-source sample size configurable (default: 5 docs)
  - Sequential execution only — no parallel API calls
  - Cost estimate shown before running — user confirm karta hai

Estimated cost at default sample sizes:
  ~$0.01-0.02 total (text-embedding-3-small @ $0.02/1M tokens)

Usage:
    python processing/api_chunking_experiment.py
    python processing/api_chunking_experiment.py --sample-size 10
    python processing/api_chunking_experiment.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
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
    ImprovedSemanticChunker,
    SemanticChunker,
)

# Logging framework aur stream formatters ko bootstrap kiya jaa raha hai runtime tracking analytics safe rakhne ke liye.
setup_logging()
# Dynamic isolation module levels par events track aur trace karne ke liye telemetry logger catch configure instantiate kiya.
logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# MLflow backend server par pipeline experiments metrics ko isolated lookups directory me map karne ke liye new identity tag define register kiya.
EXPERIMENT_NAME: str = "skylex_api_chunking"
# Central parameters management system settings properties directory jahan pipeline raw json contents files base path map data store located hai.
RAW_DATA_DIR: Path = settings.data_raw_dir
# Sizing calculations dynamic configurations translation formatting ratio values parameters integers scale tokens measure.
CHARS_PER_TOKEN: float = 4.0
# Vector transformation boundaries validation limits safe truncation parameter constraints capacity upper ceiling indicator numbers.
EMBEDDING_MAX_CHARS: int = 6000
# Mathematical matrix statistical state reproducible generations variables replication validation control setup tracking numbers.
RANDOM_SEED: int = 42

# Composite score weights
# Semantics topic context fidelity validation criteria checking formulas mapping multiplier variables parameter adjustments index.
WEIGHT_DENSITY: float = 0.40
# Sentence level structural completion accuracy indicator scale ratio multiplier calculation constraints balance setup value data map.
WEIGHT_BOUNDARY: float = 0.30
# Uniform structural data distribution variance balancing parameters mathematical coefficients weight tracker configuration map.
WEIGHT_CONSISTENCY: float = 0.20
# Performance processing hardware throughput execution velocity parameters checking logic score ratio configuration values map layer.
WEIGHT_EFFICIENCY: float = 0.10

# Default sample sizes per source — cost control
# Massive dollar costs aur bulk multi tokens parsing limits protection thresholds parameters check mapping source limits configurations.
DEFAULT_SAMPLE_SIZES: dict[str, int] = {
    "faa_cfr": 10,  # 226 total — 10 enough for pattern
    "faa_ad": 10,  # 347 total — 10 enough
    "faa_ac": 5,  # 27 total — 5 enough (large docs, expensive)
    "dgca_car": 3,  # 6 total — 3 enough (small source)
    "skybrary": 5,  # 20 total — 5 enough
}

# Sources input paths metadata matching strings text labels configurations data directory collections setup array parameters variables.
SOURCES: dict[str, str] = {
    "faa_cfr": "FAA_CFR",
    "faa_ad": "FAA_AD",
    "faa_ac": "FAA_AC",
    "dgca_car": "DGCA_CAR",
    "skybrary": "SKYBRARY",
}

# OpenAI API calls operations checks sequential execution tracking strategies pipeline queue look lists verification sets data blocks logic layer.
API_STRATEGIES: list[str] = ["semantic", "improved_semantic", "double_pass"]


# ── Result Dataclass ──────────────────────────────────────────────────────────


@dataclass
class ExperimentResult:
    """Single strategy × source result — same structure as main experiment."""

    # Downstream indexing models, metrics analytics orchestration validation parameters tracking identifier string configuration dashboard context.
    strategy: str
    # Ingestion documentation category layout keys lookup mapping dynamic configuration labels properties parameters target locations variable trace data.
    source: str
    # Quantitative documentation units boundaries configurations aggregate counter parameters code validation rule check data fields tracking block data loops.
    doc_count: int
    # Sliced inventory documentation volume calculations subset elements limits indicator value properties register tracking code path.
    sampled_doc_count: int
    # Technical transaction identification codes parsing framework parameters update variable assignments mapping database records trace logic.
    run_id: str

    # Chunks produces metrics storage placeholder memory arrays targets properties validation structure setup variables math count accumulator.
    chunk_count: float = 0.0
    # Average character capacities distributions scale criteria checking metrics integers values conversion lookup specifications logic trace.
    avg_chunk_size_chars: float = 0.0
    # Lower constraints limits evaluation configurations matching parameter checks validation boundaries lower threshold parameters variable text arrays.
    min_chunk_size_chars: float = 0.0
    # Maximum safe range capacity boundary condition checks limits validation constraints options character parameters mapping storage fields analysis.
    max_chunk_size_chars: float = 0.0
    # Size distributions variance check math tracking calculations matrix floating variables value mappings configuration metrics dashboard statistics.
    std_chunk_size_chars: float = 0.0
    # Vacant nodes structural whitespace filtering analytics indicators control counters properties check loops validation data context layout options checker.
    empty_chunk_count: float = 0.0
    # Density optimization scaling coefficient division mapping formula index calculations ratio parameters settings options execution workflow pointer track.
    chunks_per_doc_avg: float = 0.0
    # Approximate embedding cost calculation models token counts allocation matrix parameters layout summary calculation structural tracking metrics options.
    estimated_tokens_total: float = 0.0

    # Core metrics
    # Semantic consistency vector metrics dynamic measurements calculation arithmetic averages properties lookup mappings tracking validation data matrix variables.
    avg_semantic_density: float = 0.0
    # Grammatical structure compliance check accuracy indicators value mapping verification tracking validations loops parameter check metrics scale ratio.
    boundary_respect_score: float = 0.0
    # Data sizes distribution variation uniform measure coefficient calculation parameters floating precision indicators storage data configuration layout registry.
    size_consistency_score: float = 0.0
    # Performance total benchmark calculations timing durations snapshots delta calculations output variables parameters metrics track dashboard check sequence.
    latency_seconds: float = 0.0
    # Normalized relative calculation scoring index parameters properties conversions floats mapping dashboard metrics system check validation path tracking metrics.
    efficiency_score: float = 0.0
    # Master overall benchmark analytics combination scoring formula check context mapping dynamic metrics parameters layout validation options trace dashboard.
    composite_score: float = 0.0

    # Extra summary settings array maps configurations properties dictionary items mapping data collections storage metadata inventory parameters targets workspace variables.
    params: dict[str, Any] = field(default_factory=dict)


# ── Retry Decorator ───────────────────────────────────────────────────────────


def _make_embed_with_retry(embedder: OpenAIEmbeddings) -> Any:
    """
    embed_documents ko tenacity retry wrapper mein wrap karo.
    Exponential backoff + jitter — rate limit (429) handle karo.
    """

    # Multi level resilience control closure wrapper function dynamic options configuration properties logic checking exceptions boundary tracking validator code data layer.
    @retry(
        # Intercept temporary provider network drops, validation glitches, or server communication failures pattern checks condition parameters routing validation logic loop.
        retry=retry_if_exception_type(Exception),
        # Target remote services connectivity transactions constraints maximum retries thresholds options index checks variables parameter structure context validation.
        stop=stop_after_attempt(5),
        # Wait mathematical calculations timing exponential dynamic coefficients parameters layouts mapping intervals framework backoff algorithms jitter trace data metrics.
        wait=wait_exponential_jitter(initial=2, max=60, jitter=4),
        # Bubble up structural exceptions data context straight inside running executor stack lines code tracking execution error trace context mapping.
        reraise=True,
    )
    def _embed(texts: list[str]) -> list[list[float]]:
        # Outbound list data characters arrays query direct computing multi dimensional matrix models return lists vectors functions tracking metadata look validation variables.
        return embedder.embed_documents(texts)  # type: ignore[return-value]

    # Return freshly encapsulated thread safe function callback mapping reference straight back to evaluation analytics orchestrator module runner component layer.
    return _embed


# ── Factory Functions ─────────────────────────────────────────────────────────


def make_chunker(strategy_name: str) -> Any:
    """Fresh API-bound chunker instance per run."""
    # Advanced vector space drop distance calculations model sorting options matching variables selectors properties map configuration parameters strategy check data.
    if strategy_name == "semantic":
        # Initialize semantic baseline topic boundary calculation strategy freshly constructor instance object build return pipeline tracking workflow metrics logic check.
        return SemanticChunker()
    # Optimization target parameter threshold checks structural processing branch validation routing paths selector option tracking property code mapping layer variables context.
    if strategy_name == "improved_semantic":
        # Lower threshold limits validation multi level calculations chunking strategy object fresh instance allocate return structural workspace metrics pipeline tracer.
        return ImprovedSemanticChunker()
    # Multi pass tracking hybrid coarse semantic recursive fine data transformations checking function constructor execution options dictionary metrics complete indicator parameters.
    if strategy_name == "double_pass":
        # Dual operational phase logic configurations pipeline subroutines execution mapping parameters attributes model constructor instantiate runtime memory array trace parameters.
        return DoublePassChunker()
    # Inbound malformed string parameters safeguards checking dynamic verification exceptions trigger code trace execution framework options parsing block code validation report.
    raise ValueError(f"Unknown API strategy: {strategy_name}")


def make_embedder() -> OpenAIEmbeddings:
    """Fresh OpenAIEmbeddings instance — no shared httpx pool."""
    # Networking tracking dependencies states leaks connection pool corruptions prevention initialization client configuration parameters build call sequence definitions context trace variable.
    return OpenAIEmbeddings(
        # Settings metadata model properties registry directory parameters tracking lookup configuration reading values context scale analytics trackers setup mapping parameters.
        model=settings.openai_embedding_model,
        # Encapsulated tokens framework validation types configuration layout allocations memory data validations level annotations layers memory data processing track.
        openai_api_key=SecretStr(settings.openai_api_key),  # type: ignore[call-arg]
    )


def make_local_random() -> random.Random:
    """Per-run isolated random.Random — no global state mutation."""
    # Isolated stand alone state pseudo random generation structural engine mapping instance initialization allocation seed parameters return tracking model logic trace code lifecycle context.
    return random.Random(RANDOM_SEED)


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_documents(source_folder: str) -> list[dict[str, Any]]:
    """
    Source folder se saare bulk JSON files load karo.
    hash_registry.json aur _meta.json exclude karo.
    Duplicate doc_ids deduplicate — pehli occurrence rakho.
    """
    # Target filesystem directory location building template configuration maps setup file addresses string tracking pathway code pointer memory address validation tracking logic parameters path.
    source_dir: Path = RAW_DATA_DIR / source_folder
    # Output verified clean document model dictionary data collections list allocation tracking configuration storage map array variable initialized placeholder trace database models tracking loading context.
    all_docs: list[dict[str, Any]] = []
    # Real-time memory hash cache tracking sets map database indices duplicate filtering configurations lookup layer control logic setup state tracking parameters check loops validate unique variable mappings structure.
    seen_ids: set[str] = set()

    # Recursive global subdirectory trees scanning mapping search loop execution crawl storage components directory parsing structural layout matching files process data crawlers dynamic processing cycle matrix.
    for json_file in sorted(source_dir.rglob("*.json")):
        # Registry logging data indexes skipping criteria rule tracking parameters checking condition branch validation context path skip step rule baseline parameters trace context log files indicator options layout.
        if json_file.name == "hash_registry.json":
            continue
        # Descriptors metadata file layout rules structure specifications matching skip constraints processing check step memory configuration parameters trace layout code block layer text strings mapping validation logic structures.
        if json_file.name.endswith("_meta.json"):
            continue
        try:
            # Native filesystem descriptor mapping channel access setup reader configuration properties unicode standard mapping registry stream parameters framework path logic options stream read tracking pipeline context layouts options.
            with json_file.open(encoding="utf-8") as f:
                # Character stream formatting parser runtime deserialization matrix transform directly inside active object reference parameters map load processing options memory target variable parameters execution blocks context reader configuration properties unicode standard.
                data: Any = json.load(f)
            # Object schema validation rule properties tracking context verification parameters data structural integrity status loop boundary filtering check logic indicator context type configurations parsing check metrics validation options layout bounds thread properties dictionary.
            if not isinstance(data, dict) or "documents" not in data:
                continue
            # Data matrices payload list iteration mapping elements internal data structures traverse sequence workflow tracking controls execution loop parameters state data collections step parsing indicator loops tracking trace items properties traversal route execution.
            for doc in data["documents"]:
                # Individual data component layout unique indicator identifier key reference string lookups read pipeline trace matching choices validation default key parameter tracking values details options code mapping fields individual metadata item.
                doc_id: str = doc.get("doc_id", "")
                # Cross check current parsed identifier code history registry history parameters check data overlap conditions filter execution tracking cache verification matching indices rule constraints checking logic verification options history tracking cache memory lookup.
                if doc_id and doc_id not in seen_ids:
                    # Inject valid newly discovered identifier string key right inside memory framework lookup tracking collection update instantly caching index trace registry mapping framework update data sets cache tracking set layout update variables metrics memory framework tracing.
                    seen_ids.add(doc_id)
                    # Non overlapping validated genuine document database dictionary metadata record components append into destination tracking array properties variable targets data items metadata collection storage lists configuration rules elements parsing non overlapping validated baseline.
                    all_docs.append(doc)
        except (json.JSONDecodeError, OSError) as e:
            # Handle malformed files encoding rules or hardware level disk parsing block operations safely without stopping master multi threaded background orchestrator scripts main execution loop trace context.
            logger.warning("Could not load %s: %s", json_file, e)

    # Performance logging pipeline tracking reporting console terminal metrics indicator statistics feed print out execution layout trace summary metadata output channel view info tracking dashboard console log data.
    logger.info(
        "Loaded %d unique documents from %s", len(all_docs), source_folder
    )
    # Output isolated completely filtered sanitized clean documentation maps list reference back to core scheduling runner loop processes framework model data elements repository structural variables setup layout completely isolated clean dynamic.
    return all_docs


def build_doc_content_map(documents: list[dict[str, Any]]) -> dict[str, str]:
    """doc_id → full original content — semantic density ke liye, no data leak."""
    # Python dictionary comprehensions parsing syntax layout mapping equations calculations properties text variables allocation memory context loop variables lookup trace step metrics balance options mapping tracker rules expressions maps data elements block reading maps optimized list array.
    return {
        doc["doc_id"]: doc.get("content", "")
        for doc in documents
        if doc.get("doc_id") and doc.get("content")
    }


def sample_documents(
    documents: list[dict[str, Any]],
    sample_size: int,
    local_random: random.Random,
) -> list[dict[str, Any]]:
    """
    Stratified random sample — cost control ke liye.
    Agar sample_size >= total docs toh sab return karo.
    """
    # Slicing scale boundaries condition calculations rules validation checking constraints loop bypass redirect trace parameters optimization layout check framework.
    if sample_size >= len(documents):
        return documents

    # Thread isolated dynamic sample allocation engine subroutine invoke passing metrics parameters values selections list arrays extraction processing models tracking.
    return local_random.sample(documents, sample_size)


# ── Metric Calculators ────────────────────────────────────────────────────────


def compute_structural_metrics(
    chunks: list[ChunkedDocument],
) -> dict[str, float]:
    """Chunk size distribution metrics."""
    # Vacant inputs boundaries checking options configurations parameters check metrics validation default placeholder array returns mappings logic control branch code fallback routing matrix trace.
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

    # Extract dynamic list array integer sizing measurements calculations metrics characters ranges loop values list string definitions measurements context parameter.
    sizes: list[int] = [len(c.content) for c in chunks]
    # Spacing structures clean validations parameters loops counter accumulation calculation matching data elements properties conditions formulas summaries tracking checks.
    empty_count: int = sum(1 for c in chunks if not c.content.strip())
    # Distinct original baseline data uniqueness references lookup index sets initialization logic calculations parameters indices trace mapping variance validation cache.
    unique_docs: int = len({c.doc_id for c in chunks})
    # Density balancing ratio index division calculation rules formula logic parameter execution tracking metrics mapping dynamic loop context properties switch branch tracking metrics.
    chunks_per_doc: float = (
        len(chunks) / unique_docs if unique_docs > 0 else 0.0
    )

    # Packaging calculated analytics metrics floating values parameters inside reporting dashboard dictionary structures collections tracking layout variables monitors.
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
    complete_chunks / total_chunks — higher = better.
    """
    # Vacant sequences items control validation checking condition branches redirection mapping path metrics setup logic parameters code verification trace.
    if not chunks:
        return 0.0
    # Core standard grammatical punctuation layout templates markers checking choices definitions mapping structures array configuration context options parameter lookup rules text.
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
    # Text string tail character structure loops verification aggregate matches summation code pipeline validation run context layout calculations checks parameter logic metrics.
    complete: int = sum(
        1
        for c in chunks
        if c.content.strip().endswith(sentence_endings)
    )
    # Precision decimal ratios computation limits floating round parameters math division index values mapping variables framework outputs data trace code context.
    return round(complete / len(chunks), 4)


def compute_size_consistency_score(
    avg_chunk_size: float,
    std_chunk_size: float,
) -> float:
    """
    Size consistency via Coefficient of Variation.
    max(0.0, 1.0 - std/avg) — higher = more consistent sizes.
    """
    # Structural bounds check condition parameter verification conditions zero values mapping prevent options layout parameters routing path branch redirect logic trace metrics.
    if avg_chunk_size <= 0:
        return 0.0
    # Calculate variation coefficient scale variance value parameters integration mathematical formula checking metrics ratio index configuration option properties tracking context.
    return round(max(0.0, 1.0 - std_chunk_size / avg_chunk_size), 4)


def compute_semantic_density(
    chunks: list[ChunkedDocument],
    doc_content_map: dict[str, str],
    embedder: OpenAIEmbeddings,
    local_random: random.Random,
) -> float:
    """
    Average cosine similarity — chunk vs actual parent document.
    Dynamic sample: min(50, max(5, 5% of chunks)).
    Stratified random sampling — no first-N bias.
    Token-safe truncation — 6000 chars max.
    """
    # Safety parameter mapping validations checkpoints lookup verify structural data properties values database tracking maps code initialization path checking criteria constraints rules context.
    if not chunks or not doc_content_map:
        return 0.0

    # Scale percentage calculations values dynamic boundary indicators formulas calculations check limits variables parameters formulations setup context data values math model index logic.
    sample_size: int = min(50, max(5, int(len(chunks) * 0.05)))
    # Completely thread decoupled unbiased randomized selection array allocation slice window collection processing target trigger executor tracking metrics database sample mapping array lists.
    sample: list[ChunkedDocument] = local_random.sample(
        chunks, min(len(chunks), sample_size)
    )

    # Valid confirmed processing items text configurations pairs arrays lookup verification layer data parameters collections placeholder dictionary map array tracking container targets.
    valid_pairs: list[tuple[str, str]] = []
    # Scanning elements matrix traversal loop variables items metadata structures context processing loop step run pipeline runtime metrics components check conditions array tracks check dynamic path.
    for chunk in sample:
        # Dynamic memory map directory extraction lookup properties parameters safe defaults key lookup reading execution context values variable query tracing context values setup reading target map data.
        parent_content: str = doc_content_map.get(chunk.doc_id, "")
        # String content length validation matching rules verification sequence checking condition branch layout validation check control context branch logic trace layer metrics checks verification alignment.
        if parent_content:
            # Token constraints safety limitation boundary character metrics parameters map layout slicing operation dynamic extraction collection variables array save trace parameters capacity calculations.
            valid_pairs.append(
                (
                    chunk.content[:EMBEDDING_MAX_CHARS],
                    parent_content[:EMBEDDING_MAX_CHARS],
                )
            )

    # Empty validation tracking check verifying if matching elements sets has failed structural array counts context data pipeline safety pointer check logic trace validation framework monitoring processing.
    if not valid_pairs:
        logger.warning("No valid chunk-parent pairs — returning 0.0")
        return 0.0

    # Dynamically inject tenacity retry resilience layer wrapper right on top of the thread-isolated embedding engine instance function pointer variables setup tracking mapping core logic closure function connection pool.
    embed_fn = _make_embed_with_retry(embedder)

    try:
        # Request outbound remote provider network embedding vector array matrix calculations using dynamic retry wrapped routine execution result arrays return trace list data tracking parameters connectivity options tokens metrics evaluation models query stream updates trace tracking array vector.
        chunk_embeddings: list[list[float]] = embed_fn(
            [p[0] for p in valid_pairs]
        )
        # Parent original documentation sequences text vectors representation computation provider connection call address pipeline routing data maps values execution results trace remote servers computation layers data alignment vector models mappings metrics trace indicators models calculations.
        parent_embeddings: list[list[float]] = embed_fn(
            [p[1] for p in valid_pairs]
        )

        # Distance similarity arrays registry storage context target metrics track tracking array allocation initialization code variables calculations loop parameters setup metrics model analytics mapping results trace arrays configurations database registry repository.
        similarities: list[float] = []
        # Multi-variable loop tracking traversing parallel vector arrays coordinates data unpack trace execution validations sequence check control iterations blocks tracking look trace matrix multi component pairs mapping dimensions scanning iterations validation parameters coordinates tracking data lookup.
        for c_emb, p_emb in zip(chunk_embeddings, parent_embeddings):
            # Convert basic array list floats into high speed mathematical matrix processing numpy structures layout tracking objects parameters values design rules calculation trace model high performance multidimensional optimization float operations metrics calculations tracking linear structure array models formatting.
            c_arr = np.array(c_emb)
            # Alignment matching checking numerical checks validation properties configuration map elements inside high efficiency numpy object calculations parameters check runtime log trace data check metrics operations values validation properties matrix dimension checks layout tracking configurations analysis processing.
            p_arr = np.array(p_emb)
            # Vector norm dimensions square matrix multiplier calculations structural magnitude coefficients index resolution safety check variable trace formula execution data layer context rules math metrics calculation formulas magnitude multiplier scaling protection parameters checks variance boundaries division.
            norm_product: float = float(
                np.linalg.norm(c_arr) * np.linalg.norm(p_arr)
            )
            # Zero division protection boundaries exceptions branching path skip execution context loop controls validation tracker layer tracking code logic framework operations trace variables criteria validation constraints safety tracking checks loop bypass options redirector indicators.
            if norm_product < 1e-10:
                continue
            # Distance matrix resolution linear dot multiplier products logic formula resolution score parameter verification dynamic mapping data variable list append track indicator data flow checks process vector operations linear multipliers results aggregation float scores mapping index evaluation.
            similarities.append(
                float(np.dot(c_arr, p_arr) / norm_product)
            )

        # Yield completely resolved mathematical mean average metrics coefficient properties configurations layout metrics analysis output check trace return flow pointer data logic statistical mean computations coefficient results overview validation parameters floating summaries analytics reporting numbers trace.
        return (
            round(float(np.mean(similarities)), 4)
            if similarities
            else 0.0
        )

    except Exception as e:
        # Capture API connection crashes, authentication timeouts, or unhandled token data parsing issues metrics warnings tracking traces indicators screen output logs warning updates tracking connectivity failures processing exceptions context logs tracking trace indicators stream details parsing data completions.
        logger.warning("Semantic density failed: %s", e)
        # Secure failure mode path redirection standard fallback default values tracking parameters setup execution branch check metrics layout flow trace runtime code recovery options pathway index structural validation safe fallback tracking process indicators monitoring configuration mapping routing flow trace.
        return 0.0


def compute_efficiency_score(
    latency_per_doc_values: list[float],
) -> list[float]:
    """Per-source min-max normalized efficiency scores."""
    # Clean verification bounds constraints checks loop logic parameter options conditions setup metrics validation check data.
    if not latency_per_doc_values:
        return []
    # Identify lowest duration mapping coordinates values parameters tracking lookup calculations model processing trace.
    min_lat: float = min(latency_per_doc_values)
    # Find slowest performance value index limits context verification properties parameter data reading tracker check.
    max_lat: float = max(latency_per_doc_values)
    # Math expression fractions matrix mapping calculations denominator variable properties formulas epsilon protection check boundaries.
    denom: float = max_lat - min_lat + 1e-10
    # Python comprehensive loops scanning variables data structures floating score lists mapping calculation metrics format run text arrays.
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
    Weighted composite — master metric.
    0.40×density + 0.30×boundary + 0.20×consistency + 0.10×efficiency
    """
    # Apply standard multiplier weights criteria dynamic equations math formulations variables layout rounding precision return flow metrics track indicators setup value.
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
    """Thread-safe histogram via matplotlib.figure.Figure()."""
    # Extract string capacities loop list comprehension properties array metrics sizing tracking parameters variables lengths measurements check maps.
    sizes: list[int] = [len(c.content) for c in chunks]
    # Running statistical analyzer calculation computing absolute distribution center points variable values track trace parameters property configuration setup mapping.
    mean_size: float = float(statistics.mean(sizes))

    # Thread safe fully isolated decoupled canvas structure plot generator constructor architecture configuration parameter geometry properties layout initialize.
    fig = matplotlib.figure.Figure(figsize=(10, 5))
    # Coordinate grid management subplot placement panel frame initialization object creation add data view execution parameter validation code trace.
    ax = fig.add_subplot(1, 1, 1)
    # Chart plotting graphics properties layout metrics data colors canvas styling borders opacity parameters tracking frequency density bars map draw execution.
    ax.hist(sizes, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    # Typography template strings compound metrics layout visualization titles descriptions dynamic value injections formatting styling console logging print trace.
    ax.set_title(
        f"Chunk Size Distribution — {strategy.upper()} | {source} (sampled)",
        fontsize=14,
        fontweight="bold",
    )
    # Horizontal axis labeling metrics description label text formatting alignment rules structural setup values character capacity limits tracker parameter variables.
    ax.set_xlabel("Chunk Size (chars)", fontsize=12)
    # Vertical axis coordinate description mapping sequence standard text variables dashboard configurations tracker label options write runtime analytics tracking logic parameter.
    ax.set_ylabel("Frequency", fontsize=12)
    # Overlay guides indicators vertical reference line plots center measurements math calculations dash markers design rules configuration boundaries tracking lines.
    ax.axvline(
        mean_size,
        color="#DD4444",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_size:.0f} chars",
    )
    # Panel box legend identifiers visibility configuration labels box parameters placement console logs monitoring views styling validation call trace data.
    ax.legend(fontsize=11)
    # Canvas border space margins compression optimization tracking layout execution arrangement auto layout drawing clean export render system flow context options.
    fig.tight_layout()

    # Dynamic target file local destination path indexing address generation templates naming layout rules parameter variables track map registry values string.
    plot_path: str = f"{tmp_dir}/{strategy}_{source}_api_chunk_dist.png"
    # Write chart binary stream dataset directly inside system persistent file path format binary image rendering storage execute run data track context loop layer.
    fig.savefig(plot_path, dpi=150)
    # Return destination mapping files path location validation parameters direct back to metadata uploading transaction scheduler system integration layout tracking string.
    return plot_path


def save_sample_chunks_json(
    chunks: list[ChunkedDocument],
    strategy: str,
    source: str,
    tmp_dir: str,
    local_random: random.Random,
    n: int = 5,
) -> str:
    """Stratified random sample of N chunks — qualitative inspection."""
    # Unbiased randomized items tracking slice parameters arrays pulling subroutine options array data models elements extraction processing loops framework tracking.
    sample: list[ChunkedDocument] = local_random.sample(
        chunks, min(len(chunks), n)
    )
    # Transform dynamic object values mappings maps format direct inside serializable standard dictionary keys values context structure list storage variables verification.
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
    # Destination mapping coordinate file name string path coordinate folder layouts mapping template checks tracking validation parameters location variable index.
    json_path: str = (
        f"{tmp_dir}/{strategy}_{source}_api_sample_chunks.json"
    )
    # Open local operating system file streams channel writer configuration attributes read modes enabled unicode parsing safe characters data stream processes.
    with open(json_path, "w", encoding="utf-8") as f:
        # String data serialization packing writing dynamic processing operations layout style indentation properties parameters write execution run trace mapping loops.
        json.dump(output, f, indent=2, ensure_ascii=False)
    # Return saved file string path mapping direct back to metadata upload operations processing sequence context metrics allocator data validation framework indicator reference.
    return json_path


# ── Single Experiment Runner ──────────────────────────────────────────────────


def run_single_experiment(
    strategy_name: str,
    docs: list[dict[str, Any]],
    doc_content_map: dict[str, str],
    source_name: str,
    experiment_id: str,
    total_doc_count: int,
) -> ExperimentResult:
    """
    Ek API strategy × source combination ka experiment.
    Sequential execution — rate limit safe.
    MlflowClient explicit run_id — no fluent API.
    try/finally — run always terminated.
    """
    # Composite dashboard visualization tracking variable context multi component name query identifier format parameter label layout string update data trace tracking loop.
    run_name: str = f"{strategy_name}__{source_name}__sampled"
    # Instrumentation tracking logging info trace notifications print system pipeline run logs visibility display terminal variables validation metadata trace analytics options messages.
    logger.info(
        "Starting: %s | sampled_docs: %d / %d",
        run_name,
        len(docs),
        total_doc_count,
    )

    # Initialize explicit backend monitor manager interface class component completely unbinding standard generic context tracking thread states properties metrics lookup keys variables.
    client = MlflowClient()
    # Request tracking dashboard remote backend endpoint storage layer allocation unique isolated database trace data element context mappings metrics register setup model.
    run = client.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={
            "mlflow.runName": run_name,
            "run_type": "api_chunking_experiment_v2",
            "phase": "phase_2",
            "strategy": strategy_name,
            "source": source_name,
            "strategy_type": "api_bound",
            "sampled": "true",
        },
    )
    # System transaction unique reference code storage parameter identifier value parse dynamic location variables assignment reading metadata check parameters lookup.
    run_id: str = run.info.run_id

    try:
        # Fresh instances — no shared state across threads
        # Strategy localized factory module instantiation subroutine execution passing metrics rules matching structural data configurations parameters layout context verification fields.
        chunker = make_chunker(strategy_name)
        # Vector matrix mathematical transformations processor engine constructor functional invoke settings parameters mappings token configurations parsing path validation.
        embedder = make_embedder()
        # Random parameters isolation generation entropy framework instance allocation initialization sequence criteria properties checks trace metrics isolated state parameters context.
        local_random = make_local_random()

        # Log parameters
        # Properties collection setup storage definitions key variables data parameter map mappings index analytics parameters baseline logs directory map configuration properties dict.
        params: dict[str, Any] = {
            "strategy": strategy_name,
            "source": source_name,
            "total_doc_count": total_doc_count,
            "sampled_doc_count": len(docs),
        }
        # Dynamic variable mapping lookups checking properties existence verification control condition branch logic loop verification parameters checking options query search execution context attributes.
        for attr in (
            "breakpoint_threshold_type",
            "breakpoint_threshold_amount",
            "hard_cap_chars",
            "coarse_threshold",
            "fine_chunk_size",
        ):
            # Dynamic object parameter attribute presence checks verification condition layout route selection code statement execution tracking rules trace.
            if hasattr(chunker, attr):
                # Access target properties instances values directly string keys mappings lookups variables memory context properties map allocation database indicators.
                params[attr] = getattr(chunker, attr)

        # Transmit inventory parameter variables list straight inside distant data infrastructure logs server routing pipeline telemetry api execution map runtime indicator data loop.
        for key, value in params.items():
            client.log_param(run_id, key, value)

        # Chunk + measure latency
        # High resolution hardware platform chronometer checkpointsnapshot snapshots track snapshot timing register execution interval calculation check start line pointer clock.
        start_time: float = time.perf_counter()
        # Structural framework dynamic procedure loop text conversion list data parsing algorithm segmentation processing return trace list data arrays computing trace variable.
        chunks: list[ChunkedDocument] = chunker.chunk(
            docs, deduplicate=True
        )
        # Runtime tracking delta intervals differences calculations decimal constraints rounding formatting latency parameters metrics value save data trace loop level metrics.
        latency: float = round(time.perf_counter() - start_time, 3)

        # Compute metrics
        # Statistical data tracking calculations metrics subroutine context pass components data lists return array mapping layout properties checks validation code framework analysis.
        structural: dict[str, float] = compute_structural_metrics(chunks)
        # Sentence segmentation formatting compliance index indicator validation method call execute criteria validation metrics register trace step accuracy values target console.
        boundary_score: float = compute_boundary_respect_score(chunks)
        # Semantic mapping context stability tracking variables calculation subroutine invocation pass metrics computing vectors evaluation metrics trace logic step vector dimensions values.
        density: float = compute_semantic_density(
            chunks, doc_content_map, embedder, local_random
        )
        # Variation distribution uniform consistency mathematical index calculations parameters execution passing data structures return checking dynamic metrics formulas data.
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

        # Log artifacts
        # Environment system dynamic transient space virtual directory storage manager wrapper block configuration automatic cleanup rules tracking pipeline path trace context execution file system resource operations.
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create chart distribution visualizations local files location strings parsing variables tracking canvas run data trace line execution model graphics chart visualization quantitative metrics histogram png files generator address metrics.
            hist_path = save_chunk_size_histogram(
                chunks, strategy_name, source_name, tmp_dir
            )
            # Create text database items serialized storage structures validation mapping check destination path records files write update indicators check trace data structure previews information check matching serialize text configurations files path layouts storage data string format text preview.
            json_path = save_sample_chunks_json(
                chunks,
                strategy_name,
                source_name,
                tmp_dir,
                local_random,
            )
            # Send analytics plotting image artifacts directly inside master monitoring persistent folder storage pathway upload execute tracking api run trace line code server graphics pipeline uploads cloud assets registry persistent file shipping.
            client.log_artifact(run_id, hist_path, artifact_path="plots")
            # Quality assessment validation verification items lists format specifications details parameters tracking update structural telemetry upload sequence route api data logic text preview formats maps dynamic serialization files upload transactions tracking framework parameters address reference path mappings context hooks template.
            client.log_artifact(
                run_id, json_path, artifact_path="sample_chunks"
            )

        # Update status configurations parameters flags metadata data sets indicators inside distant database transaction record state finish signals execute method call runtime lifecycle completion flags central backend session state final finish termination signal.
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

        # Structured object factory instantiation using dataclass specification model mapping records values direct inside local memory container allocation object data reference placeholder ExperimentResult parameters tracking.
        return ExperimentResult(
            strategy=strategy_name,
            source=source_name,
            doc_count=total_doc_count,
            sampled_doc_count=len(docs),
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

    except Exception as e:
        # Unexpected script faults interception safeguards flags parameter updates tracking crash terminate execution signature framework pipeline execution trace call code block runtime exceptions protection unexpected failures parameters updates tracking execution state exceptions catcher model flags transaction indicators.
        client.set_terminated(run_id, status="FAILED")
        # System failures evaluation reporting error tracking message strings write console diagnostics metrics overview streams monitor parameters log text trace runtime fault analytics database failures logging error message terminal reporting console visibility updates trace logging models details instrumentation trackers check.
        logger.error("Experiment failed: %s — %s", run_name, e)
        # Propagating tracking exception data models back towards master pool queues hierarchy lifecycle control execution thread bubbles indicator code stream line exception bubble up interrupt signal routing multi thread stack execution context bubble line pipelines.
        raise


# ── Pass 2: Efficiency + Composite ───────────────────────────────────────────


def update_efficiency_and_composite(
    results: list[ExperimentResult],
) -> None:
    """
    Per-source efficiency normalization + composite score.
    MLflow mein log karo + run terminate karo.
    """
    # Security bounds array size validation checks control path branch data fallback redirection check runtime boundary condition trace model statement setup parameters framework checks list.
    if not results:
        return

    # Low level explicit monitoring management client constructor initialize bypassing framework generic state parameters explicit tracker session instance allocations data matrices tracking.
    client = MlflowClient()
    # Dynamic list comprehension parsing loop calculating runtime processing averages per individual file unit mappings results records indicators parameter evaluation tracking equations text layout elements tracking logic array data partitions tracking metrics.
    latency_per_doc: list[float] = [
        (
            r.latency_seconds / r.sampled_doc_count
            if r.sampled_doc_count > 0
            else 0.0
        )
        for r in results
    ]
    # Perform mathematical normalization conversions function call parameters passing arrays floating results return container maps metrics variables configuration layer min max scaler indices.
    efficiency_scores: list[float] = compute_efficiency_score(latency_per_doc)

    # Parallel arrays processing traversal scanning metrics unpacking structures elements collections sequential check index iteration logic context options mapping loop variables metrics tracking elements data maps framework models.
    for result, eff_score in zip(results, efficiency_scores):
        # High level master compound logic configuration formula resolution invoke passing variables calculated scores dimensions return metrics alignment score parameter value tracking indicators dynamic evaluation coefficients.
        composite: float = compute_composite_score(
            density=result.avg_semantic_density,
            boundary=result.boundary_respect_score,
            consistency=result.size_consistency_score,
            efficiency=eff_score,
        )
        # Ingest computed dynamic derived values fields direct inside local object record data attributes references values save trace block path configuration model tracking variables indicators database parameters framework tracking data records.
        result.efficiency_score = eff_score
        # Carry structural calculated final overall matrix data back into destination dataset models variables memory address update indicator parameters data fields array collections metadata matrices update logic structures mapping index.
        result.composite_score = composite

        # Ship derived dynamic performance parameters directly inside server dashboard logging pipeline analytics platform monitoring target updates data check indicators api infrastructure tracking metrics.
        client.log_metric(result.run_id, "efficiency_score", eff_score)
        # Project master benchmark parameters directly remote infrastructure tracking parameters dashboards logging modules execution data parameters write runtime check line log interface api monitoring project analytics pointers logs.
        client.log_metric(result.run_id, "composite_score", composite)
        # Update server backend transaction manager records context flags to mark current sequence lifecycle state fully closed finish status call trace execution path properties parameter central session lifecycle completion mapping.
        client.set_terminated(result.run_id, status="FINISHED")

        # Telemetry analytical report logging update info tracing message console printing results overview verification layout metrics data tracking parameter code write parameters console output print text summaries dynamic logging parameters execution updates.
        logger.info(
            "Pass2 done: %s__%s | efficiency=%.4f | composite=%.4f",
            result.strategy,
            result.source,
            eff_score,
            composite,
        )


# ── Cost Estimate ─────────────────────────────────────────────────────────────


def print_cost_estimate(sample_sizes: dict[str, int]) -> None:
    """
    Run karne se pehle cost estimate print karo.
    User confirm kare — tab hi proceed karo.
    """
    # Console visual border markers separator string format configurations indicators visibility print run tracing logic code layout parameter line write log.
    logger.info("=" * 70)
    # Cost model dashboard title info data summary terminal updates logging traces context parameters visibility mapping statement text details option.
    logger.info("COST ESTIMATE — API Chunking Experiment")
    # Border visual layout design elements strings drawing line parameters console monitors validation pipeline update tracking details trace check framework.
    logger.info("=" * 70)
    # Strategies configuration inventory information notice logs tracking properties parameters displays checking metadata records list trace mapping.
    logger.info("Strategies : semantic, improved_semantic, double_pass")
    # API model baseline costs translation reference configuration numbers properties reading variables lookup parameters data check indicators values context layout.
    logger.info("Model      : text-embedding-3-small @ $0.02 per 1M tokens")
    # White space characters padding styling formatting clear console channel text printing logic options trace monitoring layout.
    logger.info("")

    # Character summaries metrics values initialization container accumulator placeholder variable state memory track allocation code data trace loop.
    total_chars: int = 0
    # Outer dimensional folder scanning matrix looping iteration checking categories paths definitions lists variable mapping dynamic data lookup processing context.
    for source_folder, source_display in SOURCES.items():
        # Fallback dictionary reading verification access operations parameters configurations target values logic mapping tracker metrics look variables.
        n = sample_sizes.get(source_folder, 5)
        # Empirical dataset records scale estimation metadata dictionary parameters check matching structures index reference numbers metrics allocations variables.
        avg_doc_chars = {
            "faa_cfr": 57000,
            "faa_ad": 833,
            "faa_ac": 148000,
            "dgca_car": 34000,
            "skybrary": 1583,
        }.get(source_folder, 5000)
        # Multiply isolated document metrics elements count configuration variables dynamic math computation scale value parameter metrics tracking mapping value.
        source_chars = n * avg_doc_chars
        # Ingestion metrics accumulator tracking summaries update totals values parameters variables mathematics addition process operations trace container map records.
        total_chars += source_chars
        # Typography text alignments dynamic variable spacing formatting calculations fields conversions numbers printing console channel parameters output log.
        logger.info(
            "  %-12s : %2d docs × ~%6d chars = ~%8d chars",
            source_display,
            n,
            avg_doc_chars,
            source_chars,
        )

    # Statistical tokens measurement translation calculations mathematical division operations criteria tracking score rounded off metric conversion variables.
    total_tokens: float = total_chars / CHARS_PER_TOKEN
    # Multiplier constant calculation criteria processing pipelines strategy dimensions loop factor mathematical parameters calculation combination parameters equations trace.
    api_tokens: float = total_tokens * 3 * 2
    # Cost calculations precision float division formula scaling evaluation limits parameters final decimal formatting mapping resource parameters tracking.
    estimated_cost: float = (api_tokens / 1_000_000) * 0.02

    # Character space padding console write parameters details configurations lookups logic trace monitoring context formatting lines string display.
    logger.info("")
    # Metrics analytics console text formatting representation value print monitoring indicators parameter data configurations details visibility level tracing info dashboard.
    logger.info("  Estimated tokens : ~%.0f", api_tokens)
    # Financial metrics validation summary trace data parameters rounding layout logic conversions properties numbers visibility terminal display code trace logging.
    logger.info("  Estimated cost   : ~$%.4f", estimated_cost)
    # Graphics layout outline framework closing visual boundary check mapping template variables console log updates lines layout design terminal border markers.
    logger.info("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────


def run_api_experiments(
    sample_size_override: int | None = None,
    dry_run: bool = False,
) -> None:
    """
    3 API strategies × 5 sources — sequential execution.
    Two-pass: raw metrics → efficiency + composite.
    """
    # Dynamic parameter creation pseudo sequence metrics generation initializer function execution local reference memory context mapping track values seed allocator loop.
    local_random = make_local_random()

    # Build sample sizes
    # Python dictionary comprehensions dictionary inline allocation configurations options keys value parsing lookups conditions loop check variables mapping data lists matrix.
    sample_sizes: dict[str, int] = {
        source: (sample_size_override or DEFAULT_SAMPLE_SIZES[source])
        for source in SOURCES
    }

    # Cost estimate
    # Trigger financial calculation analytics routine method pass configurations mapping parameters variables calculation summary visibility console display path trace layout.
    print_cost_estimate(sample_sizes)

    # Branch routing control conditions check execution check modifiers parameters dynamic logic validation parameter framework trace path context loops options.
    if dry_run:
        # Task bypass terminal updates tracing execution info notice logging status context metrics print pipeline track progress monitor parameters console.
        logger.info("DRY RUN — no actual API calls made.")
        return

    # Inbound server connection setup url tracking paths configurations parameters baseline reading tracking store endpoint configurations layout parameter reference mapping.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # Master workspace directory indexing tracking project workspace register baseline central repository setup configurations initialization project maps data metadata tracking workspace initialization.
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    # System identifier explicit validation code context reference key reading target parameter layout details variable setup parameter lookup storage tracking pointer state checking index.
    experiment_id: str = experiment.experiment_id

    # Global tracking register results lists data properties context sequence matrix array listings tracker database variable memory space storage collection arrays data.
    all_results: list[ExperimentResult] = []

    # Master loops sequence iteration framework traversal loop scanning input folders groups matching folders listings parameter categories metadata mapping dynamic loops track data path.
    for source_folder, source_display in SOURCES.items():
        # Ingest bulk source file documents datasets using system loader parsing subroutine variable returns tracking mapping collection arrays dictionary data configurations lookups.
        all_docs: list[dict[str, Any]] = load_documents(source_folder)
        # Empty inputs checking validation parameters guard condition checks logic branching flow path skip index redirector parameters mapping configurations trace logic.
        if not all_docs:
            # System notification warning messaging output when missing data arrays discovered tracking context rules options metadata variance checks criteria limits.
            logger.warning("No documents found: %s", source_folder)
            # Shift processing loops context indices forward directly into neighboring source folders entries trace path skip iteration branch execution sequence pointer track context.
            continue

        # Extract target numeric slice conditions values read parameters options validation lookup mapping keys dynamic variables reference value config setup data.
        n_sample: int = sample_sizes[source_folder]
        # Ingest randomized subset items documents via subset generator function execution parameter pass values data fields maps returns variables array selection check parameters.
        sampled_docs: list[dict[str, Any]] = sample_documents(
            all_docs, n_sample, local_random
        )
        # Context dynamic insulation validation creation variables setup parameters maps target array lists return structural properties mapping setup data structures mapping context logic.
        doc_content_map: dict[str, str] = build_doc_content_map(sampled_docs)

        # Performance parameters messaging console traces analytics progress reports overview variables metrics configuration console log update indicators pipeline monitoring metrics console print out text.
        logger.info(
            "Source: %s | total=%d | sample=%d | strategies=%d",
            source_display,
            len(all_docs),
            len(sampled_docs),
            len(API_STRATEGIES),
        )

        # Local source calculations results data array lists memory configurations matrix items dictionary dynamic variable processing targets return datasets options mapping.
        source_results: list[ExperimentResult] = []

        # Inner matrix loop sequence scanning distinct API strategies instances parameters configurations definitions options execution look trace step block mapping sequential loops tracker options properties.
        for strategy_name in API_STRATEGIES:
            try:
                # Main single execution target tracker function call, passing dynamic parameters records combinations configurations checking context variables metrics tracking output structures database templates.
                result = run_single_experiment(
                    strategy_name=strategy_name,
                    docs=sampled_docs,
                    doc_content_map=doc_content_map,
                    source_name=source_display,
                    experiment_id=experiment_id,
                    total_doc_count=len(all_docs),
                )
                # Accumulate computed linear execution summaries data maps records direct inside local source metrics collection list arrays variables targets dictionary data items collection process.
                source_results.append(result)
            except Exception as e:
                # Intercept parsing routine deviations or API failures metrics logging console print warnings visibility error trace metadata tracking indicator report layout line print variables execution pipeline context.
                logger.error(
                    "Failed: %s × %s — %s", strategy_name, source_display, e
                )

        # Secondary normalization score balancing dynamic parameters metrics calculation subroutine invoke passing dataset elements components values tracking processing loops context variables.
        update_efficiency_and_composite(source_results)
        # Accumulate completely calculated execution summaries records maps back inside global table tracker database matrix arrays lists values collection master results context mapping loop dynamic.
        all_results.extend(source_results)

    # ── Summary Table ─────────────────────────────────────────────────────────
    # Ordering metadata entries tracking layout parameters using sorting indexing rules parameters lambda properties sequences coordinate layout process loop variable check data trace sorting parameters configurations.
    all_results.sort(key=lambda r: (r.source, -r.composite_score))

    # Visual design separators borders tracking terminal code configurations print execution log layout indicators display trace line block matrix boundaries graphics reporting print code sequence separation line.
    logger.info("\n%s", "=" * 120)
    # Master summary table console banner updates text indicators console output logger verification prints report framework statement logic setup variables trace master dashboard calculation metrics summaries printing.
    logger.info("API CHUNKING EXPERIMENT SUMMARY (sampled)")
    # Border pattern layout separator line drawing standard parameters console logs reporting analytics dashboard configuration graphic structure context variables updates trace mapping line alignment separators divider horizontal dashboard.
    logger.info("=" * 120)
    # Typography padding metrics adjustments dynamic spacing calculations character length alignment formatting fields variables configuration layout setup data definitions write text parameters padding spacing format markers column metrics calculations formatting.
    logger.info(
        f"{'Strategy':<20} {'Source':<12} {'Sample':>7} "
        f"{'Density':>9} {'Boundary':>9} {'Consist':>9} "
        f"{'Effic':>7} {'Composite':>10} {'MaxChunk':>10}"
    )
    # Spacing dividers rendering standard lines console logging presents reports visibility optimization visual formatting clean terminal view code parameters line step layout level tracking reporting console separator bars segment layout separation line chart grids.
    logger.info("-" * 120)

    # Summary results dataset maps traversal element processing configurations loop scanning steps arrays verification variables context output trace logic parameter details check code trace dynamic variable iterations mapping text processing records iterations layout dashboard numbers summary evaluation parameters.
    for r in all_results:
        # Dynamic property parameter numbers variables float conversions configuration padding layout formatting column calculations math console channel direct string updates execution line text visual string alignments formatters layout string format width configurations decimal operations metric prints console.
        logger.info(
            f"{r.strategy:<20} {r.source:<12} {r.sampled_doc_count:>7} "
            f"{r.avg_semantic_density:>9.4f} {r.boundary_respect_score:>9.4f} "
            f"{r.size_consistency_score:>9.4f} {r.efficiency_score:>7.4f} "
            f"{r.composite_score:>10.4f} {int(r.max_chunk_size_chars):>10}"
        )

    # Terminal presentation frame bounding bottom graphics execution line logs analytics indicators status tracking verification code string trace separator frame layout completion markings visual.
    logger.info("=" * 120)

    # Typography template variable string formatting visualization banner overlays main definitions description styling values parameters log terminal console updates trace output text display models template string compound charting dynamic name.
    logger.info("\n🏆 PER-SOURCE WINNERS (API strategies):")
    # Dynamic tracking history verification cache collection parameters initialize lookup key set memory tracking constraints state logic filter execution trace loop cache history registry check validations list.
    seen: set[str] = set()
    # Summary results maps listings scanning items iteration variables sequence data models formatting layout padding measurements metrics print tracking statement records data maps framework.
    for r in all_results:
        # Cross check validation tracking elements arrays visibility verifying if current category has skipped first unique high scoring entry parameters visibility matrix validation mapping check.
        if r.source not in seen:
            # Inject newly indexed category direct inside cache memory tracker collections parameters variable visibility update dynamic validation cache tracking array check trace lookup memory.
            seen.add(r.source)
            # Dynamic tracking property string parameter values float numbers layout column alignment configuration parameters metrics terminal console write data tracking summary logs trace text message info print metrics.
            logger.info(
                "  %-12s → %-20s | composite=%.4f | density=%.4f | boundary=%.4f",
                r.source,
                r.strategy,
                r.composite_score,
                r.avg_semantic_density,
                r.boundary_respect_score,
            )

    # User guidelines tracking paths URLs instructions details terminal updates console trace configuration properties print mapping documentation logs visibility level path line data check manual guidelines directions display accessibility dashboard instructions locations terminal printout navigation guidance reference layouts links data path log text framework.
    logger.info(
        "\nMLflow UI: mlflow ui --backend-store-uri mlruns → http://127.0.0.1:5000"
    )
    # Enterprise project traces index reference analytics pipelines variables dashboard links path logger terminal text output printing layout monitoring trace pipeline parameters config browser trace logging directions options cloud telemetry indices reference dashboard configuration links terminal logs view project trace logging pathway configuration blueprints options.
    logger.info(
        "LangSmith : https://smith.langchain.com → project: %s",
        settings.langchain_project,
    )


# ── Entry Point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Command operational layout interface terminal arguments parser component structure module initial instantiation parameters definition context block tracking lookup setup map trace control execution variables dictionary properties parsing.
    parser = argparse.ArgumentParser(
        description="SkyLex API Chunking Experiment — semantic, improved_semantic, double_pass"
    )
    # CLI command arguments option modifier flag definitions allocation dynamic binary switches data logic true values parameters configuration variables properties tracing step level assignment modifier execution toggles custom modifiers option parameters switches variables checking.
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override sample size for all sources (default: per-source defaults)",
    )
    # Target visual evaluation options switcher command flag properties compilation parameters trace monitoring checklist selector configuration mapping parameters data layout rules option parameters checking.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Cost estimate dikhao — actual API calls mat karo",
    )
    # Input terminal data configurations lookup validation process trigger execution variables parameter collection structure read values parameters lookup setup metrics fields data line execution variables dictionary lookups parser arguments checking evaluation context variables query mappings fetch record storage.
    args = parser.parse_args()
    # Execute master orchestrator loop routing configurations pass properties flags conditions experiments matrix matrix analysis processing pipeline loops execution indicator complete trace run core loop trigger execution pathways core master routine orchestration call parameters passing modifier flags dynamic conditional processing.
    run_api_experiments(
        sample_size_override=args.sample_size,
        dry_run=args.dry_run,
    )