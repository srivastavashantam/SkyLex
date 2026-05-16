"""
processing/chunking_experiment.py

SkyLex Phase 2 — Chunking Strategy Experiment Runner.

5 strategies × 5 sources = 25 MLflow runs.
Agentic strategy sequential chalti hai (rate limit safe).
Baaki 4 strategies ThreadPoolExecutor se parallel chalti hain.

Thread-safety guarantees:
  - Fresh chunker + embedder instance per task (factory pattern)
  - Per-thread random.Random instance (no global state mutation)
  - MLflow MlflowClient explicit run_id (fluent API bypass)
  - matplotlib.figure.Figure() (thread-safe plot generation)
  - tenacity retry + exponential backoff + jitter (rate limit safe)

MLflow UI:
    mlflow ui --backend-store-uri mlruns
    http://127.0.0.1:5000 → experiment: skylex_chunking

LangSmith:
    https://smith.langchain.com → project: skylex

Usage:
    python processing/chunking_experiment.py
    python processing/chunking_experiment.py --skip-agentic
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    AgenticChunker,
    HierarchicalChunker,
    HybridChunker,
    RecursiveChunker,
    SemanticChunker,
)

# Pura system-wide logging configuration initialize ho raha hai taaki terminal aur logs file properly structured format me update ho skein.
setup_logging()
# Multithreaded orchestration environments me isolated logger metrics and pipeline traces catch karne k liye current module context handle setup kiya.
get_logger(__name__)
logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# MLflow UI visualization dashboard par unique experiment run workspace run bundle karne k liye main identity path name set kiya.
EXPERIMENT_NAME: str = "skylex_chunking"
# Central directory file layout management path pointer jahan baseline documentation datasets locate karte hain.
RAW_DATA_DIR: Path = settings.data_raw_dir
# OpenAI tokens window rate protection metrics ceiling cap limit taaki iterative agentic LLM calls out of budget na jayein.
AGENTIC_MAX_DOCS: int = 3
# Empirical mapping formula variable calculation translation ratios characters scale to tokens scale measurement trace.
CHARS_PER_TOKEN: float = 4.0
# Vector transformation boundaries validation context size processing limit max safety parameter threshold constraints mapping setup.
EMBEDDING_MAX_CHARS: int = 6000
# High performance pool executor workers allocation bounds concurrency limits safety control indicators matrix allocation property.
MAX_WORKERS: int = 4
# Linear evaluation reproducibility matrices execution calculations tracking states dynamic configuration constants index initialization.
RANDOM_SEED: int = 42

# Source data folders definitions and descriptive layout names link maps storage setup array values.
SOURCES: dict[str, str] = {
    "faa_cfr": "FAA_CFR",
    "faa_ad": "FAA_AD",
    "faa_ac": "FAA_AC",
    "dgca_car": "DGCA_CAR",
    "skybrary": "SKYBRARY",
}


# ── Retry Decorators ──────────────────────────────────────────────────────────


def _make_embed_with_retry(
    embedder: OpenAIEmbeddings,
) -> Any:
    """
    OpenAI embed_documents ko tenacity retry wrapper mein wrap karo.
    Exponential backoff + jitter — rate limit (429) aur transient errors handle karo.
    Per-call wrapper banate hain — embedder instance thread-isolated rehta hai.
    """

    # Tenacity dynamic optimization mechanism wrapping rule definition layout checking trigger functions.
    @retry(
        # Intercept and match target runtime validation faults or networking dropped state errors verification layer mapping.
        retry=retry_if_exception_type(Exception),
        # Maximum attempts limit tracker boundary bounds counters evaluation context execution rule validation block checking parameters.
        stop=stop_after_attempt(5),
        # Wait mathematical formula adjustment backoff algorithm configuration settings scaling ranges numbers parameters logic.
        wait=wait_exponential_jitter(initial=2, max=60, jitter=4),
        # Propagate error state context straight upwards into core engine system layers failure tracking mapping assertions code flow.
        reraise=True,
    )
    def _embed(texts: list[str]) -> list[list[float]]:
        # Ingestion target lists collections vector generation process client method invoke run execution state pointer variables model trace.
        return embedder.embed_documents(texts)  # type: ignore[return-value]

    # Return freshly wrapped thread safe execution pipeline functional closure reference back to caller routine.
    return _embed


def _make_llm_call_with_retry(fn: Any) -> Any:
    """
    LLM call function ko tenacity retry wrapper mein wrap karo.
    AgenticChunker ke `_extract_propositions` ke liye.
    """

    # Operational errors resiliency orchestration wrapper parameters control criteria evaluation dynamic configuration rule mapping properties.
    @retry(
        # Intercept and map dynamic systemic exception parameters context lookup validation condition parsing checks tracking block code.
        retry=retry_if_exception_type(Exception),
        # API requests processing thresholds counters capacity numbers verification check limit tracker parameters context mapping variable.
        stop=stop_after_attempt(5),
        # Sizing calculations backoff exponential dynamic timing equations rules structure scales configuration boundaries trace model indicators.
        wait=wait_exponential_jitter(initial=2, max=60, jitter=4),
        # Bubble up errors directly inside calling stack execution frameworks failure monitoring sequence parameters trigger map check.
        reraise=True,
    )
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        # Dynamic variable array passing parameters map directly inside functional logic executor processing pipeline call trace execute handler.
        return fn(*args, **kwargs)

    # Return clean encapsulated thread safe callable wrapper functional logic reference straight back to core tracking environment pipeline.
    return _wrapped


# ── Factory Functions ─────────────────────────────────────────────────────────


def make_chunker(strategy_name: str) -> Any:
    """
    Fresh chunker instance per task — no shared state across threads.
    Factory pattern: shared instances = thread-safety risk.
    """
    # String mapping identity variables value evaluations structural pipeline check conditions execution branch logic selector path data.
    if strategy_name == "recursive":
        # Create fresh independent parsing strategy structure properties passing localized parameters context setups model configurations.
        return RecursiveChunker(chunk_size=1000, chunk_overlap=150)
    # Advanced semantic calculations vector transformations lookups condition check matching models selector state routing path branch logic.
    if strategy_name == "semantic":
        # Initialize isolated semantic variance drop check algorithm processing model structure entity configuration return flow context trace.
        return SemanticChunker()
    # Structural document regex boundary checks selector validation data metadata registration parameters path check condition flow mapping.
    if strategy_name == "hierarchical":
        # Enforce maximum safe character threshold dimensions passing rules components data allocations trace code model setup indicator.
        return HierarchicalChunker(max_chunk_size=3000)
    # Hybrid pipeline checking architecture pattern rules configuration parameter attributes selector validation branch check layout context options code.
    if strategy_name == "hybrid":
        # Slicing limit definitions sliding boundaries rules mapping settings class instance allocation return framework component execution flow tracking.
        return HybridChunker(max_chunk_size=1500, chunk_overlap=150)
    # Heavy language parsing model context tracking execution parameters conditions criteria check block dynamic mapper route statement block trace.
    if strategy_name == "agentic":
        # Atomic entities proposition extraction engine processing runtime instantiation fresh instance entity return code execution sequence lifecycle.
        return AgenticChunker()
    # Faulty configuration keys verification validation error indicators exception trigger tracking parameters code trace execution line report.
    raise ValueError(f"Unknown strategy: {strategy_name}")


def make_embedder() -> OpenAIEmbeddings:
    """
    Fresh OpenAIEmbeddings instance per task.
    Shared httpx connection pool = thread-safety risk.
    """
    # Networking tracking dependencies state leaks connection pooling corruptions prevent parameters builder call sequence instantiation execution layer run.
    return OpenAIEmbeddings(
        # Systems metadata config model directory parameters tracking lookup configuration reading parameters scale assignment trace context check variables.
        model=settings.openai_embedding_model,
        # Encapsulated token storage system protection structure validations type annotation options configuration parameters assignment execute line code layer.
        openai_api_key=SecretStr(settings.openai_api_key),  # type: ignore[call-arg]
    )


def make_local_random() -> random.Random:
    """
    Per-thread isolated random.Random instance.
    Global random.seed() = shared state mutation across threads.
    """
    # Standalone isolated entropy tracking pseudo random model engine instance execution initialization metrics allocation seed return values mapping.
    return random.Random(RANDOM_SEED)


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_documents(source_folder: str) -> list[dict[str, Any]]:
    """
    Source folder se saare bulk JSON files load karo.
    hash_registry.json aur _meta.json exclude karo.
    Duplicate doc_ids deduplicate karo — pehli occurrence rakho.
    """
    # Systems filesystem path building configurations pattern string template calculations targeted source directory mapping path tracking code pointer.
    source_dir: Path = RAW_DATA_DIR / source_folder
    # Primary output clean verified dynamic mapping data lists allocation placeholder storage array variable initialization tracker layout block trace.
    all_docs: list[dict[str, Any]] = []
    # Fast memory tracking sets database index duplicate filtering validations execution tracking lookup layer control structures setup state tracking map.
    seen_ids: set[str] = set()

    # Recursive global subdirectory file crawling scanning parameters matching loops search workflow process code execution pipeline trace look data.
    for json_file in sorted(source_dir.rglob("*.json")):
        # Registry logging files bypassing system checking condition metrics filter execution path skip branch rule index context trace parameter mapping.
        if json_file.name == "hash_registry.json":
            continue
        # Descriptors properties description file layout configurations metadata skip checks control parameter optimization sequence tracking setup layer context.
        if json_file.name.endswith("_meta.json"):
            continue
        try:
            # IO systems resources file streams descriptor setup read access options parameters check unicode transformations safe character map streaming.
            with json_file.open(encoding="utf-8") as f:
                # String sequence buffer parsing deserializer module call direct memory target object transformation setup run execution pointer parameters.
                data: Any = json.load(f)
            # Object mapping framework metadata schema tracking verification criteria check loop validation structure data condition rule indicator context type.
            if not isinstance(data, dict) or "documents" not in data:
                continue
            # Data lists payload iteration mapping elements internal data models traverse sequence workflow tracking controls execution loop parameters state data.
            for doc in data["documents"]:
                # Individual data item identity tracker component string reading access lookup safe configurations mapping default check key variable string track.
                doc_id: str = doc.get("doc_id", "")
                # Cross match dynamic verification caches history records list verifying data overlap states rules checking criteria context check layer pointer.
                if doc_id and doc_id not in seen_ids:
                    # Ingest valid newly discovered identifier string key right inside memory framework lookup tracking collection update instantly caching index trace.
                    seen_ids.add(doc_id)
                    # Non overlapping genuine original verified record metadata matrix elements append inside destination tracking array properties variable targets data.
                    all_docs.append(doc)
        except (json.JSONDecodeError, OSError) as e:
            # Intercept syntax formatting violations or disk level read runtime disruptions safely without crashing master multi threaded orchestrator main tracking sequence.
            logger.warning("Could not load %s: %s", json_file, e)

    # Ingestion metrics tracker console log telemetry reports dashboard feedback print summaries datasets parsing metrics check line execution layout trace report.
    logger.info("Loaded %d unique documents from %s", len(all_docs), source_folder)
    # Output isolated completely filtered sanitized clean documentation maps list reference back to core scheduling runner loop processes framework model.
    return all_docs


def build_doc_content_map(
    documents: list[dict[str, Any]],
) -> dict[str, str]:
    """
    doc_id → full original content map.
    Semantic density mein chunk ko actual parent se compare karne ke liye.
    Single Responsibility — load_documents se deliberately alag.
    """
    # Python dictionary comprehensions parsing syntax layout mapping equations calculations properties text variables allocation memory context loop variables lookup trace step.
    return {
        doc["doc_id"]: doc.get("content", "")
        for doc in documents
        if doc.get("doc_id") and doc.get("content")
    }


# ── Metric Calculators ────────────────────────────────────────────────────────


def compute_structural_metrics(
    chunks: list[ChunkedDocument],
) -> dict[str, float]:
    """Chunk size distribution metrics — strategy ke structural behavior."""
    # Blank sequence bounds protection validation parameters check condition branch default mapping variables fallback limits structural logic matrix return path trace.
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

    # Extract single list array integer configurations containing text lengths parameters measurements metrics scale calculations loop metrics data.
    sizes: list[int] = [len(c.content) for c in chunks]
    # Space characters tracking loop parameters validator context counters space clearing filtration values aggregate sum processing tracking variables.
    empty_count: int = sum(1 for c in chunks if not c.content.strip())
    # Distinct original baseline documentation parent reference uniqueness metrics mapping dynamic tracking cache set collection tracker verification index.
    unique_docs: int = len({c.doc_id for c in chunks})
    # Density balancing ratio index division calculations formula rule integration scaling context check metrics parameters layout switch branch tracking step.
    chunks_per_doc: float = len(chunks) / unique_docs if unique_docs > 0 else 0.0

    # Packaging calculated metric parameters floating precision values inside reporting dashboard structure dictionary collections tracking formatters logs.
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
    Chunks jo sentence-ending punctuation pe khatam hote hain unka ratio.
    Higher = better sentence completeness = less mid-sentence cuts.
    Research reference: Shi et al. (2023) — Chunking for RAG evaluation.
    """
    # Vacant item sequence boundary check validation parameters control redirection path mapping trace configuration code assignment rules tracking framework.
    if not chunks:
        return 0.0
    # Central grammatical punctuation standard components markers tracking patterns lookups choices arrays configurations parameters route lookup mapping context setup.
    sentence_endings: tuple[str, ...] = (
        ".", "?", "!", ".'", "?'", "!'", '."', '?"', '!"',
    )
    # Text string end formatting validations checks loop accumulator index counting criteria values configuration mapping trace run indicator tracking.
    complete: int = sum(
        1 for c in chunks if c.content.strip().endswith(sentence_endings)
    )
    # Precision decimal calculation rules metrics constraints rounding formulas conversion values return flow layout matrix tracking indices configuration parameters.
    return round(complete / len(chunks), 4)


def compute_semantic_density(
    chunks: list[ChunkedDocument],
    doc_content_map: dict[str, str],
    embedder: OpenAIEmbeddings,
    local_random: random.Random,
) -> float:
    """
    Average cosine similarity — chunk vs actual parent document.

    Dynamic sample size: min(50, max(5, 5% of chunks))
    Per-thread random.Random — no global state mutation.
    Token-safe truncation — 6000 chars (text-embedding-3-small: 8191 tokens).
    Tenacity retry — rate limit aur transient errors handle karo.
    No data leak — chunk ko apne aap se compare nahi karte.
    """
    # Guardrail checks conditions verification parameters lookup path mapping validation fallback redirection target logic structure variables check data trace step.
    if not chunks or not doc_content_map:
        return 0.0

    # Scale percentage dynamic metrics variable formulations equations calculations checking logic scale values matching indicators parameter allocation trace code.
    sample_size: int = min(50, max(5, int(len(chunks) * 0.05)))
    # Completely thread decoupled unbiased randomized selection array allocation slice window collection processing target trigger execution execution model.
    sample: list[ChunkedDocument] = local_random.sample(
        chunks, min(len(chunks), sample_size)
    )

    # Valid verified processing element configurations text combinations repository tracking variables maps list collection initialization variable storage location.
    valid_pairs: list[tuple[str, str]] = []
    # Scanning elements trajectory data looping tracking items attributes properties sequence processing runtime matrix step execution flow handler block layout trace.
    for chunk in sample:
        # Source target indexing map reading context lookup extraction properties variables parameters safe defaults identification properties values track data trace lookup.
        parent_content: str = doc_content_map.get(chunk.doc_id, "")
        # String data content lengths availability checking structural criteria path filter execution sequence check control branch trace layer layout condition code.
        if parent_content:
            # Absolute token constraints protection parameter boundaries slice character capacity mapping assignment variable collection insertion execution parameters context.
            valid_pairs.append((
                chunk.content[:EMBEDDING_MAX_CHARS],
                parent_content[:EMBEDDING_MAX_CHARS],
            ))

    # Empty validation tracking check verifying if matching elements sets has failed structural array counts context data pipeline safety pointer check logic trace.
    if not valid_pairs:
        logger.warning("No valid chunk-parent pairs — semantic density returning 0.0")
        return 0.0

    # Segregating chunk elements array values records inside dedicated linear textual extraction mapping inputs arrays processing models lists context variables parsing.
    chunk_texts: list[str] = [p[0] for p in valid_pairs]
    # Isolate parent original historical baseline documents sequence lines layouts data map integrity checks arrays parameters calculation model configurations trace options.
    parent_texts: list[str] = [p[1] for p in valid_pairs]

    # Dynamically inject tenacity retry resilience layer wrapper right on top of the thread-isolated embedding engine instance function pointer variables setup tracking mapping.
    embed_fn = _make_embed_with_retry(embedder)

    try:
        # Request outbound remote provider network embedding vector array matrix calculations using dynamic retry wrapped routine execution result arrays return trace list data tracking parameters.
        chunk_embeddings: list[list[float]] = embed_fn(chunk_texts)
        # Parent records mathematical representation dimensions calculation model code pipeline address call layout execution data values results arrays collection parameters tracking options.
        parent_embeddings: list[list[float]] = embed_fn(parent_texts)

        # Distance similarity arrays registry storage context target metrics track tracking array allocation initialization code variables calculations loop parameters setup metrics model.
        similarities: list[float] = []
        # Multi-variable loop tracking traversing parallel vector arrays coordinates data unpack trace execution validations sequence check control iterations blocks tracking look trace matrix.
        for c_emb, p_emb in zip(chunk_embeddings, parent_embeddings):
            # Convert basic array list floats into high speed mathematical matrix processing numpy structures layout tracking objects parameters values design rules calculation trace model.
            c_arr = np.array(c_emb)
            # Alignment matching checking numerical checks validation properties configuration map elements inside high efficiency numpy object calculations parameters check runtime log trace data check.
            p_arr = np.array(p_emb)
            # Vector norm dimensions square matrix multiplier calculations structural magnitude coefficients index resolution safety check variable trace formula execution data layer context rules.
            norm_product: float = float(
                np.linalg.norm(c_arr) * np.linalg.norm(p_arr)
            )
            # Zero division protection boundaries exceptions branching path skip execution context loop controls validation tracker layer tracking code logic framework operations trace variables.
            if norm_product < 1e-10:
                continue
            # Distance matrix resolution linear dot multiplier products logic formula resolution score parameter validation dynamic mapping data variable list append track indicator data flow checks process.
            similarities.append(float(np.dot(c_arr, p_arr) / norm_product))

        # Yield computed statistical average mapping metrics output coefficients values configuration layout calculation trace return result context data framework logs updates parameters.
        return round(float(np.mean(similarities)), 4) if similarities else 0.0

    except Exception as e:
        # Capture API connection crashes, authentication timeouts, or unhandled token data parsing issues metrics warnings tracking traces indicators screen output logs warning updates tracking.
        logger.warning("Semantic density failed: %s", e)
        # Secure failure mode path redirection standard fallback default values tracking parameters setup execution branch check metrics layout flow trace runtime code recovery options pathway index.
        return 0.0


# ── Artifact Generators ───────────────────────────────────────────────────────


def save_chunk_size_histogram(
    chunks: list[ChunkedDocument],
    strategy: str,
    source: str,
    tmp_dir: str,
) -> str:
    """
    Thread-safe histogram — matplotlib.figure.Figure() directly use karo.
    plt global state thread-safe nahi — Figure object isolated hai.
    """
    # Extract structural length dimension properties loop comprehension tracking capacity character variables lists calculations metrics elements array length measuring metric calculations variables trace.
    sizes: list[int] = [len(c.content) for c in chunks]
    # Running statistical analysis calculations formula computing exact distribution pattern midpoint evaluation values track trace parameters property configuration setup model parameters.
    mean_size: float = float(statistics.mean(sizes))

    # Thread safe fully isolated decoupled canvas tracking architecture plot generation instance class invocation context setup mapping variables parameters trace log code step execution line layout.
    fig = matplotlib.figure.Figure(figsize=(10, 5))
    # Coordinate grid frame manager allocation section mapping layout active index object generation add execution layout variables parameters check code runtime layer processing frame management axis.
    ax = fig.add_subplot(1, 1, 1)
    # Chart plotting data graphics mapping parameters color specifications borders styling layout alpha frequency density tracking data blocks canvas trace draw context parameters design settings.
    ax.hist(sizes, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    # Typography template variable string formatting visualization title overlays main definitions description styling values parameters log terminal console updates trace output text display models template string.
    ax.set_title(
        f"Chunk Size Distribution — {strategy.upper()} | {source}",
        fontsize=14,
        fontweight="bold",
    )
    # Horizontal canvas layout parameters naming labels text specifications design metrics styling criteria parameter code target maps tracking level data line alignments rules configurations parameters.
    ax.set_xlabel("Chunk Size (chars)", fontsize=12)
    # Vertical coordinate sequence tracking metric level parameter label details formatting string text dashboard configuration settings write analytics values trace runtime logic metrics level parameters check.
    ax.set_ylabel("Frequency", fontsize=12)
    # Overlay reference line markers plotting statistical center calculations dash format setup indicators variables balance display path alignment logic code trace statement parameters mapping lines design boundaries.
    ax.axvline(
        mean_size,
        color="#DD4444",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_size:.0f} chars",
    )
    # Legend description layout panel canvas visibility activation parameters trigger maps text details styling parameters tracking method call trace code logic updates visibility panel interface dashboard.
    ax.legend(fontsize=11)
    # Spacing margins calculation auto optimization bounding compress chart layouts tidy graphics canvas drawing save output processing layer code run step rendering alignments formatting canvas space.
    fig.tight_layout()

    # Dynamic target file system repository destination path layout string naming templates mappings parameters properties location variable string data look files directory locations naming conventions.
    plot_path: str = f"{tmp_dir}/{strategy}_{source}_chunk_dist.png"
    # Write chart binary sequence streams directly onto active system local disk persistence track image format rendering file saving action execution indicator code block log data stream write maps.
    fig.savefig(plot_path, dpi=150)
    # Return destination mapping folder storage address pointer variable direct back to metadata uploading transaction scheduler system modules interface logic path analytics tracking logs.
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
    Stratified random sample of N chunks — qualitative inspection ke liye.
    Per-thread local_random — global state mutation nahi.
    """
    # Random metrics selection indices calculations pool tracking array selection window slice elements logic lists variables extraction tracking execution routine code step processing parameters allocation window.
    sample: list[ChunkedDocument] = local_random.sample(
        chunks, min(len(chunks), n)
    )
    # Data mapping format conversion subroutine keys values configuration mapping structures transformation direct inside serializable dataset dictionaries lists storage variables check properties fields layout model.
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
    # Destination location coordinate string template formatting configuration fields address variables mapping setup indicators tracking file variables check layout address layout destination setup naming.
    json_path: str = f"{tmp_dir}/{strategy}_{source}_sample_chunks.json"
    # File descriptor handler channel allocation, enabling local resource path writing configurations conversions processing parameters integrity checks code path modes enabled unicode safe character conversions.
    with open(json_path, "w", encoding="utf-8") as f:
        # Packing serializations writing data maps records json data dump operations layouts style indentation properties variable configurations flow trace code run level packing data serialization metrics formatters.
        json.dump(output, f, indent=2, ensure_ascii=False)
    # Return saved destination address pathway character string direct back to tracking uploading manager integration modules context interface framework trace output files properties variable address.
    return json_path


# ── Single Experiment Runner ──────────────────────────────────────────────────


def run_single_experiment(
    strategy_name: str,
    docs: list[dict[str, Any]],
    doc_content_map: dict[str, str],
    source_name: str,
    experiment_id: str,
    is_agentic: bool = False,
) -> dict[str, Any]:
    """
    Ek strategy × source combination ka experiment.

    Thread-safety:
    - Fresh chunker via make_chunker() — no shared state
    - Fresh embedder via make_embedder() — no shared httpx pool
    - Fresh random.Random via make_local_random() — no global mutation
    - MlflowClient explicit run_id — no fluent API thread-local context
    - tenacity retry — rate limit aur transient errors handle
    - try/finally — MLflow run always terminated (FINISHED or FAILED)
    """
    # Composite dashboard visualization indexing label template string definitions parameter combinations mapping configuration indicator target code values trace value identity formulation properties.
    run_name: str = f"{strategy_name}__{source_name}"
    # Performance telemetry track notification logging info print process run monitoring console update terminal trace metrics display layout block line text tracking variables verification options.
    logger.info("Starting: %s | docs: %d", run_name, len(docs))

    # Initialize low-level backend monitoring management interface client object completely unbinding standard framework contextual tracking thread state parameters explicit transaction tracker system.
    client = MlflowClient()
    # Invoke backend monitoring systems directory to allocate unique isolated database recording track element context model maps setup value mapping trace context transaction request server dashboard.
    run = client.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={
            "mlflow.runName": run_name,
            "run_type": "chunking_experiment",
            "phase": "phase_2",
            "strategy": strategy_name,
            "source": source_name,
            "agentic_sampled": str(is_agentic),
        },
    )
    # Technical transaction identification sequence reference code string tracking identifier parameter location updates reading parameter value data mapping variable database unique transaction parameters.
    run_id: str = run.info.run_id

    try:
        # Fresh instances — no shared state across threads
        # Strategy localized object instantiation subroutine factory design execution passing targets schema context variable values indicator check layout mapping track properties model allocation.
        chunker = make_chunker(strategy_name)
        # Vector spatial metrics translation engine builder instantiation function invoke configurations criteria mapping setup call level check tracking code pointer line core modeling parameter mappings.
        embedder = make_embedder()
        # Randomized parameters selection entropy device tracker instantiation function execution isolation variables assignment sequence validation setup code tracing metrics evaluation random state.
        local_random = make_local_random()

        # Log params
        # Properties configuration collection parameters storage inventory listings dictionary schema definition mapping parameters setup data variable tracing tracker values checklist parameters dynamic inventory.
        params: dict[str, Any] = {
            "strategy": strategy_name,
            "source": source_name,
            "doc_count": len(docs),
        }
        # Dynamic property lookup check checking variable validity condition branch routing validation parameters path selection validation track properties look variables code layer mapping branch.
        if hasattr(chunker, "chunk_size"):
            params["chunk_size"] = chunker.chunk_size
        # Dynamic checking property attribute lookup validation check parameters values tracking condition execution code block data variable trace look step block mapping structural setups evaluation context.
        if hasattr(chunker, "chunk_overlap"):
            params["chunk_overlap"] = chunker.chunk_overlap
        # Structural boundary indices constraints constraints limit parameter configurations structural check variables code tracking setup parameter value mapping asset pointer variance constraints variables.
        if hasattr(chunker, "max_chunk_size"):
            params["max_chunk_size"] = chunker.max_chunk_size
        # Multi dynamic cutoff algorithm parameter validation check properties check configurations models variable values data parameter mapping criteria tracking trace boundary settings calculations rules.
        if hasattr(chunker, "breakpoint_threshold_type"):
            params["breakpoint_threshold_type"] = chunker.breakpoint_threshold_type
        # Mathematical scaling computations options indicators validity checks variable parameters code tracking updates layout storage mapping properties verification trace intensity variables calculations.
        if hasattr(chunker, "breakpoint_threshold_amount"):
            params["breakpoint_threshold_amount"] = chunker.breakpoint_threshold_amount

        # Transmit single structural configuration items sequence straight into distant database logs server tracking framework telemetry upload loop api code execution layer metadata registration.
        for key, value in params.items():
            client.log_param(run_id, key, value)

        # Chunk + measure latency
        # High resolution hardware platform chronometer checkpoint snapshot register timing sequence benchmark evaluation track runtime interval delta start track pointer line clock execution baseline.
        start_time: float = time.perf_counter()
        # Polymorphic functional procedure orchestration layout text content arrays computational loops text segmentation data arrays mapping return computation trace variable text parsing strategy model.
        chunks: list[ChunkedDocument] = chunker.chunk(docs, deduplicate=True)
        # Runtime calculation results time window formatting parameters float decimal limits parameters difference tracking indicator score value save data trace loop level metrics hardware.
        latency: float = round(time.perf_counter() - start_time, 3)

        # Compute metrics
        # Statistical data layout check metrics parsing calculations function invocation datasets input array properties maps return configuration code validation layer structure parameters calculations.
        structural: dict[str, float] = compute_structural_metrics(chunks)
        # Sentence segmentation formatting compliance index indicator calculation routine call target matrix variables register tracking validation console indicator trace step accuracy measurement scores.
        boundary_score: float = compute_boundary_respect_score(chunks)
        # Semantic semantic space distance matrix mapping values calculation method call passing variables calculations validation vector dimension levels indicators tracking step context fidelity checks.
        semantic_density: float = compute_semantic_density(
            chunks, doc_content_map, embedder, local_random
        )

        # Log metrics
        # Packaging all computed tracking values calculations into unified dictionary structures payload data stream for direct monitoring database pipeline storage cloud tracking server payloads.
        all_metrics: dict[str, float] = {
            **structural,
            "boundary_respect_score": boundary_score,
            "avg_semantic_density": semantic_density,
            "latency_seconds": latency,
        }
        # Iterate over formatting metrics mapping elements data charts to send variables numbers straight inside dynamic infrastructure metrics logger tracker interface line server dashboard charts.
        for key, value in all_metrics.items():
            client.log_metric(run_id, key, value)

        # Log artifacts
        # Environment system dynamic transient space virtual directory storage manager wrapper block configuration automatic cleanup rules tracking pipeline path trace context execution file system resource operations.
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create chart distribution visualizations local files location strings parsing variables tracking canvas run data trace line execution model graphics chart visualization.
            hist_path = save_chunk_size_histogram(
                chunks, strategy_name, source_name, tmp_dir
            )
            # Create text database items serialized storage structures validation mapping check destination path records files write update indicators check trace data structure previews information check.
            json_path = save_sample_chunks_json(
                chunks, strategy_name, source_name, tmp_dir, local_random
            )
            # Send analytics plotting image artifacts directly inside master monitoring persistent folder storage pathway upload execute tracking api run trace line code server graphics pipeline uploads.
            client.log_artifact(run_id, hist_path, artifact_path="plots")
            # Quality assessment validation verification items lists format specifications details parameters tracking update structural telemetry upload sequence route api data logic text preview formats maps.
            client.log_artifact(
                run_id, json_path, artifact_path="sample_chunks"
            )

        # Update status configurations parameters flags metadata data sets indicators inside distant database transaction record state finish signals execute method call runtime lifecycle completion flags.
        client.set_terminated(run_id, status="FINISHED")
        # Technical engineering trace logging status complete confirmation messaging terminal write updates performance details indicators metrics console print summary block loop execution parameters reporting.
        logger.info(
            "Done: %s | chunks: %d | latency: %.2fs | density: %.4f | boundary: %.4f",
            run_name, len(chunks), latency, semantic_density, boundary_score,
        )

        # Output overall summary evaluation variables records fields maps back into master monitoring scheduler execution system framework details return flow layer consolidated tracking parameters.
        return {
            "strategy": strategy_name,
            "source": source_name,
            "doc_count": len(docs),
            **structural,
            "boundary_respect_score": boundary_score,
            "avg_semantic_density": semantic_density,
            "latency_seconds": latency,
        }

    except Exception as e:
        # Unexpected script faults interception safeguards flags parameter updates tracking crash terminate execution signature framework pipeline execution trace call code block runtime exceptions protection.
        client.set_terminated(run_id, status="FAILED")
        # System failures evaluation reporting error tracking message strings write console diagnostics metrics overview streams monitor parameters log text trace runtime fault analytics.
        logger.error("Experiment failed: %s — %s", run_name, e)
        # Propagating tracking exception data models back towards master pool queues hierarchy lifecycle control execution thread bubbles indicator code stream line exception bubble up.
        raise


# ── Main ──────────────────────────────────────────────────────────────────────


def run_all_experiments(skip_agentic: bool = False) -> None:
    """
    25 MLflow runs — 5 strategies × 5 sources.

    Execution model:
    - Recursive, Semantic, Hierarchical, Hybrid: ThreadPoolExecutor (MAX_WORKERS=4)
    - Agentic: sequential (heavy API calls — parallel = rate limit risk)
    """
    # Sync core monitoring environments master tracing indicators using dynamic project configurations path tracking storage registration project initial context validation tracker name backend store links.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # Central baseline index workspace folder instantiation project tracking registration database fields properties metadata values mapping indicator data lookup check line setup master configuration repository.
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    # Internal dashboard system entry identifier tracking reference values variable reading destination context tracking records variables check parameter lookup code tracking step unique experiment identity keys.
    experiment_id: str = experiment.experiment_id

    # Non-agentic strategies — parallel safe
    # Concurrency safe non-blocking data operations strategies validation variables lists parameters arrays schema listings models items configurations checklist trace array values matrix testing checks.
    parallel_strategies: list[tuple[str, bool]] = [
        ("recursive", False),
        ("semantic", False),
        ("hierarchical", False),
        ("hybrid", False),
    ]

    # Global compilation summary results lists tracking data map properties metrics arrays final storage variable pointer trace calculation records list memory allocation master results collection register.
    all_results: list[dict[str, Any]] = []

    # Master sequence iteration traversal loop scanning content repositories groups matching directory paths listings folder options elements data trace tracking variable mapping matrix dimensions loop.
    for source_folder, source_display in SOURCES.items():
        # Ingest overall dataset content text strings configurations dictionary array records using localized load reading routine outputs variable array check memory trace parameters data extraction loader utility.
        docs: list[dict[str, Any]] = load_documents(source_folder)
        # Empty variables checklist security guardrails parameters checking condition branch validation context loop bypass index forwarding layout dynamic tracking rules trace step rule check validation filter boundaries.
        if not docs:
            # System notification warning level indicators console tracking triggers when vacant inputs tracking datasets encountered context checks options variable mapping limits tracking missing sources message.
            logger.warning("No documents found: %s", source_folder)
            # Forward processing master scheduling tracking index loops directly into adjacent source groups entries tracking bypass branch iteration trace execution block line level loops redirector pointer.
            continue
        # Context dynamic injection leakage protection mapping data validation processing subroutine parameter pass collections layout lists results tracking details return data flow loop insulation data setup logic.
        doc_content_map: dict[str, str] = build_doc_content_map(docs)

        # ── Parallel: 4 non-agentic strategies ───────────────────────────────

        # Concurrency asynchronous thread manager pools setups configuration, limits indicator allocation tracking workflow context block initiation trigger multi processing execute layout thread clusters allocation loops.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map dynamic items loop comprehension submitting tasks straight inside worker thread streams returning execution identifiers matrix tracking lists variables array collections setup code parallel thread scheduler mapping.
            futures = {
                executor.submit(
                    run_single_experiment,
                    strategy_name=strategy_name,
                    docs=docs,
                    doc_content_map=doc_content_map,
                    source_name=source_display,
                    experiment_id=experiment_id,
                    is_agentic=is_agentic,
                ): strategy_name
                for strategy_name, is_agentic in parallel_strategies
            }
            # Active tracking pooling event checkpoints traversal checking variable data fetch iterations validation status updates trace tracking variables configuration loop look check layer completion checking routines.
            for future in as_completed(futures):
                # Fetch localized parsing strategy tracking label template string matching registry references variables mappings layout indices properties look text context check future tasks completions tracker.
                strategy_name = futures[future]
                try:
                    # Ingest completely processed tracking table dataset record items out of thread data parameters channel save array collections list variables storage memory output metrics results collection.
                    all_results.append(future.result())
                except Exception as e:
                    # Capture thread execution failures tracking logic errors options structural configurations trace parameters data exceptions print console dashboard reporting metrics code concurrency failures report.
                    logger.error(
                        "Failed: %s × %s — %s",
                        strategy_name, source_display, e,
                    )

        # ── Sequential: Agentic ───────────────────────────────────────────────

        # Cost tracking budget boundary parameters checklist condition validation variable switches control execution logic pattern configurations mapping context rules setup parameter track check financial parameters safety flag.
        if not skip_agentic:
            try:
                # Running single standalone execution tracking routine logic call parameters components passing tracking variable schemas database models reports layout indicator execution pointer sequence execution task.
                result = run_single_experiment(
                    strategy_name="agentic",
                    docs=docs[:AGENTIC_MAX_DOCS],
                    doc_content_map=doc_content_map,
                    source_name=source_display,
                    experiment_id=experiment_id,
                    is_agentic=True,
                )
                # Accumulate compiled metrics data maps records right inside primary global matrix arrays lists items structural summary collection variables database trace values data continuous results mapping arrays.
                all_results.append(result)
            except Exception as e:
                # Intercept parsing routine errors or API exceptions console logs warning levels visibility error track trace metadata tracking indicator report layout line print variables execution pipeline exceptions.
                logger.error("Agentic failed: %s — %s", source_display, e)

    # ── Summary Table ─────────────────────────────────────────────────────────
    # Sequence sorting parameters arrangement configurations index data items sorting rules criteria lambda variables tracking process check execution logic flow model trace block metrics sorting formulas list.
    all_results.sort(key=lambda r: (r["source"], r["strategy"]))

    # Visual alignment console tracking design borders drawing standard patterns line text log execution parameters visualization track metrics block layout graphics print code sequence separation line layout graphics.
    logger.info("\n%s", "=" * 110)
    # Master dashboard evaluation summary title terminal printout logger status tracking message details metrics verification output console trace variable dataset mapping text level final console overview banner.
    logger.info("CHUNKING EXPERIMENT SUMMARY")
    # Border pattern layout separator line drawing standard parameters console logs reporting analytics dashboard configuration graphic structure context variables updates trace mapping line alignment separators divider.
    logger.info("=" * 110)
    # Typography padding metrics adjustments dynamic spacing calculations character length alignment formatting fields variables configuration layout setup data definitions write text parameters padding spacing format markers.
    logger.info(
        f"{'Strategy':<15} {'Source':<12} {'Docs':>5} {'Chunks':>7} "
        f"{'Avg Size':>9} {'Std':>7} {'Density':>9} {'Boundary':>9} {'Latency':>9}"
    )
    # Spacing dividers rendering standard lines console logging presents reports visibility optimization visual formatting clean terminal view code parameters line step layout level tracking reporting console separator bars.
    logger.info("-" * 110)
    # Summary results dataset maps traversal element processing configurations loop scanning steps arrays verification variables context output trace logic parameter details check code trace dynamic variable iterations mapping.
    for r in all_results:
        # Dynamic property parameter numbers variables float conversions configuration padding layout formatting column calculations math console channel direct string updates execution line text visual string alignments formatters.
        logger.info(
            f"{r['strategy']:<15} {r['source']:<12} {r['doc_count']:>5} "
            f"{int(r['chunk_count']):>7} {r['avg_chunk_size_chars']:>9.0f} "
            f"{r['std_chunk_size_chars']:>7.0f} {r['avg_semantic_density']:>9.4f} "
            f"{r['boundary_respect_score']:>9.4f} {r['latency_seconds']:>9.2f}s"
        )
    # Terminal display structure frame closing layout canvas complete validation marker metrics logs trace compilation tracking validation setup templates code mapping framework indicator line visual footer border markings.
    logger.info("=" * 110)
    # User guidelines tracking paths URLs instructions details terminal updates console trace configuration properties print mapping documentation logs visibility level path line data check manual guidelines directions display.
    logger.info(
        "MLflow UI: mlflow ui --backend-store-uri mlruns → http://127.0.0.1:5000"
    )
    # Enterprise project traces index reference analytics pipelines variables dashboard links path logger terminal text output printing layout monitoring trace pipeline parameters config browser trace logging directions options.
    logger.info(
        "LangSmith : https://smith.langchain.com → project: %s",
        settings.langchain_project,
    )


# ── Entry Point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Inbound options parsing command management standard configuration interface framework context validation variables tracking target setup parameter initialization tracking lookup terminal parser handler blueprints.
    parser = argparse.ArgumentParser(
        description="SkyLex Chunking Experiment Runner"
    )
    # CLI command arguments option modifier flag definitions allocation dynamic binary switches data logic true values parameters configuration variables properties tracing step level assignment modifier execution toggles.
    parser.add_argument(
        "--skip-agentic",
        action="store_true",
        help="AgenticChunker skip karo — OpenAI API cost bachane ke liye",
    )
    # Input terminal data configurations lookup validation process trigger execution variables parameter collection structure read values parameters lookup setup metrics fields data line execution variables dictionary lookups.
    args = parser.parse_args()
    # Execute master orchestrator loop routing configurations pass properties flags conditions experiments matrix matrix analysis processing pipeline loops execution indicator complete trace run core loop trigger execution pathways.
    run_all_experiments(skip_agentic=args.skip_agentic)