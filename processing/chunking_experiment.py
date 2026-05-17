"""
processing/chunking_experiment.py

SkyLex Phase 2 — Chunking Strategy Experiment Runner.

9 strategies × 5 sources = 45 MLflow runs.
(AgenticChunker excluded — cost/latency not justified at this stage)

Strategies compared:
    Original  : recursive, semantic, hierarchical, hybrid
    Improved  : improved_recursive, improved_semantic, improved_hybrid
    New       : structure_aware, double_pass

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

# Logging infrastructure and terminal setups bootstrap execute kiya jaa raha hai context telemetry sync ke liye.
setup_logging()
# Multithreaded orchestration layout variables maps debugging matrix ke tracking traces isolate karne ke liye current module standard instance allocate kiya.
logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# MLflow workspace central parameters management dashboard tracking project registration identifier name set.
EXPERIMENT_NAME: str = "skylex_chunking"
# Central systems repository directory path jahan dynamic unsegmented raw files databases schemas structures ready located hain.
RAW_DATA_DIR: Path = settings.data_raw_dir
# Standard character count calculation statistical factor variables calculation rules transformation ratio metrics mapping.
CHARS_PER_TOKEN: float = 4.0
# Core vector conversion limits overflow mitigation specifications conditions parameters safety range capping threshold capacity index.
EMBEDDING_MAX_CHARS: int = 6000
# Concurrency executor pool ceilings thresholds performance control limiters numbers counters settings allocations track parameters mapping.
MAX_WORKERS: int = 4
# Absolute matrices mathematical reproducibility state validation trace sync configurations constants variables registry token setup.
RANDOM_SEED: int = 42

# Dynamic evaluation data collections points directory mapping values labels dashboard mapping configuration variables text layouts.
SOURCES: dict[str, str] = {
    "faa_cfr": "FAA_CFR",
    "faa_ad": "FAA_AD",
    "faa_ac": "FAA_AC",
    "dgca_car": "DGCA_CAR",
    "skybrary": "SKYBRARY",
}

# OpenAI endpoints validation tracking checking metrics tokens usage optimization blocks rate limits prevention classification rules parameters registry sets.
API_BOUND_STRATEGIES: set[str] = {
    "semantic",
    "improved_semantic",
    "double_pass",
}

# Concurrency thread isolated parallel safe execution algorithms configuration parameters lists sequences models listings targets checklist tracker array layout.
NON_API_STRATEGIES: list[str] = [
    "recursive",
    "hierarchical",
    "hybrid",
    "improved_recursive",
    "improved_hybrid",
    "structure_aware",
]


# ── Retry Decorators ──────────────────────────────────────────────────────────


def _make_embed_with_retry(embedder: OpenAIEmbeddings) -> Any:
    """
    OpenAI embed_documents ko tenacity retry wrapper mein wrap karo.
    Exponential backoff + jitter — rate limit (429) handle karo.
    """

    # Multi stage retry architecture closure function dynamic configurations properties parameters control metrics layer mapping runtime logic flow parameters.
    @retry(
        # Intercept dynamic network connections dropped states ya pipeline format failures conditions check parsing validation rules type checking path branch.
        retry=retry_if_exception_type(Exception),
        # Endpoint transaction restrictions capacity limits max threshold counters tracking execution checks validation boundaries parameter.
        stop=stop_after_attempt(5),
        # Mathematical interval updates calculation wait backoff metrics jitter variance logic scale constraints limits timing calculation equations block maps runtime.
        wait=wait_exponential_jitter(initial=2, max=60, jitter=4),
        # Fault isolation assertions error code context updates direct bubble up parameters routing layout validation code trace context pipeline.
        reraise=True,
    )
    def _embed(texts: list[str]) -> list[list[float]]:
        # Inbound raw lists collection array text mapping properties direct vector processing client method trigger invoke variables tracking memory look run indicator.
        return embedder.embed_documents(texts)  # type: ignore[return-value]

    # Clean retry wrapped functional callback method references return mapping pipeline architecture execution block wrapper level pointer context trace.
    return _embed


# ── Factory Functions ─────────────────────────────────────────────────────────


def make_chunker(strategy_name: str) -> Any:
    """
    Fresh chunker instance per task — no shared state across threads.
    Factory pattern: shared instances = thread-safety risk.
    """
    # Dynamic parameter structural selection comparison logic checking condition branching selector routing variables path code layer matrix check layout.
    if strategy_name == "recursive":
        # Baseline limits configuration metrics properties values allocations parameters pass initialize fresh strategy class execution trace pointer return.
        return RecursiveChunker(chunk_size=1000, chunk_overlap=150)
    # Advanced semantic distance matrix mapping configurations values comparison branch properties lookup tracking variables rule options selector context trace block.
    if strategy_name == "semantic":
        # Localized vector cluster boundaries drop index calculator core layer class instantiation component configuration mapping dynamic initialization return workflow trace.
        return SemanticChunker()
    # Structural rule extraction formatting validations regex separation check metrics mapping setup variables criteria checklist branch line parameter route options data.
    if strategy_name == "hierarchical":
        # Target sizing thresholds restrictions safety upper capacity constraints pass configuration parameters model data variable allocation direct instance code model block trace.
        return HierarchicalChunker(max_chunk_size=3000)
    # Combined hybrid orchestration data flow criteria matching variables validation check properties parameter layout settings selector branching rule setup layer code.
    if strategy_name == "hybrid":
        # Slicing limits parameters configuration sliding boundary properties definition setup memory instance allocation return framework components tracking flow layout.
        return HybridChunker(max_chunk_size=1500, chunk_overlap=150)
    # Custom optimization baseline strategies matching variable value criteria checks condition branching select execution tracking setup pointer model trace layout check fields.
    if strategy_name == "improved_recursive":
        # Advanced character ranges limits parameters layout tracking data boundaries adjustments pass fresh structural configurations variable local setup return trigger.
        return ImprovedRecursiveChunker(chunk_size=1000, chunk_overlap=200)
    # Optimization semantic drops boundary check conditions selector properties mapping runtime configuration tracking trace validation properties map model step.
    if strategy_name == "improved_semantic":
        # Lower threshold constraints validation multi stage calculation strategy record model fresh instance allocate return structural workflow metrics track indicators data context.
        return ImprovedSemanticChunker()
    # Hybrid pipeline scale balance settings checking architecture patterns rules parameter configuration attributes validation branch lookup trace context details layout code layer.
    if strategy_name == "improved_hybrid":
        # Section capacity bounds threshold sliding memory segments metrics preservation definitions setup allocation values layout return pipeline track flow framework.
        return ImprovedHybridChunker(max_chunk_size=2000, chunk_overlap=200)
    # Advanced metadata injection prefix contextual orchestration strategy processing rules condition matching model selector state checkpoint trace parameters context call layer.
    if strategy_name == "structure_aware":
        # Sizing limits variables parameter tracking constraints safety upper bounds capacity values configuration model records lists return layout target setup indicator execution sequence context trace code line.
        return StructureAwareChunker(max_chunk_size=1800, chunk_overlap=180)
    # Multi pass hybrid coarse semantic recursive fine layout checking validation structural pipeline checking rule model constructor execution parameters branch tracking matrix data context code.
    if strategy_name == "double_pass":
        # Isolated dual processing phase strategy engine constructor instance execution initialization dynamic return code execution complete layout parameters map logic tracker framework models properties.
        return DoublePassChunker()
    # Inbound wrong string parameters safeguards verification error indicators specifications exception throw codes validation tracking data layout check reporting trace line context message.
    raise ValueError(f"Unknown strategy: {strategy_name}")


def make_embedder() -> OpenAIEmbeddings:
    """
    Fresh OpenAIEmbeddings instance per task.
    Shared httpx connection pool = thread-safety risk.
    """
    # Networking tracking dependencies states leaks connection pool corruptions prevent framework initialization client configuration parameters build call sequence metrics.
    return OpenAIEmbeddings(
        # Settings metadata model profile directory parameters checking target configurations tracking memory lookup model properties reading parameters scale assignment trace context validation.
        model=settings.openai_embedding_model,
        # Encapsulated token storage system protection structure validations type annotation options configuration parameters assignment execute line code layer annotations validations dynamic trace.
        openai_api_key=SecretStr(settings.openai_api_key),  # type: ignore[call-arg]
    )


def make_local_random() -> random.Random:
    """Per-thread isolated random.Random — no global state mutation."""
    # Thread isolated clean standalone state mathematical pseudo generator setup initialize context state parameter tracking rules map variables layout return logic trace code.
    return random.Random(RANDOM_SEED)


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_documents(source_folder: str) -> list[dict[str, Any]]:
    """
    Source folder se bulk JSON files load karo.
    hash_registry.json aur _meta.json exclude karo.
    Duplicate doc_ids deduplicate karo — pehli occurrence rakho.
    """
    # Filesystem parameters context dynamic mapping directory paths template combinations formula calculations target input data storage check code pointer path directory.
    source_dir: Path = RAW_DATA_DIR / source_folder
    # Output verified clean dynamic dictionary records tracking collections array list memory container definition allocation variables step data block tracking baseline placeholder trace mapping.
    all_docs: list[dict[str, Any]] = []
    # Real-time memory hash cache tracking sets map database indices duplicate filtering configurations lookup layer control logic setup state tracking parameters check loops validate.
    seen_ids: set[str] = set()

    # Recursive global subdirectory trees scanning mapping search loop execution crawl storage components directory parsing structural layout matching files process data.
    for json_file in sorted(source_dir.rglob("*.json")):
        # Registry logging data indexes skipping criteria rule tracking parameters checking condition branch validation context path skip step rule baseline parameters trace context log.
        if json_file.name == "hash_registry.json":
            continue
        # Descriptors metadata file layout rules structure specifications matching skip constraints processing check step memory configuration parameters trace layout code block layer text.
        if json_file.name.endswith("_meta.json"):
            continue
        try:
            # Native filesystem descriptor mapping channel access setup reader configuration properties unicode standard mapping registry stream parameters framework path logic options stream.
            with json_file.open(encoding="utf-8") as f:
                # Character stream formatting parser runtime deserialization matrix transform directly inside active object reference parameters map load processing options memory target variable parameters execution.
                data: Any = json.load(f)
            # Object schema validation rule properties tracking context verification parameters data structural integrity status loop boundary filtering check logic indicator context type configurations parsing check metrics.
            if not isinstance(data, dict) or "documents" not in data:
                continue
            # Data matrices payload list iteration mapping elements internal data structures traverse sequence workflow tracking controls execution loop parameters state data collections step parsing indicator loops tracking trace.
            for doc in data["documents"]:
                # Individual data component layout unique indicator identifier key reference string lookups read pipeline trace matching choices validation default key parameter tracking values details options.
                doc_id: str = doc.get("doc_id", "")
                # Cross check current parsed identifier code history registry history parameters check data overlap conditions filter execution tracking cache verification matching indices rule constraints checking logic.
                if doc_id and doc_id not in seen_ids:
                    # Inject valid newly discovered identifier string key right inside memory framework lookup tracking collection update instantly caching index trace registry mapping framework update data sets cache tracking set.
                    seen_ids.add(doc_id)
                    # Non overlapping validated genuine document database dictionary metadata record components append into destination tracking array properties variable targets data items metadata collection storage lists configuration rules.
                    all_docs.append(doc)
        except (json.JSONDecodeError, OSError) as e:
            # Catching corrupt formatting rules or runtime OS platform reading blocks exceptions without terminating master orchestrator execution loop thread context pipeline framework encoding error conditions safety loop bounds thread control.
            logger.warning("Could not load %s: %s", json_file, e)

    # Runtime console dashboard logs analytics performance feedback metrics ingestion summary prints data trace validation line parameters testing console visibility printout indicators tracker monitoring channel output view info.
    logger.info(
        "Loaded %d unique documents from %s", len(all_docs), source_folder
    )
    # Output completely isolated clean dynamic data collection references package lists back to main processing orchestrator loop scheduler pipeline components hierarchy models structural return configurations data layout tracker.
    return all_docs


def build_doc_content_map(
    documents: list[dict[str, Any]],
) -> dict[str, str]:
    """
    doc_id → full original content map.
    Semantic density mein chunk ko actual parent se compare karne ke liye.
    No data leak — chunk ko apne aap se compare nahi karte.
    """
    # Python dictionary comprehensions parsing syntax layout mapping equations calculations properties text variables allocation memory context loop variables lookup trace step metrics balance options mapping tracker rules expressions maps data elements block.
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
    # Vacant inputs tracking criteria safety parameters condition branch default mapping configurations fallback boundary indexes returns calculation matrix tracking step bypass routing model control parameter layouts placeholder check logic path.
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

    # Extract single list array integer configurations containing text lengths parameters measurements metrics scale calculations loop metrics data parsing items length values variables matrix sizing criteria dimension tracking profile integers array.
    sizes: list[int] = [len(c.content) for c in chunks]
    # Blank nodes text layout space filtering count accumulation loops checking values conditions formulas summaries tracking validation space checker accumulation counter loops checking structural content rules parameters balance.
    empty_count: int = sum(1 for c in chunks if not c.content.strip())
    # Distinct original baseline documentation parent reference uniqueness metrics mapping dynamic tracking cache set collection tracker verification index lookup data configurations checking variance indices variance structures collections boundary trackers.
    unique_docs: int = len({c.doc_id for c in chunks})
    # Density metric ratio check logic calculation structural formula scaling index checking rules conditions criteria variable execution trace parameters step block layout formulas calculations factors density scaling boundary splits condition validations rules tracker options.
    chunks_per_doc: float = (
        len(chunks) / unique_docs if unique_docs > 0 else 0.0
    )

    # Packaging calculated metric parameters floating precision values inside reporting dashboard structure dictionary collections tracking formatters logs metrics dashboards inventory data tracking metrics fields analysis.
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
    # Empty items sequence boundary check validation options control redirection path execution setup code criteria assignment variables tracking data layout model trace pipeline check options conditions parameters routing models context logic setup.
    if not chunks:
        return 0.0
    # Central grammatical punctuation standard components markers tracking patterns lookups choices arrays configurations parameters route lookup mapping context setup text rules expressions symbols delimiters prioritization choices checklists properties data layout indicators.
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
    # Text string end formatting validations checks loop accumulator index counting criteria values configuration mapping trace run indicator tracking pattern checks dynamic edge layout verification calculations metrics rules.
    complete: int = sum(
        1
        for c in chunks
        if c.content.strip().endswith(sentence_endings)
    )
    # Precision decimal calculation rules metrics constraints rounding formulas conversion values return flow layout matrix tracking indices configuration parameters precision float mathematical division operations metrics status validation reporting index layouts.
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
    Stratified random sampling — pehle N nahi, random N.
    Token-safe truncation — 6000 chars max.
    No data leak — chunk ko apne aap se compare nahi karte.
    """
    # Safety boundary parameter checking validation mapping options parameters lookup verify structural state tracking array model branch code fallback redirection trace block logic database configurations constraints rules framework models verification layer metadata path settings lookup step tracking.
    if not chunks or not doc_content_map:
        return 0.0

    # Scale percentage parameters numerical boundary criteria equations metrics formulations checks scaling allocation variable maps evaluation step dynamic scale constraints variables formulation calculations index rules logic matching.
    sample_size: int = min(50, max(5, int(len(chunks) * 0.05)))
    # Completely thread isolated unbiased randomized items picker execution subroutine logic slice passing parameters arrays tracking data matrix return layout block pipeline runtime scheduler metrics context sample arrays collection pull worker step execution configuration trace context variables mapping.
    sample: list[ChunkedDocument] = local_random.sample(
        chunks, min(len(chunks), sample_size)
    )

    # Valid confirmed processing elements pairs text configurations collections directory variables pointer context dataset map array tracking container data allocation register matched items records verification layer lists collection placeholders.
    valid_pairs: list[tuple[str, str]] = []
    # Scanning elements lifecycle tracking matrix traversal loop variables items records attributes data execution sequence loop block trace step pipeline runtime handler logic data stream operations components checks validation conditions array traces traversal route.
    for chunk in sample:
        # Dynamic memory key map lookup extraction safe parameters configurations metadata reading access check property variables tracking trace context data value lookup step parameters keys validation query index mapping extraction properties configurations tracking.
        parent_content: str = doc_content_map.get(chunk.doc_id, "")
        # String content availability validation processing check structural rule verification matching tracking sequence branch layout condition code trace layer context block configurations checks verification alignment filter boundary parameters paths trace layer criteria framework mapping.
        if parent_content:
            # Token constraints safety protection bounds character limit criteria parameters map layout slicing operation dynamic extraction collection variables array save trace parameters capacity length tracking options slices metrics collection insertions variable arrays data.
            valid_pairs.append(
                (
                    chunk.content[:EMBEDDING_MAX_CHARS],
                    parent_content[:EMBEDDING_MAX_CHARS],
                )
            )

    # Structural item size validation check to prevent vacant computational arrays data loop execution processing error safety block path pointer data pipeline logic trace tracker validations framework monitoring operations index items calculation limits indicators.
    if not valid_pairs:
        logger.warning(
            "No valid chunk-parent pairs — semantic density returning 0.0"
        )
        return 0.0

    # Extract target text sequences components out of pairs maps direct inside separate textual calculations lists vectors variables tracking arrays model processing collections separate mapping inputs fields list variables storage text inputs arrays vector structures data models variables.
    chunk_texts: list[str] = [p[0] for p in valid_pairs]
    # Isolate original historical raw baseline documents text lines layout configurations data input maps listing arrays parameters check data collection tracking options setup master reference baseline textual components mappings configurations lists tracking models target metrics properties data context data sets text traces validation reporting arrays.
    parent_texts: list[str] = [p[1] for p in valid_pairs]
    # Dynamically inject tenacity retry resilience layer wrapper right on top of the thread-isolated embedding engine instance function pointer variables setup tracking mapping core logic closure function connection pool safety wrappers parameter setup.
    embed_fn = _make_embed_with_retry(embedder)

    try:
        # Request outbound remote provider network embedding vector array matrix calculations using dynamic retry wrapped routine execution result arrays return trace list data tracking parameters connectivity options tokens metrics evaluation models query stream updates trace tracking array vector collections functions tracking.
        chunk_embeddings: list[list[float]] = embed_fn(chunk_texts)
        # Parent original documentation sequences text vectors representation computation provider connection call address pipeline routing data maps values execution results trace remote servers computation layers data alignment vector models mappings metrics trace indicators models.
        parent_embeddings: list[list[float]] = embed_fn(parent_texts)

        # Distance similarity arrays registry storage context target metrics track tracking array allocation initialization code variables calculations loop parameters setup metrics model analytics mapping results trace arrays configurations database registry repository metrics calculations array components collection models parameters values.
        similarities: list[float] = []
        # Multi-variable loop tracking traversing parallel vector arrays coordinates data unpack trace execution validations sequence check control iterations blocks tracking look trace matrix multi component pairs mapping dimensions scanning iterations validation parameters coordinates tracking data lookup values checks metrics tracker.
        for c_emb, p_emb in zip(chunk_embeddings, parent_embeddings):
            # Convert basic array list floats into high speed mathematical matrix processing numpy structures layout tracking objects parameters values design rules calculation trace model high performance multidimensional optimization float operations metrics calculations tracking linear structure array models formatting arrays data maps format object.
            c_arr = np.array(c_emb)
            # Alignment matching checking numerical checks validation properties configuration map elements inside high efficiency numpy object calculations parameters check runtime log trace data check metrics operations values validation properties matrix dimension checks layout tracking configurations analysis processing matrices.
            p_arr = np.array(p_emb)
            # Vector norm dimensions square matrix multiplier calculations structural magnitude coefficients index resolution safety check variable trace formula execution data layer context rules math metrics calculation formulas magnitude multiplier scaling protection parameters checks variance boundaries division runtime exception safeguard values.
            norm_product: float = float(
                np.linalg.norm(c_arr) * np.linalg.norm(p_arr)
            )
            # Zero division protection boundaries exceptions branching path skip execution context loop controls validation tracker layer tracking code logic framework operations trace variables criteria validation constraints safety tracking checks loop bypass options redirector indicators layers structure pointer.
            if norm_product < 1e-10:
                continue
            # Distance matrix resolution linear dot multiplier products logic formula resolution score parameter verification dynamic mapping data variable list append track indicator data flow checks process vector operations linear multipliers results aggregation float scores mapping index evaluation trace updates.
            similarities.append(
                float(np.dot(c_arr, p_arr) / norm_product)
            )

        # Yield completely resolved mathematical mean average metrics coefficient properties configurations layout metrics analysis output check trace return flow pointer data logic statistical mean computations coefficient results overview validation parameters floating summaries analytics reporting numbers trace structures complete.
        return (
            round(float(np.mean(similarities)), 4)
            if similarities
            else 0.0
        )

    except Exception as e:
        # Capture API connection crashes, authentication timeouts, or unhandled token data parsing issues metrics warnings tracking traces indicators screen output logs warning updates tracking connectivity failures processing exceptions context logs tracking trace indicators stream details parsing data.
        logger.warning("Semantic density failed after retries: %s", e)
        # Secure failure mode path redirection standard fallback default values tracking parameters setup execution branch check metrics layout flow trace runtime code recovery options pathway index structural validation safe fallback tracking process indicators monitoring configuration mapping routing flow trace data logic framework parameters.
        return 0.0


# ── Artifact Generators ───────────────────────────────────────────────────────


def save_chunk_size_histogram(
    chunks: list[ChunkedDocument],
    strategy: str,
    source: str,
    tmp_dir: str,
) -> str:
    """
    Thread-safe histogram via matplotlib.figure.Figure().
    plt global state thread-safe nahi — Figure object isolated hai.
    """
    # Extract structural length dimension properties loop comprehension tracking capacity character variables lists calculations metrics elements array length measuring metric calculations variables trace inline list comprehension operations character capacity evaluations tracker integer arrays sizes metrics value mapping configurations layout.
    sizes: list[int] = [len(c.content) for c in chunks]
    # Running statistical analysis calculations formula computing exact distribution pattern midpoint evaluation values track trace parameters property configuration setup model parameters arithmetic average calculations formulation variables tracking parameters indicators summary.
    mean_size: float = float(statistics.mean(sizes))

    # Thread safe fully isolated decoupled canvas tracking architecture plot generation instance class invocation context setup mapping variables parameters trace log code step execution line layout standalone chart container canvas initialize allocation parameters geometry configuration.
    fig = matplotlib.figure.Figure(figsize=(10, 5))
    # Coordinate grid frame manager allocation section mapping layout active index object generation add execution layout variables parameters check code runtime layer processing frame management axis placement subplot structural canvas indexing coordinate grid setup mapping variables rules monitor data track.
    ax = fig.add_subplot(1, 1, 1)
    # Chart plotting data graphics mapping parameters color specifications borders styling layout alpha frequency density tracking data blocks canvas trace draw context parameters design settings frequency metrics chart drawings rendering bars metrics alignment properties layout visualization palette color.
    ax.hist(sizes, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    # Typography template variable string formatting visualization title overlays main definitions description styling values parameters log terminal console updates trace output text display models template string compound charting dynamic name parameters header formatting style layout details trace.
    ax.set_title(
        f"Chunk Size Distribution — {strategy.upper()} | {source}",
        fontsize=14,
        fontweight="bold",
    )
    # Horizontal canvas layout parameters naming labels text specifications design metrics styling criteria parameter code target maps tracking level data line alignments rules configurations parameters horizontal axis labeling character dimensions tracker layout definitions.
    ax.set_xlabel("Chunk Size (chars)", fontsize=12)
    # Vertical coordinate sequence tracking metric level parameter label details formatting string text dashboard configuration settings write analytics values trace runtime logic metrics level parameters check vertical scale metrics frequency values mapping configurations visibility screen parameters text.
    ax.set_ylabel("Frequency", fontsize=12)
    # Overlay reference line markers plotting statistical center calculations dash format setup indicators variables balance display path alignment logic code trace statement parameters mapping lines design boundaries performance median indicators overlays dashboard guidelines trace display markers graphics settings drawing line parameters options.
    ax.axvline(
        mean_size,
        color="#DD4444",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_size:.0f} chars",
    )
    # Legend description layout panel canvas visibility activation parameters trigger maps text details styling parameters tracking method call trace code logic updates visibility panel interface dashboard active indicators labels panel box placement console monitoring metrics styling views visibility parameter checks tracking.
    ax.legend(fontsize=11)
    # Spacing margins calculation auto optimization bounding compress chart layouts tidy graphics canvas drawing save output processing layer code run step rendering alignments formatting canvas space auto arrangement boundary padding limits adjustment execution graphics draw engine clean render options.
    fig.tight_layout()

    # Dynamic target file system repository destination path layout string naming templates mappings parameters properties location variable string data look files directory locations naming conventions destination local disk filename structural index mappings variable template tracking lookup indicators strings address paths.
    plot_path: str = f"{tmp_dir}/{strategy}_{source}_chunk_dist.png"
    # Write chart binary sequence streams directly onto active system local disk persistence track image format rendering file saving action execution indicator code block log data stream write maps filesystem persistent file output allocation binary conversion export trace options line.
    fig.savefig(plot_path, dpi=150)
    # Return destination mapping folder storage address pointer variable direct back to metadata uploading transaction scheduler system modules interface logic path analytics tracking logs dynamic file path location validation string upload task tracking framework component reference indicator.
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
    # Random metrics selection indices calculations pool tracking array selection window slice elements logic lists variables extraction tracking execution routine code step processing parameters allocation window randomized data sub selections sampling executor array slice limits tracking rules operations framework parameters options.
    sample: list[ChunkedDocument] = local_random.sample(
        chunks, min(len(chunks), n)
    )
    # Data mapping format conversion subroutine keys values configuration mapping structures transformation direct inside serialization key value format dictionaries lists storage variables check properties fields layout model serializable standard reporting dictionary metadata alignments formatting.
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
    # Destination location coordinate string template formatting configuration fields address variables mapping setup indicators tracking file variables check layout address layout destination setup naming coordinate folder path templates indexing character maps variables validation locations reference.
    json_path: str = (
        f"{tmp_dir}/{strategy}_{source}_sample_chunks.json"
    )
    # File descriptor handler channel allocation, enabling local resource path writing configurations conversions processing parameters integrity checks code path modes enabled unicode safe character conversions text file creation stream tracking layout settings options validations.
    with open(json_path, "w", encoding="utf-8") as f:
        # Packing serializations writing data maps records json data dump operations layouts style indentation properties variable configurations flow trace code run level packing data serialization metrics formatters serialization parameters dump streaming actions text variables processing write execution track variables.
        json.dump(output, f, indent=2, ensure_ascii=False)
    # Return saved destination address pathway character string direct back to tracking uploading manager integration modules context interface framework trace output files properties variable address clean reference character string data validation framework indicators pointer reference pathway path mappings context tracker.
    return json_path


# ── Single Experiment Runner ──────────────────────────────────────────────────


def run_single_experiment(
    strategy_name: str,
    docs: list[dict[str, Any]],
    doc_content_map: dict[str, str],
    source_name: str,
    experiment_id: str,
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
    # Composite dashboard visualization indexing label template string definitions parameter combinations mapping configuration indicator target code values trace value identity formulation properties multi component variable name compound metrics tag identifiers format settings lookup value mapping criteria context.
    run_name: str = f"{strategy_name}__{source_name}"
    # Performance telemetry track notification logging info print process run monitoring console update terminal trace metrics display layout block line text tracking variables verification options startup notifications trigger logs info console output print trace analytics message status updates.
    logger.info("Starting: %s | docs: %d", run_name, len(docs))

    # Initialize low-level backend monitoring management interface client object completely unbinding standard framework contextual tracking thread state parameters explicit transaction tracker system class instantiation isolate backend tracking operations from high level dependencies.
    client = MlflowClient()
    # Invoke backend monitoring systems directory to allocate unique isolated database recording track element context model maps setup value mapping trace context transaction request server dashboard session configurations metadata metrics keys register mapping target setups indicators.
    run = client.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={
            "mlflow.runName": run_name,
            "run_type": "chunking_experiment",
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
    # Technical transaction identification sequence reference code string tracking identifier parameter location updates reading parameter value data mapping variable database unique transaction parameters variable parsing explicit storage token identifier parameters access keys.
    run_id: str = run.info.run_id

    try:
        # Fresh instances — no shared state across threads
        # Strategy localized object instantiation subroutine factory design execution passing targets schema context variable values indicator check layout mapping track properties model allocation dynamic components factory pattern execution loop checks values mappings handlers initialization.
        chunker = make_chunker(strategy_name)
        # Vector spatial metrics translation engine builder instantiation function invoke configurations criteria mapping setup call level check tracking code pointer line core modeling parameter mappings vector engines initialization request handlers parameters pass dynamic logic mappings.
        embedder = make_embedder()
        # Randomized parameters selection entropy device tracker instantiation function execution isolation variables assignment sequence validation setup code tracing metrics evaluation random state per thread separate entropy isolated instance creation setup rules mapping value parameters context.
        local_random = make_local_random()

        # Log params
        # Properties configuration collection parameters storage inventory listings dictionary schema definition mapping parameters setup data variable tracing tracker values checklist parameters dynamic inventory baseline record metrics parameters key indexing maps tracking backend register storage dashboard records parameters initialization maps.
        params: dict[str, Any] = {
            "strategy": strategy_name,
            "source": source_name,
            "doc_count": len(docs),
        }
        # Dynamic property lookup check checking variable validity condition branch routing validation parameters path selection validation track properties look variables code layer mapping branch attribute search execution condition context variable validations logic.
        if hasattr(chunker, "chunk_size"):
            params["chunk_size"] = chunker.chunk_size
        # Dynamic checking property attribute lookup validation check parameters values tracking condition execution code block data variable trace look step block mapping structural setups evaluation context tracking property configurations boundaries validation context limits.
        if hasattr(chunker, "chunk_overlap"):
            params["chunk_overlap"] = chunker.chunk_overlap
        # Structural boundary indices constraints constraints limit parameter configurations structural check variables code tracking setup parameter value mapping asset pointer variance constraints variables max length capacities tracking specifications conditions mapping.
        if hasattr(chunker, "max_chunk_size"):
            params["max_chunk_size"] = chunker.max_chunk_size
        # Post processing layout capping properties metrics character threshold parameters existence validation checks variable assignments checking control constraints conditional boundary limit settings data pointers property.
        if hasattr(chunker, "hard_cap_chars"):
            params["hard_cap_chars"] = chunker.hard_cap_chars
        # Coarse semantic separation logic thresholds parameters existence evaluation configurations mapping check options validation indicators tracker properties variable lookup condition checks settings formulas trace mapping rules.
        if hasattr(chunker, "coarse_threshold"):
            params["coarse_threshold"] = chunker.coarse_threshold
        # Secondary fine processing mechanical micro splitting baseline constraint thresholds numbers existence checks data arrays formatting structures collection parameter tracking index mapping criteria context code validations limits layer.
        if hasattr(chunker, "fine_chunk_size"):
            params["fine_chunk_size"] = chunker.fine_chunk_size
        # Multi dynamic cutoff algorithm parameter validation check properties check configurations models variable values data parameter mapping criteria tracking trace boundary settings calculations rules variance indices lookups parameters mappings context validations.
        if hasattr(chunker, "breakpoint_threshold_type"):
            params["breakpoint_threshold_type"] = (
                chunker.breakpoint_threshold_type
            )
        # Mathematical scaling computations options indicators validity checks variable parameters code tracking updates layout storage mapping properties verification trace intensity variables calculations statistical metrics limits parameter variance check thresholds formulas.
        if hasattr(chunker, "breakpoint_threshold_amount"):
            params["breakpoint_threshold_amount"] = (
                chunker.breakpoint_threshold_amount
            )

        # Transmit single structural configuration items sequence straight into distant database logs server tracking framework telemetry upload loop api code execution layer metadata registration parameters logging actions loop execution tracking parameters maps dashboard update indices parameters.
        for key, value in params.items():
            client.log_param(run_id, key, value)

        # Chunk + measure latency
        # High resolution hardware platform chronometer checkpoint snapshot register timing sequence benchmark evaluation track runtime interval delta start track pointer line clock execution baseline precise timing device tracking snapshots check metrics.
        start_time: float = time.perf_counter()
        # Polymorphic functional procedure orchestration layout text content arrays computational loops text segmentation data arrays mapping return computation trace variable text parsing strategy model structural framework procedure mapping components array return data.
        chunks: list[ChunkedDocument] = chunker.chunk(
            docs, deduplicate=True
        )
        # Runtime calculation results time window formatting parameters float decimal limits parameters difference tracking indicator score value save data trace loop level metrics hardware operational latency evaluation time calculation differences metrics mapping float numbers.
        latency: float = round(time.perf_counter() - start_time, 3)

        # Compute metrics
        # Statistical data layout check metrics parsing calculations function invocation datasets input array properties maps return configuration code validation layer structure parameters calculations statistical text property analysis values calculation matrices maps models formatting metrics context loop.
        structural: dict[str, float] = compute_structural_metrics(chunks)
        # Sentence segmentation formatting compliance index indicator calculation routine call target matrix variables register tracking validation console indicator trace step accuracy measurement scores sentence layout termination check indicators metrics reporting.
        boundary_score: float = compute_boundary_respect_score(chunks)
        # Semantic semantic space distance matrix mapping values calculation method call passing variables calculations validation vector dimension levels indicators tracking step context fidelity checks vector mathematical calculations validation tracking matrix parameters.
        semantic_density: float = compute_semantic_density(
            chunks, doc_content_map, embedder, local_random
        )

        # Log metrics
        # Packaging all computed tracking values calculations into unified dictionary structures payload data stream for direct monitoring database pipeline storage cloud tracking server payloads aggregated metrics floats dictionary records pipeline stream operations.
        all_metrics: dict[str, float] = {
            **structural,
            "boundary_respect_score": boundary_score,
            "avg_semantic_density": semantic_density,
            "latency_seconds": latency,
        }
        # Iterate over formatting metrics mapping elements data charts to send variables numbers straight inside dynamic infrastructure metrics logger tracker interface line server dashboard charts dynamic data telemetry tracking analytics server dashboard log updates.
        for key, value in all_metrics.items():
            client.log_metric(run_id, key, value)

        # Log artifacts
        # Environment system dynamic transient space virtual directory storage manager wrapper block configuration automatic cleanup rules tracking pipeline path trace context execution file system resource operations virtual operating system dynamic files placeholders tracking.
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create chart distribution visualizations local files location strings parsing variables tracking canvas run data trace line execution model graphics chart visualization quantitative metrics histogram png files generator address.
            hist_path = save_chunk_size_histogram(
                chunks, strategy_name, source_name, tmp_dir
            )
            # Create text database items serialized storage structures validation mapping check destination path records files write update indicators check trace data structure previews information check matching serialize text configurations files path layouts storage data string format.
            json_path = save_sample_chunks_json(
                chunks,
                strategy_name,
                source_name,
                tmp_dir,
                local_random,
            )
            # Send analytics plotting image artifacts directly inside master monitoring persistent folder storage pathway upload execute tracking api run trace line code server graphics pipeline uploads cloud assets registry persistent file shipping.
            client.log_artifact(
                run_id, hist_path, artifact_path="plots"
            )
            # Quality assessment validation verification items lists format specifications details parameters tracking update structural telemetry upload sequence route api data logic text preview formats maps dynamic serialization files upload transactions tracking framework parameters address reference path mappings context hooks.
            client.log_artifact(
                run_id, json_path, artifact_path="sample_chunks"
            )

        # Update status configurations parameters flags metadata data sets indicators inside distant database transaction record state finish signals execute method call runtime lifecycle completion flags central backend session state final finish termination signal execution trace.
        client.set_terminated(run_id, status="FINISHED")
        # Technical engineering trace logging status complete confirmation messaging terminal write updates performance details indicators metrics console print summary block loop execution parameters reporting instrumentation validation notification overview reports metrics lines write summaries console tracking.
        logger.info(
            "Done: %s | chunks: %d | latency: %.2fs | density: %.4f | boundary: %.4f",
            run_name,
            len(chunks),
            latency,
            semantic_density,
            boundary_score,
        )

        # Output overall summary evaluation variables records fields maps back into master monitoring scheduler execution system framework details return flow layer consolidated tracking parameters comprehensive output metadata database maps return structural variables array components.
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
        # Unexpected script faults interception safeguards flags parameter updates tracking crash terminate execution signature framework pipeline execution trace call code block runtime exceptions protection unexpected failures parameters updates tracking execution state exceptions catcher model flags.
        client.set_terminated(run_id, status="FAILED")
        # System failures evaluation reporting error tracking message strings write console diagnostics metrics overview streams monitor parameters log text trace runtime fault analytics database failures logging error message terminal reporting console visibility updates trace logging models details.
        logger.error("Experiment failed: %s — %s", run_name, e)
        # Propagating tracking exception data models back towards master pool queues hierarchy lifecycle control execution thread bubbles indicator code stream line exception bubble up interrupt signal routing multi thread stack execution context bubble line.
        raise


# ── Main ──────────────────────────────────────────────────────────────────────


def run_all_experiments(skip_api: bool = False) -> None:
    """
    45 MLflow runs — 9 strategies × 5 sources.

    Execution model per source:
    - Non-API strategies (6): ThreadPoolExecutor (MAX_WORKERS=4) — parallel
    - API-bound strategies (3): Sequential — rate limit safe
    """
    # Sync core monitoring environments master tracing indicators using dynamic project configurations path tracking storage registration project initial context validation tracker name backend store links server connectivity routes configuration setup parameters model lookup data paths mapping.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # Central baseline index workspace folder instantiation project tracking registration database fields properties metadata values mapping indicator data lookup check line setup master configuration repository baseline central tracking registry parameters workspace initialization project maps data.
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    # Internal dashboard system entry identifier tracking reference values variable reading destination context tracking records variables check parameter lookup code tracking step unique experiment identity keys system context storage tracking reference variable assignments allocation models mapping index trace database lookup.
    experiment_id: str = experiment.experiment_id

    # Global compilation summary results lists tracking data map properties metrics arrays final storage variable pointer trace calculation records list memory allocation master results collection register overall metrics monitoring records tracking listing array targets initialization matrix database tables configurations repository setup trace memory address.
    all_results: list[dict[str, Any]] = []

    # Master sequence iteration traversal loop scanning content repositories groups matching directory paths listings folder options elements data trace tracking variable mapping matrix dimensions loop master directory location collections configurations data loading process cycle matrix traversal paths layout.
    for source_folder, source_display in SOURCES.items():
        # Ingest overall dataset content text strings configurations dictionary array records using localized load reading routine outputs variable array check memory trace parameters data extraction loader utility unsegmented raw data mapping listings loading components dynamic background array readings data lists mappings modules parser.
        docs: list[dict[str, Any]] = load_documents(source_folder)
        # Empty variables checklist security guardrails parameters checking condition branch validation context loop bypass index forwarding layout dynamic tracking rules trace step rule check validation filter boundaries safety constraints checks condition branches control loop forward pointer redirects options matrix configurations.
        if not docs:
            # System notification warning level indicators console tracking triggers when vacant inputs tracking datasets encountered context checks options variable mapping limits tracking missing sources message structural missing data streams warning indications parameters lookup dashboard values mapping rules.
            logger.warning("No documents found: %s", source_folder)
            # Forward processing master scheduling tracking index loops directly into adjacent source groups entries tracking bypass branch iteration trace execution block line level loops redirector pointer sequential forwarding execution pointer track context loops mapping structures redirect indicators check layer.
            continue
        # Context dynamic injection leakage protection mapping data validation processing subroutine parameter pass collections layout lists results tracking details return data flow loop insulation data setup logic memory context leak protection isolated content parsing dataset builders dynamic variable collections returns mapping code flow layout data structures mapping context mappings setup trace data values execution.
        doc_content_map: dict[str, str] = build_doc_content_map(docs)

        # Performance notification mapping console traces monitoring instrumentation statistics updates overview parameters write tracking operations logs indicators pipeline validation tracking message indicators console output terminal summary indicator track progress display metrics logging message updates printout setup data details context variables routing.
        logger.info(
            "Processing source: %s | docs: %d | strategies: %d",
            source_display,
            len(docs),
            9,
        )

        # ── Parallel: Non-API strategies ─────────────────────────────────────

        # Concurrency asynchronous thread manager pools setups configuration, limits indicator allocation tracking workflow context block initiation trigger multi processing execute layout thread clusters allocation loops thread resource allocation context orchestrator setups thread configurations tracking parameters execution bounds max worker execution.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map dynamic items loop comprehension submitting tasks straight inside worker thread streams returning execution identifiers matrix tracking lists variables array collections setup code parallel thread scheduler mapping task mapper thread async submission process worker schedules allocations configurations registry dynamic map definitions structures futures list variables placeholder cache tracking.
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
            # Active tracking pooling event checkpoints traversal checking variable data fetch iterations validation status updates trace tracking variables configuration loop look check layer completion checking routines event asynchronous monitoring queues loop check polling status configurations task updates look layer checks data mappings tracking.
            for future in as_completed(futures):
                # Fetch localized parsing strategy tracking label template string matching registry references variables mappings layout indices properties look text context check future tasks completions tracker retrieval identification parameter label dictionary references map data structure key options index values properties trace context variables check.
                strategy_name = futures[future]
                try:
                    # Ingest completely processed tracking table dataset record items out of thread data parameters channel save array collections list variables storage memory output metrics results collection thread items fetch calculation outputs results maps storage database lists target compilation results register array array lists data models update registry.
                    all_results.append(future.result())
                except Exception as e:
                    # Capture thread execution failures tracking logic errors options structural configurations trace parameters data exceptions print console dashboard reporting metrics code concurrency failures report thread failures diagnostics interception logs error tracking dynamic exceptions context traces data mapping loop context exceptions parameters error.
                    logger.error(
                        "Failed: %s × %s — %s",
                        strategy_name,
                        source_display,
                        e,
                    )

        # ── Sequential: API-bound strategies ─────────────────────────────────

        # Cost tracking budget boundary parameters checklist condition validation variable switches control execution logic pattern configurations mapping context rules setup parameter track check financial parameters safety flag rate limit thresholds check modifiers options logic check conditional variables configuration tracking check constraints bypass loops.
        if not skip_api:
            # Sequential sorting iterations strategy items tracking arrays listings mapping loops traversal metrics rules ordering parameters arrangements alphanumeric index list tracking look context loops traversing configurations setup execution flow steps mapping execution paths logic loop matrix tracing options.
            for strategy_name in sorted(API_BOUND_STRATEGIES):
                try:
                    # Running single standalone execution tracking routine logic call parameters components passing tracking variable schemas database models reports layout indicator execution pointer sequence execution task core runner invocation call direct metrics parameters mapping setup records list updates data arrays configuration trace indicators maps dynamic workflows models execution parameters.
                    result = run_single_experiment(
                        strategy_name=strategy_name,
                        docs=docs,
                        doc_content_map=doc_content_map,
                        source_name=source_display,
                        experiment_id=experiment_id,
                    )
                    # Accumulate computed linear execution summaries matrices records right inside global table dataset array listings collection master monitoring summary register matrix tables items configurations dynamic results list array update tracking metrics fields compiled structures datasets matrix lists storage allocations parameters.
                    all_results.append(result)
                except Exception as e:
                    # Intercept parsing routine errors or API exceptions console logs warning levels visibility error track trace metadata tracking indicator report layout line print variables execution pipeline exceptions data conversion exceptions network failures handling error logs display traces variables checks monitoring trace.
                    logger.error(
                        "API strategy failed: %s × %s — %s",
                        strategy_name,
                        source_display,
                        e,
                    )

    # ── Summary Table ─────────────────────────────────────────────────────────
    # Ordering metadata entries tracking layout parameters using sorting indexing rules parameters lambda properties sequences coordinate layout process loop variable check data trace sorting parameters configurations index structural sorting logic alignment sorting tracking metrics arrangement calculations formulas values listing data maps index loops configurations layout sorting arrays options.
    all_results.sort(key=lambda r: (r["source"], r["strategy"]))

    # Visual alignment console tracking design borders drawing standard patterns line text log execution parameters visualization track metrics block layout graphics print code sequence separation line layout graphics dashboard configurations framework separator bar visual console layouts markings graphics mapping details.
    logger.info("\n%s", "=" * 115)
    # Master summary table console banner updates text indicators console output logger verification prints report framework statement logic setup variables trace master dashboard calculation metrics summaries printing terminal visibility update overview indicators text values tracking database updates console visibility layout trace models description tables metadata map view display settings parameters.
    logger.info(
        "CHUNKING EXPERIMENT SUMMARY — 9 Strategies × 5 Sources"
    )
    # Border pattern layout separator line drawing standard parameters console logs reporting analytics dashboard configuration graphic structure context variables updates trace mapping line alignment separators divider horizontal dashboard framing visual layout marker separations graphic borders lines drawings mapping traces console.
    logger.info("=" * 115)
    # Typography padding metrics adjustments dynamic spacing calculations character length alignment formatting fields variables configuration layout setup data definitions write text parameters padding spacing format markers column metrics calculations formatting typography configurations labels definitions write terminal output console channels log trace stream.
    logger.info(
        f"{'Strategy':<20} {'Source':<12} {'Docs':>5} {'Chunks':>7} "
        f"{'Avg Size':>9} {'Std':>7} {'Density':>9} {'Boundary':>9} {'Latency':>9}"
    )
    # Spacing dividers rendering standard lines console logging presents reports visibility optimization visual formatting clean terminal view code parameters line step layout level tracking reporting console separator bars segment layout separation line chart grids console presentations layout screen guidelines alignment separator line drawing indicators canvas format.
    logger.info("-" * 115)
    # Summary results dataset maps traversal element processing configurations loop scanning steps arrays verification variables context output trace logic parameter details check code trace dynamic variable iterations mapping text processing records iterations layout dashboard numbers summary evaluation parameters checking values data configurations listing loops console print tracking view details matrices.
    for r in all_results:
        # Dynamic property parameter numbers variables float conversions configuration padding layout formatting column calculations math console channel direct string updates execution line text visual string alignments formatters layout string format width configurations decimal operations metric prints console outputs numbers conversions metadata logging trace elements variables text layout parameters details.
        logger.info(
            f"{r['strategy']:<20} {r['source']:<12} {r['doc_count']:>5} "
            f"{int(r['chunk_count']):>7} {r['avg_chunk_size_chars']:>9.0f} "
            f"{r['std_chunk_size_chars']:>7.0f} {r['avg_semantic_density']:>9.4f} "
            f"{r['boundary_respect_score']:>9.4f} {r['latency_seconds']:>9.2f}s"
        )
    # Terminal display structure frame closing layout canvas complete validation marker metrics logs trace compilation tracking validation setup templates code mapping framework indicator line visual footer border markings canvas terminal framing completion markings dynamic logs trace parameters framework complete validation border indicators.
    logger.info("=" * 115)
    # User guidelines tracking paths URLs instructions details terminal updates console trace configuration properties print mapping documentation logs visibility level path line data check manual guidelines directions display accessibility dashboard instructions locations terminal printout navigation guidance reference layouts links data path log text tracking setup framework options.
    logger.info(
        "MLflow UI: mlflow ui --backend-store-uri mlruns → http://127.0.0.1:5000"
    )
    # Enterprise project traces index reference analytics pipelines variables dashboard links path logger terminal text output printing layout monitoring trace pipeline parameters config browser trace logging directions options cloud telemetry indices reference dashboard configuration links terminal logs view project trace logging pathway configuration blueprints options mapping track parameters framework analytics browser options layout trace logs.
    logger.info(
        "LangSmith : https://smith.langchain.com → project: %s",
        settings.langchain_project,
    )


# ── Entry Point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Command interface system terminal inputs parsing constructor handler instantiation configurations register variables tracking control execution initialization setup metrics field data line execution variables dictionary lookups structural arguments parsing terminal handling blueprints validation setups setup dynamic controller rules mapping properties layout option.
    parser = argparse.ArgumentParser(
        description="SkyLex Chunking Experiment Runner — 9 strategies × 5 sources"
    )
    # CLI command arguments option modifier flag definitions allocation dynamic binary switches data logic true values parameters configuration variables properties tracing step level assignment modifier execution toggles custom modifiers option parameters switches variables checking boolean parameters allocation layout modifiers definitions choices flags rules map data setup.
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="API-bound strategies skip karo (semantic, improved_semantic, double_pass)",
    )
    # Input terminal data configurations lookup validation process trigger execution variables parameter collection structure read values parameters lookup setup metrics fields data line execution variables dictionary lookups parser arguments checking evaluation context variables query mappings fetch record storage indicators parameters mapping setups layout options fields validations logic metrics check.
    args = parser.parse_args()
    # Execute master orchestrator loop routing configurations pass properties flags conditions experiments matrix matrix analysis processing pipeline loops execution indicator complete trace run core loop trigger execution pathways core master routine orchestration call parameters passing modifier flags dynamic conditional processing run loops matrix matrix checks analytics triggers.
    run_all_experiments(skip_api=args.skip_api)