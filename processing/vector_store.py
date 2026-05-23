"""
processing/vector_store.py

SkyLex Vector Store — ChromaDB persistent collection manager.

Collection naming convention: skylex_{source}_{strategy}
Example: skylex_faa_cfr_recursive, skylex_faa_ad_hierarchical

Kyun source × strategy alag collections:
  - Har strategy ke chunks alag store hote hain
  - RAGAS evaluation mein same golden queries alag collections pe chalti hain
  - Results compare karke final strategy decision hoti hai
  - Ek strategy ka data doosri ko pollute nahi karta

Phase 2 se candidate strategies per source:
  FAA_CFR  → semantic, recursive, double_pass
  FAA_AD   → hierarchical, improved_hybrid, hybrid
  FAA_AC   → double_pass, recursive, improved_semantic
  DGCA_CAR → double_pass, recursive, improved_semantic
  SKYBRARY → recursive, improved_semantic, improved_recursive

Total collections: 5 sources × 3 candidates = 15 collections

Persistence: data/vector_store/ folder mein disk pe save hota hai.
Program restart ke baad bhi data rehta hai.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from monitoring.logger import get_logger

# Database synchronization transitions, multi-collection upserts, aur automated structural recovery states ko trace aur debug karne k liye main telemetry engine instantiate kiya.
logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Local persistent system level cluster directory target path jahan local collections data storage structures mapped localized execute hote hain.
_PERSIST_DIR: str = "data/vector_store"

# System indexing standard key indicator jisse standard system namespaces collisions bypass kiye ja sakein.
_COLLECTION_PREFIX: str = "skylex"

# Phase 2 performance optimization metrics matrix outcomes k filters dynamic validation allocation dictionaries structures parameters mappings directory setup tracking variables.
CANDIDATE_STRATEGIES: dict[str, list[str]] = {
    "FAA_CFR": ["semantic", "recursive", "double_pass"],
    "FAA_AD": ["hierarchical", "improved_hybrid", "hybrid"],
    "FAA_AC": ["double_pass", "recursive", "improved_semantic"],
    "DGCA_CAR": ["double_pass", "recursive", "improved_semantic"],
    "SKYBRARY": ["recursive", "improved_semantic", "improved_recursive"],
}

# Input variables parameters check range controls checklists collections mappings constraints layout validation indicators frozen set data.
SUPPORTED_SOURCES: frozenset[str] = frozenset(CANDIDATE_STRATEGIES.keys())

# HNSW vector graphing index parameters adjustments settings layout: text similarity metric standard space algorithms calculations variables tracking definitions limits checklist metrics options properties index maps.
_HNSW_METADATA: dict[str, Any] = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 100,
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_collection_name(source: str, strategy: str) -> str:
    """
    Source + strategy se ChromaDB collection name banao.

    Format: skylex_{source_lower}_{strategy}
    Example: skylex_faa_cfr_recursive

    ChromaDB collection name constraints:
      - Lowercase only
      - Alphanumeric + underscore + hyphen
      - 3-512 characters
    """
    # Text data standardization conversion routine formatting characters space mutations pipeline safe regex execution value allocation map trace path setup data.
    source_clean = source.lower().replace(" ", "_")
    # Composite string compilation lookup formatting dynamic concatenation execution templates definitions keyword register target mapping pointer variable data lookup check.
    return f"{_COLLECTION_PREFIX}_{source_clean}_{strategy}"


def get_all_collection_names() -> dict[tuple[str, str], str]:
    """
    Saare source x strategy combinations ke collection names return karo.

    Returns:
        Dict mapping (source, strategy) -> collection_name
    """
    # Composite keys mapping matrix comprehension configurations inline calculations loops traversal across cross boundaries collections parameters lists updates memory pointer tracking trace codes.
    return {
        (source, strategy): get_collection_name(source, strategy)
        for source, strategies in CANDIDATE_STRATEGIES.items()
        for strategy in strategies
    }


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class QueryResult:
    """
    Vector store query ka structured result.

    Attributes:
        chunk_ids   : Retrieved chunk IDs (deterministic MD5)
        documents   : Actual chunk text content
        metadatas   : Per-chunk metadata dicts
        distances   : Cosine distances (0 = identical, 2 = opposite)
        source      : Jis source collection se retrieve kiya
        strategy    : Jis strategy ki collection se retrieve kiya
        query_time  : Retrieval latency in seconds
    """

    # Structured unique sequence identity variables list array metadata tracking indexing tokens result elements store placeholders.
    chunk_ids: list[str]
    # Linear original source strings parsed parts documents collection database references lists text segments content variables target map.
    documents: list[str]
    # Ingested configurations meta models parameters records dictionaries lists structures storage metadata lookup indicators check trace properties fields allocation.
    metadatas: list[dict[str, Any]]
    # Coordinate float calculations metrics space resolution vector dot multiplier coefficients arrays list mapping validation checks metrics values tracking variance context parameters.
    distances: list[float]
    # Source metadata categories indicators text string layout lookups matching parameters display configurations tracking data fields maps framework line.
    source: str
    # Algorithmic strategy identification string tag definitions tracking indices configuration options structural pipeline check variables pointer layout trace data properties.
    strategy: str
    # Performance benchmark intervals timestamp clock calculation delta variables duration calculations metrics numbers decimal limits evaluation check runtime logs.
    query_time: float


@dataclass
class CollectionStats:
    """
    Ek collection ki current state.

    Attributes:
        name        : ChromaDB collection name
        source      : SkyLex source identifier
        strategy    : Chunking strategy name
        chunk_count : Total chunks stored
        persist_dir : Disk location
    """

    # Underlying database storage folder identity matching label tag definition strings properties map allocation context data line tracking.
    name: str
    # Category dynamic indicator string formatting specifications lookup parameters check validations targets values storage pointer mapping read trace data block logic.
    source: str
    # Stratified parsing optimization metrics indicators tags matching registry configurations models values parameters checking indicators level data count tracking loops variable.
    strategy: str
    # Total volume aggregate capacity numeric segments text lists counters variables items data processing update trace calculations metrics dashboards storage layer value parameters.
    chunk_count: int
    # Local filesystem persistent storage path check location verification variables layout properties allocation direct file track path pointer context directory setups models properties execution.
    persist_dir: str


# ── Vector Store ──────────────────────────────────────────────────────────────


class SkyLexVectorStore:
    """
    ChromaDB-backed persistent vector store for SkyLex.

    15 collections manage karta hai — 5 sources x 3 candidate strategies.
    Embeddings bahar se pass kiye jaate hain (OpenAI ya BGE-M3) —
    ChromaDB sirf vectors store aur retrieve karta hai.

    Usage:
        store = SkyLexVectorStore()

        # Chunks add karo
        store.add_chunks(chunk_ids, documents, embeddings, metadatas,
                         source="FAA_CFR", strategy="recursive")

        # Query karo
        results = store.query(embedding, source="FAA_CFR",
                              strategy="recursive", n_results=5)

        # Ek source ke saare strategies query karo (RAGAS ke liye)
        all_results = store.query_all_strategies(embedding, source="FAA_CFR")
    """

    def __init__(self, persist_dir: str = _PERSIST_DIR) -> None:
        # Automated missing file system path builder routine execution targets checks parameters paths check locations track structure.
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # Persistent baseline client instance instantiation passing parameters rules configurations types storage memory safe checks indicator.
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Local parameters value allocation directory assignment variable registry context layout process run trace code pointer path.
        self._persist_dir = persist_dir

        # Multi collection indexing map keys initialization multi strategy coordinates arrays schema combinations dictionary placeholders data space models mapping infrastructure pointers storage layer context layout components.
        self._collections: dict[tuple[str, str], chromadb.Collection] = {}
        # Multi cluster idempotent setup automation subroutine execution trigger method call trace level mapping code properties configuration loops context validation step.
        self._init_collections()

        # Operational summary data statistics feedback tracking logs info messaging updates console parameter instrumentation checks monitoring.
        logger.info(
            "SkyLexVectorStore initialized | persist_dir=%s | collections=%d",
            persist_dir,
            len(self._collections),
        )

    def _init_collections(self) -> None:
        """
        Saare source x strategy combinations ki collections get_or_create karo.
        HNSW cosine space configure karo — text embeddings ke liye standard.
        Idempotent — existing collections unaffected rehti hain.
        """
        # Outer matrix coordinates loops structural parameters tracking combinations checklist definitions cross scan dictionary data elements layout tracking properties context check variables mapping iterations.
        for (
            source,
            strategy,
        ), collection_name in get_all_collection_names().items():
            # Request database internal platform allocation endpoints client process methods call parameters pass configuration rule registry dynamic handle code tracing validation context parameters.
            collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata=_HNSW_METADATA,
            )
            # Inject freshly allocated system cluster object references directly inside core storage lookup dictionary variables database trace loop mapping values target.
            self._collections[(source, strategy)] = collection
            # System log indicator metrics calculations debug visibility tracking telemetry updates reporting variables tracking check trace console notifications debug stream parameters layout.
            logger.debug(
                "Collection ready | %s | chunks=%d",
                collection_name,
                collection.count(),
            )

    def _validate(self, source: str, strategy: str) -> None:
        """
        Source aur strategy valid combination hai ya nahi — early fail karo.
        """
        # Alphanumeric category tags parameters validation range control boundary conditions branch routing control check code path tracking options validation rule checklist data trace.
        if source not in SUPPORTED_SOURCES:
            # Operational fault handler system exception throw variables parameters allocation code context tracking cancel indicators raise configuration layout line.
            raise ValueError(
                f"Invalid source '{source}'. "
                f"Supported: {sorted(SUPPORTED_SOURCES)}"
            )
        # Structural dictionary cross matrix lookups parameters validation check properties checking conditions code execution selector mapping option validation branch logic framework loop variables check.
        if strategy not in CANDIDATE_STRATEGIES[source]:
            # Trigger abort transaction metrics flags exception throwing method configuration dynamic string template format updates trace logging models execution control line data.
            raise ValueError(
                f"Invalid strategy '{strategy}' for source '{source}'. "
                f"Candidates: {CANDIDATE_STRATEGIES[source]}"
            )

    def add_chunks(
        self,
        chunk_ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        source: str,
        strategy: str,
    ) -> int:
        """
        Chunks ko source x strategy collection mein add karo.

        Duplicate chunk_ids automatically skip ho jaate hain —
        ChromaDB upsert semantics use karta hai.

        Args:
            chunk_ids  : Deterministic MD5 IDs (Phase 2 se same)
            documents  : Chunk text content
            embeddings : Pre-computed float vectors (1536-dim ya 1024-dim)
            metadatas  : Per-chunk metadata dicts
            source     : Source identifier (FAA_CFR, FAA_AD, etc.)
            strategy   : Chunking strategy name (recursive, hierarchical, etc.)

        Returns:
            Number of chunks successfully added.
        """
        # Ingestion payload structural checking subroutine function execution passing parameters context validation lookups safety tracing data framework step path code loop model.
        self._validate(source, strategy)

        # Empty data lists sequences protections checkpoints conditional branching validation checks bypass redirect trace options parameter mapping metrics code variable assignments check layer logic.
        if not chunk_ids:
            # Inform metrics tracker components when vacant arrays encountered parameters settings code log trace execution summary indicator code context lookups path mapping variable updates.
            logger.warning(
                "add_chunks called with empty list | source=%s | strategy=%s",
                source,
                strategy,
            )
            return 0

        # Data integrity check — sab lists equal length honi chahiye.
        # Multi element sizing balance arrays cross check validation true assertions parameters verify dimensions matrix properties match structural evaluations checker conditions data loop indicators calculations checks.
        if not (
            len(chunk_ids)
            == len(documents)
            == len(embeddings)
            == len(metadatas)
        ):
            # Error handler mechanism exception throw variables allocation parameters structural type modifications code trace execution line report parameters lookup baseline configuration checks.
            raise ValueError(
                "chunk_ids, documents, embeddings, metadatas — "
                "sab ki length equal honi chahiye."
            )

        # Extract specific targets framework database cluster pointer reference out of composite keys dictionary tracking memory addresses configuration layout properties context data blocks.
        collection = self._collections[(source, strategy)]

        # Batch mein add karo — 500 chunks per batch.
        # Memory aur ChromaDB API limits ke andar safe rehta hai.
        batch_size = 500
        # Ingestion metrics progress metrics track variable counter increment measurements storage dynamic numeric data context initialization allocation options code trace runtime properties value model.
        total_added = 0

        # Incremental sliding steps sequence operations parsing loop configuration limits batch size parameters chunks processing execution trace pipeline blocks layout loops values items metadata.
        for i in range(0, len(chunk_ids), batch_size):
            # Dynamic boundaries limit calculation equations adjustments mathematical min function parameters check validation rules scale variable mapping logic calculations variables option.
            batch_end = min(i + batch_size, len(chunk_ids))

            # Database dynamic client upside methods execution passing targeted arrays sliced frames layout configuration context update metrics direct cloud transaction mapping rule run text indicators.
            collection.upsert(
                ids=chunk_ids[i:batch_end],
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],  # type: ignore[arg-type]
                metadatas=metadatas[i:batch_end],  # type: ignore[arg-type]
            )
            # Cumulative summary arithmetic computations increment updates tracking counters calculation loop values updates state data trace performance calculations integers logic tracking checks loop values.
            total_added += batch_end - i

        # Telemetry metrics pipeline reporting completion notice console display logging dynamic indicators overview metrics trace execution print layout block line parameters text visibility parameters variables metrics data maps framework.
        logger.info(
            "Chunks added | source=%s | strategy=%s | count=%d | total=%d",
            source,
            strategy,
            total_added,
            collection.count(),
        )

        # Output final accumulated integer processing counts back to master scheduler execution engine flow context return path properties value trace data logic framework parameter.
        return total_added

    def query(
        self,
        query_embedding: list[float],
        source: str,
        strategy: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> QueryResult:
        """
        Dense vector similarity search — specific source x strategy collection mein.

        Args:
            query_embedding : Query ka embedded vector
            source          : Kaunsa source (FAA_CFR, FAA_AD, etc.)
            strategy        : Kaunsi strategy collection (recursive, etc.)
            n_results       : Top-k results (default: 5)
            where           : Optional metadata filter

        Returns:
            QueryResult dataclass with chunks, metadata, distances.
        """
        # Source target indexing checking mapping verification structural parameters criteria lookups selector processing path validation validation indicators check multi variables mapping verification.
        self._validate(source, strategy)

        # Access system target collection pointer reference from data mapping model configurations key memory variables address layout tracking properties dictionary models composite coordinates parameters.
        collection = self._collections[(source, strategy)]

        # Collection empty hai toh early return.
        # Zero dataset validation constraint checks condition branching prevent pipeline execution faults crashes trace pointer shift logic parameters recovery setup data indicators framework monitoring operations.
        if collection.count() == 0:
            # Inform framework metrics monitors when vacant components context encountered checks parameter layout configuration options variable logging trace report line text formatting data pipelines logs.
            logger.warning(
                "Query on empty collection | source=%s | strategy=%s",
                source,
                strategy,
            )
            # Yield isolated clean empty format layout properties dynamic parameters metrics dataclass specifications return framework component path pointer destination variables check layout attributes.
            return QueryResult(
                chunk_ids=[],
                documents=[],
                metadatas=[],
                distances=[],
                source=source,
                strategy=strategy,
                query_time=0.0,
            )

        # High performance precise chronometer verification snapshot snapshot benchmark hardware time parameters logging execution trace point line clock initialization baseline precise timing tracking.
        start_time = time.perf_counter()

        # n_results collection size se zyada nahi ho sakta.
        # Mathematical ceiling limitation calculation checks adjusting target requirements count bounds variables calculations formula dynamic matching constraints limits condition scale values tracking parameter variance indices variables.
        safe_n = min(n_results, collection.count())

        # Target arguments properties parameters setup inventory listing dictionary schema parameters allocation matching tracking matrix setup variables memory config parameters lookup backend register.
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": safe_n,
            "include": ["documents", "metadatas", "distances"],
        }

        # Dynamic property lookup check checking variable parameter presence conditions check routing layout validation code trace context pipeline parameters mapping logic search dynamic attribute configuration.
        if where:
            query_kwargs["where"] = where

        # Outbound local embedded engine mapping distance equations search call method trigger execution query components returns dictionary arrays calculations vector collections functions.
        raw = collection.query(**query_kwargs)

        # Delta computation processing interval duration formulation equations rounding parameters float decimals metrics measurement latency calculations tracking values stop clock marker calculations differences.
        query_time = time.perf_counter() - start_time

        # ChromaDB batch queries ke liye nested lists return karta hai —
        # single query ke liye [0] index lena hai.
        # Micro structural array index extraction layers unpacking sub elements nested lists parameters mappings context conversion data models factory setup records dataclass maps return metrics data layout.
        return QueryResult(
            chunk_ids=raw["ids"][0],
            documents=raw["documents"][0],  # type: ignore[index]
            metadatas=raw["metadatas"][0],  # type: ignore[index]
            distances=raw["distances"][0],  # type: ignore[index]
            source=source,
            strategy=strategy,
            query_time=query_time,
        )

    def query_all_strategies(
        self,
        query_embedding: list[float],
        source: str,
        n_results: int = 5,
    ) -> dict[str, QueryResult]:
        """
        Ek source ke saare candidate strategy collections mein query karo.
        RAGAS evaluation ke liye — same query, alag strategy results compare karo.

        Args:
            query_embedding : Query ka embedded vector
            source          : Kaunsa source search karna hai
            n_results       : Top-k per strategy

        Returns:
            Dict mapping strategy_name -> QueryResult
        """
        # Category identification dynamic constraint boundaries checks lookup parameters index validator path branch code redirection mappings checks rules.
        if source not in SUPPORTED_SOURCES:
            raise ValueError(f"Invalid source '{source}'.")

        # Local results collection database dictionary initialization memory mapping configurations placeholder values keys direct updates trace arrays.
        results: dict[str, QueryResult] = {}

        # Inner matrix loop sequence scanning cross available operational target candidate strategies parameters listings sequences loop processing calculations mapping.
        for strategy in CANDIDATE_STRATEGIES[source]:
            # Dynamic vector search execution routing direct method passing parameters targets coordinates options return dataclass objects save array key dictionary.
            results[strategy] = self.query(
                query_embedding=query_embedding,
                source=source,
                strategy=strategy,
                n_results=n_results,
            )

        # Output completely compiled mapping framework results records back inside main execution orchestrator loop context stream return block data elements.
        return results

    def query_all_sources_and_strategies(
        self,
        query_embedding: list[float],
        n_results: int = 3,
    ) -> dict[str, dict[str, QueryResult]]:
        """
        Saare sources aur saari strategies mein query karo.
        Global evaluation ke liye — source unknown ho toh use karo.

        Returns:
            Dict mapping source -> {strategy -> QueryResult}
        """
        # Python nested dictionary comprehension formatting dynamic iterations maps cross matrix parameters listings execution paths loop step context architecture returns structures collections register layout values.
        return {
            source: self.query_all_strategies(
                query_embedding=query_embedding,
                source=source,
                n_results=n_results,
            )
            for source in SUPPORTED_SOURCES
        }

    def get_stats(self) -> list[CollectionStats]:
        """
        Saari collections ki current state return karo.
        Monitoring aur debugging ke liye.
        """
        # Tracking reporting matrix allocation placeholder storage mapping variables collections list memory container initialization variable data properties matrix tables.
        stats: list[CollectionStats] = []

        # Operational summary loop scanning memory items attributes properties sequence variables tracking elements map data definitions trace structure data loop configurations metrics check calculations lists.
        for (source, strategy), collection in self._collections.items():
            # Ingest standard dataclass metrics instance item mapping fields direct inside tracking listing target results array updates collection store memory parameter structures metadata parameters.
            stats.append(
                CollectionStats(
                    name=get_collection_name(source, strategy),
                    source=source,
                    strategy=strategy,
                    chunk_count=collection.count(),
                    persist_dir=self._persist_dir,
                )
            )

        # Source phir strategy ke order mein sort karo — readable output.
        # Custom collection list arrangement algorithms execution checking priorities criteria parameters sorting lambda variables logic mapping trace code structure.
        stats.sort(key=lambda s: (s.source, s.strategy))
        # Yield completed parsed quantitative collections statistics metrics array back to monitoring orchestration frameworks.
        return stats

    def print_stats(self) -> None:
        """Console pe readable stats print karo."""
        # Analytics calculations retrieval metrics function invocation passing components returns lists metadata array values map execution check variables context lookups parsing.
        stats = self.get_stats()
        # Summary arithmetic quantitative aggregation total capacity counters tracking lines mathematics addition formulations summary calculations total counter loops variables setup metrics pipeline.
        total = sum(s.chunk_count for s in stats)

        # Visual formatting border lines layout strings configuration print terminal view code indicators block design terminal line drawings configurations layout dashboard formatting.
        print("\n" + "─" * 72)
        # Main descriptive console banner updates text display metrics logging centered terminal string template format parameters typography view text visibility master summary console layouts banner.
        print(f"{'SkyLex Vector Store Stats':^72}")
        # Segment separator boundary line drawing format definitions adjustments console monitors reporting data check layout visibility indicators terminal separator lines graphics alignment separators dashboard columns labels framework padding.
        print("─" * 72)
        # Table columns definitions headers character width parameters typography structural setup values mapping console channel direct layout text string writes.
        print(
            f"  {'Source':<12} │ {'Strategy':<22} │ {'Collection':<26} │ Chunks"
        )
        # Spacing divider segment drawing layout graphics definitions specifications console line draw indicator options parameter code matching line reporting console visual separations.
        print("─" * 72)

        # Traversal reporting array loops variables elements outputs layout padding formatting characters conversions integers tracking display script loops variables metrics context iterative mapping terminal write operations.
        for s in stats:
            print(
                f"  {s.source:<12} │ {s.strategy:<22} │ "
                f"{s.name:<26} │ {s.chunk_count:>6}"
            )

        # Visual footer row calculations characters adjustments alignments width formatting values aggregate metrics print console dynamic text conversions.
        print("─" * 72)
        print(f"  {'TOTAL':<12} │ {'':<22} │ {'':<26} │ {total:>6}")
        # Visual footer bounding frame closing layout canvas draw completed lines logs complete confirmation typography border layout indicators framework components completion border marking lines.
        print("─" * 72 + "\n")

    def delete_collection(self, source: str, strategy: str) -> None:
        """
        Ek source x strategy collection delete karke re-create karo.
        Re-indexing ke liye use karo.

        WARNING: Irreversible — us collection ka saara data jaayega.
        """
        # Input validation identifier criteria properties target check parameters specifications check loop branching routing control execution layer context safety checkpoint trace parameters checklist validation.
        self._validate(source, strategy)

        # Access system collection storage layout string matching template name variable data check parameter config setup dictionary composite string calculations values.
        collection_name = get_collection_name(source, strategy)
        # Request backend client platform configuration deletion engine method call tracking instance unique transaction code execute cancel operations client runtime infrastructure.
        self._client.delete_collection(collection_name)

        # Fresh collection re-create karo.
        # Re allocate fresh structure object framework template memory context inside local tracking dict variable endpoint mapping parameters initialization call execution path configurations mapping options.
        self._collections[(source, strategy)] = (
            self._client.get_or_create_collection(
                name=collection_name,
                metadata=_HNSW_METADATA,
            )
        )

        # Ingestion analytics pipeline tracking indicators warnings console logging updates tracer system alert message parameter checking configuration rule code dynamic logs display text execution pipelines instrumentation alerts.
        logger.warning(
            "Collection deleted and re-created | source=%s | strategy=%s | name=%s",
            source,
            strategy,
            collection_name,
        )

    def delete_source(self, source: str) -> None:
        """
        Ek source ki saari strategy collections delete karke re-create karo.
        Source-level re-indexing ke liye.

        WARNING: Irreversible.
        """
        # Range criteria parameters verification data sets checks validation control branching routing options parameters exception indicators.
        if source not in SUPPORTED_SOURCES:
            raise ValueError(f"Invalid source '{source}'.")

        # Structural listings loops scanning targeted categories strategies parameters options matrix data clear subroutines calls trace code levels execution paths.
        for strategy in CANDIDATE_STRATEGIES[source]:
            self.delete_collection(source, strategy)

        # Operational status updates logger warnings notifications print out terminal dashboard analytics traces metrics monitoring checking.
        logger.warning(
            "All collections deleted for source=%s | strategies=%s",
            source,
            CANDIDATE_STRATEGIES[source],
        )

    def reset_all(self) -> None:
        """
        Saari 15 collections delete karke fresh start karo.

        WARNING: Irreversible — poora vector store wipe ho jaayega.
        Sirf development/re-indexing ke liye.
        """
        # Outer matrix dictionary elements list context copy modifications loops sequence traversal tracking options parameters checklist control loop branch data components erasure loops mapping trace.
        for source in list(SUPPORTED_SOURCES):
            # Master structural wipe call executing clear signal routines across entire strategy tree matrix data configuration lines deletion.
            self.delete_source(source)

        # Intercept database level clear signal context updates console logs statements tracking analytics parameters diagnostics screen warnings report text complete validation indicators tracker code manual setups alerts notice.
        logger.warning("All collections reset — vector store is empty.")