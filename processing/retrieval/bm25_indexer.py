"""
processing/retrieval/bm25_indexer.py

SkyLex BM25 (Best Match 25) Index Builder.

Kya karta hai:
  - ChromaDB collections se saare chunks fetch karta hai
  - Har chunk ko tokenize karta hai (lowercase, punctuation remove, stopwords remove)
  - BM25Okapi index build karta hai
  - Index ko pickle file mein save karta hai — retrieval time pe dobara build na ho

Kyun alag index file:
  - BM25 index build karna ek baar ka kaam hai — 72,827 chunks ka
  - Har query pe dobara build karna impractical hai (~30-60 seconds)
  - Pickle se load karna instant hai (~0.5 seconds)

Index naming convention:
  data/bm25_indexes/{source_lower}_{strategy}.pkl
  Example: data/bm25_indexes/faa_cfr_recursive.pkl

BM25 Parameters (BM25Okapi defaults — research-backed):
  k1 = 1.5  — Term frequency saturation (diminishing returns control)
  b  = 0.75 — Document length normalization strength
"""

from __future__ import annotations

import pickle
import re
import string
from datetime import datetime
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from monitoring.logger import get_logger
from processing.vector_store import CANDIDATE_STRATEGIES, SkyLexVectorStore

# BM25 lexical indexing pipeline ke lifecycle events, tokenization warnings, aur disk I/O operations ko trace karne ke liye dedicated logger initialize kiya.
logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Local filesystem target path jahan serialized pickle binaries as lexical search index store honge.
_BM25_INDEX_DIR: Path = Path("data/bm25_indexes")

# English stopwords — BM25 mein noise hote hain, remove karo.
# Aviation domain ke liye standard English stopwords use kar rahe hain.
# "section", "part", "shall" intentionally NAHI hataaye — regulatory text mein meaningful hain.
# Custom vocabulary definitions set taaki common conjunctions aur prepositions rank scoring algorithms ko dilute na karein, keeping domain-specific terms intact.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "this", "that",
    "these", "those", "it", "its", "as", "not", "no", "nor", "so",
    "yet", "both", "either", "neither", "each", "any", "all", "than",
    "such", "if", "then", "than", "too", "very", "just", "more",
})

# ChromaDB se ek baar mein itne chunks fetch karenge — memory safe.
# Pagination bounds control parameter taaki massive collections fetch karte time OOM (Out Of Memory) crashes prevent ho sakein.
_FETCH_BATCH_SIZE: int = 1000


# ── Tokenizer ─────────────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """
    Chunk text ko BM25 ke liye tokens mein convert karo.

    Pipeline:
      1. Lowercase karo — "Section" aur "section" same ho jaayein
      2. Punctuation remove karo — lekin hyphen rakho (AD-2023-15-08 ke liye)
      3. Whitespace se split karo
      4. Stopwords remove karo
      5. Empty tokens aur single characters remove karo

    Hyphen kyun rakha:
      FAA ADs mein "AD-2023-15-08" jaise codes hote hain.
      Agar hyphen remove karo toh "2023", "15", "08" alag tokens bante hain
      aur exact AD number match impossible ho jaata hai.

    Args:
        text: Raw chunk content string

    Returns:
        List of cleaned tokens

    Example:
        Input:  "§ 121.135 requires each certificate holder to maintain a manual"
        Output: ["121.135", "requires", "certificate", "holder", "maintain", "manual"]
    """
    # Lowercase — case-insensitive matching ke liye.
    # Text standardization taaki query aur corpus dono same string formatting level par compare ho sakein.
    text = text.lower()

    # Section symbol aur special chars replace karo space se.
    # Lekin decimal points aur hyphens rakho — "121.135" aur "AD-2023" ke liye.
    # Regex substitution engine call jahan specific legal markers aur symbols ko spaces se nullify kiya jata hai bina numerical sequence break kiye.
    text = re.sub(r"[§©®™]", " ", text)

    # Punctuation remove karo except hyphen aur decimal point.
    # string.punctuation se hyphen (-) aur dot (.) alag karo.
    # Lookup table string operations optimization mapping; standard punctuation string me se aviation identifiers ke symbols filter-out karke baaki sab whitespace me convert hote hain.
    punct_to_remove = string.punctuation.replace("-", "").replace(".", "")
    text = text.translate(str.maketrans(punct_to_remove, " " * len(punct_to_remove)))

    # Whitespace se split karo.
    # Standard splitting method jo tab, newline aur standard spaces ko delimiter maan kar unigram list generate karta hai.
    raw_tokens = text.split()

    # Filter: stopwords remove karo, single chars remove karo, empty remove karo.
    # Optimized list comprehension applying triple conditions: cross-referencing frozen set, checking dimensional string lengths, aur whitespace validation.
    tokens = [
        token
        for token in raw_tokens
        if token not in _STOPWORDS
        and len(token) > 1
        and not token.isspace()
    ]

    # Return sanitized lexical index tokens string list directly into the algorithm loop memory placeholder.
    return tokens


# ── Index Builder ─────────────────────────────────────────────────────────────


def _get_index_path(source: str, strategy: str) -> Path:
    """Source + strategy ke liye index file path return karo."""
    # Source categories normalization lowercase conversion aur path string generation configuration rules execution step.
    source_clean = source.lower().replace(" ", "_")
    # Exact filesystem addressing trace returning structured template path mapped to predefined persistence directory.
    return _BM25_INDEX_DIR / f"{source_clean}_{strategy}.pkl"


def build_bm25_index(
    source: str,
    strategy: str,
    store: SkyLexVectorStore,
    force_rebuild: bool = False,
) -> Path:
    """
    Ek source × strategy combination ke liye BM25 index build karo.

    Process:
      1. Check karo ki index already exist karta hai ya nahi
      2. ChromaDB se saare chunks fetch karo (batched)
      3. Har chunk tokenize karo
      4. BM25Okapi index build karo
      5. Pickle file mein save karo

    Args:
        source        : FAA_CFR, FAA_AD, etc.
        strategy      : recursive, hierarchical, etc.
        store         : Initialized SkyLexVectorStore
        force_rebuild : True → existing index delete karke fresh build karo

    Returns:
        Path to saved pickle file.
    """
    # Fetch dynamically computed disk storage location paths for current strategy mapping.
    index_path = _get_index_path(source, strategy)

    # Already exist karta hai aur force_rebuild nahi hai toh skip karo.
    # Optimization bypass check taaki massive compute heavy indexing loops dobara trigger na ho agar binary data map pehle se disk par available ho.
    if index_path.exists() and not force_rebuild:
        logger.info(
            "BM25 index already exists — skipping | source=%s | strategy=%s | path=%s",
            source, strategy, index_path,
        )
        return index_path

    # Telemetry notification milestone start trigger logs printout parameters indicating indexing computation has formally begun.
    logger.info(
        "Building BM25 index | source=%s | strategy=%s",
        source, strategy,
    )

    # ChromaDB collection se chunks fetch karo.
    # Format template string variables extraction target mappings and backend persistent store retrieval call trigger loop.
    collection_name = f"skylex_{source.lower()}_{strategy}"
    chunk_ids, texts = _fetch_chunks_from_chromadb(store, source, strategy)

    # Missing database entries protection bounds checklist routing to exception handlers aborting execution gracefully.
    if not chunk_ids:
        raise ValueError(
            f"No chunks found in collection '{collection_name}'. "
            f"Embedding pipeline pehle chalao."
        )

    # Ingestion complete notification logger tracking total sequence parameters output values indicating memory load status context.
    logger.info(
        "Chunks fetched | source=%s | strategy=%s | count=%d",
        source, strategy, len(chunk_ids),
    )

    # Tokenize karo — har chunk ko tokens ki list mein convert karo.
    # Matrix map conversions processing linear arrays inputs texts lists using defined extraction unigram subroutine functions.
    logger.info("Tokenizing chunks | source=%s | strategy=%s", source, strategy)
    corpus_tokens: list[list[str]] = [tokenize(text) for text in texts]

    # Empty token lists check karo — BM25 crash kar sakta hai.
    # Critical evaluation parameters variables counting loops checking edge cases logic variables options structural arrays mapping.
    empty_count = sum(1 for tokens in corpus_tokens if not tokens)
    if empty_count > 0:
        # Fallback padding mapping values trace conditions checks logic logging empty arrays variables to ensure okapi math formulation does not division fault.
        logger.warning(
            "Empty token lists found | count=%d — replacing with ['empty']",
            empty_count,
        )
        corpus_tokens = [
            tokens if tokens else ["empty"]
            for tokens in corpus_tokens
        ]

    # BM25Okapi index build karo.
    # k1=1.5, b=0.75 — research-backed defaults (Robertson & Zaragoza, 2009).
    # Core instantiation metrics algorithms computation parameter processing function loop triggering TF-IDF mathematical variations index mapping variables layout.
    logger.info(
        "Building BM25Okapi index | source=%s | strategy=%s | tokens=%d",
        source, strategy, sum(len(t) for t in corpus_tokens),
    )
    bm25_index = BM25Okapi(corpus_tokens, k1=1.5, b=0.75)

    # Save karo — pickle format mein.
    # Standard dictionary object wrapping packaging data schemas fields variables values for unified storage persistence mapping logic variables.
    index_data: dict[str, Any] = {
        "bm25_index":    bm25_index,
        "chunk_ids":     chunk_ids,
        "corpus_tokens": corpus_tokens,
        "source":        source,
        "strategy":      strategy,
        "chunk_count":   len(chunk_ids),
        "built_at":      datetime.utcnow().isoformat(),
    }

    # Verify storage pathways creation structural directories existence rules bounds checking logic parameter.
    _BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Local binary stream writer setup using fastest protocol data byte conversions directly saving RAM memory footprints onto disk sectors.
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Write confirmation updates display tracing pipeline progress completions variable properties data path lines parameters framework logic trace variables values.
    logger.info(
        "BM25 index saved | source=%s | strategy=%s | path=%s | chunks=%d",
        source, strategy, index_path, len(chunk_ids),
    )

    # Returning final complete destination binary reference strings context values variables tracking arrays options loop.
    return index_path


def _fetch_chunks_from_chromadb(
    store: SkyLexVectorStore,
    source: str,
    strategy: str,
) -> tuple[list[str], list[str]]:
    """
    ChromaDB collection se saare chunk IDs aur texts fetch karo.

    ChromaDB mein directly "get all" karne ka ek limitation hai —
    bohot bade collections mein memory issue ho sakta hai.
    Isliye batched fetch use karte hain.

    Returns:
        Tuple of (chunk_ids, texts)
    """
    # Vector store cluster mapping database endpoints pointers fetching dictionary variables parameter references mapping lookups.
    collection = store._collections.get((source, strategy))

    # Error handling mechanism tracking unknown references validation constraints check limits exceptions logic.
    if collection is None:
        raise ValueError(
            f"Collection not found for source='{source}', strategy='{strategy}'"
        )

    # Database internal measurement tracking counting attributes mapping value trace parameters calculation integer output variables logic.
    total_count = collection.count()

    # Empty table protections validation sequences bypass redirect logic loops returning vacant lists variables memory setup.
    if total_count == 0:
        return [], []

    # Local accumulators list placeholders memory container targets map structures dictionary properties configuration loop data streams variables.
    all_ids: list[str] = []
    all_texts: list[str] = []

    # Batched fetch — memory safe.
    # Sub-partitioning execution scanning variables iteration ranges checking rules options mapping context boundaries offset variables index values tracking bounds.
    for offset in range(0, total_count, _FETCH_BATCH_SIZE):
        # Trigger query retrieving metadata parameters conditions values limits database calls pagination mapping variable variables dictionary options values.
        batch = collection.get(
            limit=_FETCH_BATCH_SIZE,
            offset=offset,
            include=["documents"],  # type: ignore[arg-type]

        )

        # Merge fetched dictionary subsets elements parameters into unified continuous lists structures tracking matrices.
        all_ids.extend(batch["ids"])
        all_texts.extend(batch["documents"] or [])  # type: ignore[arg-type]

        # Debug level progress tracking metrics loops values evaluations metrics console update logging text print layout formatting context framework structures parameters.
        logger.debug(
            "Fetched batch | source=%s | strategy=%s | offset=%d | count=%d",
            source, strategy, offset, len(batch["ids"]),
        )

    # Full collection export output arrays tuple formats return path context structure variable maps.
    return all_ids, all_texts


# ── Index Loader ──────────────────────────────────────────────────────────────


def load_bm25_index(source: str, strategy: str) -> dict[str, Any]:
    """
    Disk se BM25 index load karo.

    Args:
        source   : FAA_CFR, FAA_AD, etc.
        strategy : recursive, hierarchical, etc.

    Returns:
        Dict with keys: bm25_index, chunk_ids, corpus_tokens, source,
                        strategy, chunk_count, built_at

    Raises:
        FileNotFoundError: Agar index exist nahi karta.
    """
    # Fetch dynamically computed disk storage location paths for requested parameters combinations.
    index_path = _get_index_path(source, strategy)

    # Integrity verification constraints condition checks preventing invalid read attempts fault path trace operations loops rules parameters logic context tracking.
    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {index_path}. "
            f"Pehle build_bm25_index() chalao."
        )

    # Binary read byte sequences channel opening operations parsing parameters structures variables load definitions memory assignments configurations maps structure variables.
    with open(index_path, "rb") as f:
        index_data: dict[str, Any] = pickle.load(f)

    # Diagnostic parameters check logging updates terminal metrics outputs text string values variable parameter options.
    logger.debug(
        "BM25 index loaded | source=%s | strategy=%s | chunks=%d",
        source, strategy, index_data["chunk_count"],
    )

    # Return fully structured instantiated python dictionary payload holding lexical algorithms memory context fields maps tracking object.
    return index_data


# ── Full Build Pipeline ───────────────────────────────────────────────────────


def build_all_indexes(
    sources: list[str] | None = None,
    force_rebuild: bool = False,
) -> dict[str, Path]:
    """
    Saare source × strategy combinations ke liye BM25 indexes build karo.

    Args:
        sources       : Specific sources — None matlab saare 5
        force_rebuild : True → saare indexes fresh se build karo

    Returns:
        Dict mapping "source__strategy" → index file path
    """
    # Fallback configuration default selection tracking parameters loops lookup validation limits variable targets values options condition map layout.
    target_sources = sources or list(CANDIDATE_STRATEGIES.keys())
    # Persistent target tracking databases vector clusters instantiation parameters setup directory parameters configuration variables models data code framework loop.
    store = SkyLexVectorStore()

    # Master results records dictionaries configurations layout memory path placeholder structure tracking properties matrix array lists metrics tracking structures assignments.
    results: dict[str, Path] = {}
    # Arithmetic combinations counts summation calculations tracking metrics target mapping parameters evaluations context index loop.
    total = sum(len(CANDIDATE_STRATEGIES[s]) for s in target_sources)
    # Status counters integer assignment limits processing execution evaluation value context.
    done = 0

    # Iteration spanning multi dimensional category collections loop structure trace options metrics sequence matrix definitions maps data logic framework code trace execution tracking options bounds.
    for source in target_sources:
        # Second tier loop iterating specific algorithm mapping criteria conditions variable values tracking loops path configurations trace lines execution properties options mappings array.
        for strategy in CANDIDATE_STRATEGIES[source]:
            run_key = f"{source}__{strategy}"
            done += 1
            # Update sequence execution progress trace notifications loop console print outputs string layout properties parameters string display template definitions code string text check updates.
            logger.info(
                "Building index %d/%d | %s", done, total, run_key
            )

            try:
                # Execution method triggering core index mapping subroutine tracking parameters setup data passing configurations path assignment return paths.
                path = build_bm25_index(
                    source=source,
                    strategy=strategy,
                    store=store,
                    force_rebuild=force_rebuild,
                )
                # Recording returned path string addresses into central tracking lookup lists metrics validation mapping.
                results[run_key] = path

            except Exception as e:
                # Catching execution process errors reporting failures trace logging warning outputs visibility screen logs updates strings template context mapping logic trace structure loop context.
                logger.error(
                    "Failed to build index | %s | error=%s", run_key, e
                )

    # Master final log summary reports data configuration definitions properties metrics terminal write console display options trace variables loop setup data configuration variables tracking frameworks lists models strings framework parameter.
    logger.info(
        "All BM25 indexes built | total=%d | success=%d",
        total, len(results),
    )

    # Yield completed path addresses mapping strings records dictionary container arrays framework modules flow step parameters options path return value data structures.
    return results


def print_index_stats() -> None:
    """Disk pe existing BM25 indexes ki stats print karo."""
    # Verifying core physical repository constraints validations conditions error handling bypass options redirect execution rules properties.
    if not _BM25_INDEX_DIR.exists():
        print("\n  No BM25 indexes found. Run build_all_indexes() first.\n")
        return

    # Glob sorting arrays listings file contents structure loops scan parameters configurations layout string matches file directory definitions.
    index_files = sorted(_BM25_INDEX_DIR.glob("*.pkl"))

    # Size validations checking logic to bypass empty files collections pipelines run checks control branching path sequence redirect setup validation options layout models properties logic loops.
    if not index_files:
        print("\n  No BM25 indexes found.\n")
        return

    # Border typography separator console lines formatting drawing rendering boundaries view limits text display format rules setups.
    print("\n" + "─" * 70)
    # Master dashboard banner string visual updates terminal variables configuration data mapping layout definitions parameters context limits loop print context tracking rules mapping string view metrics limits framework line rules options logic code view string output formats parameters text.
    print(f"{'BM25 Index Stats':^70}")
    print("─" * 70)
    # Table headers padding alignments values printing console parameters tracking text options layout parameters.
    print(f"  {'File':<40} │ {'Chunks':>8} │ {'Size':>8} │ Built At")
    print("─" * 70)

    # Summary accumulator integer initialization state mapping check values processing framework setup.
    total_chunks = 0

    # Iteration scanning actual extracted filesystem metadata paths trace layout mapping sequences validation list array context values processing configuration loops parameter data variables lists elements logic configurations layout check variables loops framework data sets loop structure trace.
    for pkl_file in index_files:
        try:
            # Operational memory bytes array loads decoding structures dictionary object structures.
            with open(pkl_file, "rb") as f:
                data: dict[str, Any] = pickle.load(f)

            # Metadata parsing floating math logic conversion divisions size calculation formatting.
            size_mb = pkl_file.stat().st_size / (1024 * 1024)
            total_chunks += data["chunk_count"]

            # Visual column outputs rows formatting values mappings parameter rendering padding text display values limits tracking numbers print loop template structures.
            print(
                f"  {pkl_file.name:<40} │ {data['chunk_count']:>8} │ "
                f"{size_mb:>6.1f}MB │ {data['built_at'][:19]}"
            )
        except Exception as e:
            # Fallback error catch print formats parameters layout configuration mapping variables tracking string updates error metrics reports context display values parameters logic trace line variables limits array lists check text updates output details.
            print(f"  {pkl_file.name:<40} │ ERROR: {e}")

    # Closing visual structural block boundaries printing formatting layout string line border limits trace tracking data context parameters string updates parameters framework values mapping text format line check text rendering parameter.
    print("─" * 70)
    print(f"  {'TOTAL':<40} │ {total_chunks:>8}")
    print("─" * 70 + "\n")