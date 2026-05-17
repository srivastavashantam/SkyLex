"""
processing/strategies/structure_aware_chunker.py

Structure-Aware Chunker with Metadata Injection.

Anthropic internal (2024) approach — chunk content mein parent section
header explicitly inject karo taaki har chunk self-contained ho.

Three-stage pipeline:
  Stage 1 — Regex-based section boundary detection (§, numbered, ALL CAPS)
  Stage 2 — Oversized sections ko RecursiveCharacterTextSplitter se split
  Stage 3 — Har chunk ke content mein parent section header prefix inject

Example output chunk:
    "[FAA_CFR | § 121.135]\n(a) Each manual required by this subpart..."

Why metadata injection:
  - Retrieval mein chunk apna own context carry karta hai
  - "§ 121.135 kya kehta hai" query → direct semantic match
  - No dependency on surrounding chunks for context
  - Dense Passage Retrieval (Karpukhin et al., 2020) recommendation

Pros : Self-contained chunks, best retrieval precision, size controlled
Cons : Chunk size slightly larger (header overhead), regex miss on irregular PDFs
Use  : FAA_CFR, FAA_AD, DGCA_CAR — structured regulatory sources
"""

from __future__ import annotations

import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processing.base_chunker import BaseChunker, ChunkedDocument
from monitoring.logger import get_logger

# Complex asynchronous parsing systems aur document boundaries partitioning tracks ko lifecycle monitor karne k liye logger setup initialize kiya.
logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Ingestion text data pipeline me har single segment fragment text chunk k liye max length target constraints metric integer value variable setup.
_MAX_CHUNK_CHARS: int = 1800
# Continuous layout sequence elements overlapping parameters metrics preservation bounds sliding memory window configuration.
_CHUNK_OVERLAP: int = 180  # 10% of max chunk size

# Section patterns — priority order (most specific first)
# Multi hierarchy sorting matching patterns compilation checks structures lists setups indices definitions configurations array trace tracking rules.
_SECTION_PATTERNS: list[re.Pattern[str]] = [
    # Aviation regulations standard document sections index patterns checks lookahead tracking matching setup boundary.
    re.compile(r"(?=§\s*\d+[\.\d]*)", re.MULTILINE),
    # High specific structural level sub headings numeric multi level sequence trackers definitions match lookups.
    re.compile(r"(?=^\d+\.\d+[\.\d]*\s+[A-Z])", re.MULTILINE),
    # Main continuous point numbering sections tracking layout expressions filter code boundaries mapping configurations.
    re.compile(r"(?=^\d+\.\s+[A-Z])", re.MULTILINE),
    # Standalone master capitalized headings identifiers block strings patterns matching limits tracking indicators evaluation.
    re.compile(r"(?=^[A-Z][A-Z\s]{8,}$)", re.MULTILINE),
]

# Section header extractor — first meaningful line of a section
# Documents parts first valid semantic metadata headings extraction optimization formulas text scanning configuration check pattern regex model.
_HEADER_EXTRACTOR: re.Pattern[str] = re.compile(
    r"^(§\s*[\d\.]+[^\n]*|[\d\.]+\s+[A-Z][^\n]*|[A-Z][A-Z\s]{4,})",
    re.MULTILINE,
)

# Inherent static backup layout initialization wrapper memory placeholder configuration model data boundaries allocation trace system pipeline model setup.
_FALLBACK_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_MAX_CHUNK_CHARS,
    chunk_overlap=_CHUNK_OVERLAP,
    separators=["\n\n", ".\n", ". ", "! ", "? ", "\n", " ", ""],
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_section_header(section_text: str) -> str:
    """
    Section text se first meaningful header line extract karo.
    Metadata injection prefix ke roop mein use hoga.
    """
    # Performance overhead checks control karne k liye pure document blocks ke bajaye starting 500 characters context block window scan trigger loop kiya.
    match = _HEADER_EXTRACTOR.search(section_text[:500])

    # Extraction regex condition verified execution mapping validation pointer tracking branch logic control check routing check.
    if match:
        # Context extraction string constraints parameters properties bounds length slice maximum truncation index calculation safe storage trace metadata.
        return match.group(0).strip()[:120]  # Max 120 chars — reasonable prefix

    # Fallback — first non-empty line
    # Regex extraction mismatch checks recovery layer layout configurations lists strings iterations lines traversal matrix track execution.
    for line in section_text.splitlines():
        # Text block alignment padding formatting clear space parameters read dynamic layout operations track run execution.
        stripped = line.strip()

        # Check structural properties string text validations to guarantee target data blocks hold authentic alphanumeric variables mapping flow.
        if stripped:
            # First clean found valid metadata title header return execution route fallback string parameters selection tracking block path data trace.
            return stripped[:120]

    # Vacant defaults text model configuration mapping parameters framework indicator string layout fallback structure return path.
    return ""


def _inject_header(chunk_text: str, header: str, source: str) -> str:
    """
    Chunk content mein parent section header inject karo as prefix.
    Format: "[SOURCE | HEADER]\\nchunk_text"
    Agar chunk already header se start hota hai toh inject skip karo.
    """
    # Context variable tracking value availability validation checking condition routing safe fallback path structural evaluation check branch rule trace block layout data.
    if not header:
        return chunk_text

    # Standard format metadata schema generation variable concatenation layout configuration string tracking template values definitions mapping.
    prefix: str = f"[{source} | {header}]\n"

    # Avoid duplicate injection — agar chunk already prefix se start karta hai
    # Double validation metadata override tracking data leak safeguards metrics collision index check performance logic criteria filter check branch layer code loop.
    if chunk_text.startswith(prefix):
        return chunk_text

    # Merge isolated metadata structured template payload seamlessly right before the target structural chunk body elements string memory map setup trace flow pointer data.
    return prefix + chunk_text


def _split_on_boundaries(content: str) -> list[str]:
    """
    Regex patterns se section boundaries detect karo.
    Pehli successful split return hoti hai — koi boundary nahi mili
    toh full content single element list mein.
    """
    # Strict order hierarchy prioritized regex definitions checklists tracking loop configuration matching indicators execution traversal check run.
    for pattern in _SECTION_PATTERNS:
        # Array structural content separation matrix computations, whitespace cleanups inline operations list comprehensions extraction variables mapping logic check filter array lists trace items run.
        sections: list[str] = [
            s.strip() for s in pattern.split(content) if s.strip()
        ]

        # Valid text partitions tracking scale thresholds evaluation matching indicators conditional counting verification routing step branch logic pointer check dynamic trace layer.
        if len(sections) > 1:
            # Debug tracking instrumentation log status telemetry terminal reporting indicators values console view validation processing trace logging info print.
            logger.debug(
                "StructureAwareChunker: pattern matched — %d sections",
                len(sections),
            )
            # Yield isolated target document sections segments textual strings collections layout configuration return framework data elements block pointer path.
            return sections

    # Overall structural regex mapping checks exhaustive calculations cycles missed status trace indication logger console updates parameters tracker record data layout.
    logger.debug("StructureAwareChunker: no boundary found — single chunk")
    # Linear fallback safety routing capsule array encapsulate direct return operations framework design model setup matrix properties trace flow logic step.
    return [content]


# ── Chunker ───────────────────────────────────────────────────────────────────


class StructureAwareChunker(BaseChunker):
    """
    Three-stage structure-aware chunker with metadata header injection.

    Stage 1 — Section boundary detection via regex
    Stage 2 — Size-controlled recursive splitting of oversized sections
    Stage 3 — Parent section header injected into each chunk as prefix

    Args:
        max_chunk_size : Hard size limit per chunk (default: 1800)
        chunk_overlap  : Overlap for recursive fallback (default: 180)
    """

    # Advanced retrieval indexing dashboard matrix data comparison variants testing evaluation RAG pipeline track configuration standard identifying design metadata key parameter target register name mapping.
    strategy_name: str = "structure_aware"

    def __init__(
        self,
        max_chunk_size: int = _MAX_CHUNK_CHARS,
        chunk_overlap: int = _CHUNK_OVERLAP,
    ) -> None:
        # Dynamic variable bounds parsing constraints capacity sizing parameters tracking local properties variables directory context values allocation mapping storage parameter setup.
        self.max_chunk_size = max_chunk_size
        # Overlapping boundary context retention metrics alignment buffers storage variables settings configuration indices properties assignment tracker memory map tracking logic.
        self.chunk_overlap = chunk_overlap

        # Stage 2 secondary internal recursive mechanical structural splitter algorithm constructor configuration instantiate runtime processor layer component property mapping setup.
        self._splitter = RecursiveCharacterTextSplitter(
            # Structural single block maximum dimensions capacity bounds limit value pass assignment validation criteria matrix processing tracker trace code step.
            chunk_size=max_chunk_size,
            # Sliding memory windows parameters data synchronization sequence tracking factor parameters pass setup configurations parameter metric factor indicators check.
            chunk_overlap=chunk_overlap,
            # Hierarchical scanning tracking priority boundaries delimiter list configurations symbols matching check lookups standard symbols patterns lists array string definitions.
            separators=["\n\n", ".\n", ". ", "! ", "? ", "\n", " ", ""],
        )

    def _chunk_document(
        self,
        doc: dict[str, Any],
    ) -> list[ChunkedDocument]:
        # Payload dynamic container data mapping content safe reading parameter dictionary configurations lookups key validation properties index trace tracking variables default value fetch context map step.
        content: str = doc.get("content", "")
        # Category classification indexing tracker descriptor string read parameters lookup validations target values storage pointer mapping read pipeline step tracing asset layout logic.
        source: str = doc.get("source", "")

        # Stage 1 — Section boundary detection
        # Core partition bounds calculation script execute subroutine invoke run, text regex separation array strings extraction maps tracking process indicator metrics.
        sections: list[str] = _split_on_boundaries(content)

        # Stage 2 + Stage 3 — Size control + header injection
        # Hard limits capacity filter output database items compilation array container placeholder tracking registry dynamic data collection records tracking initialization list.
        final_texts: list[str] = []

        # Structural segments lists sequence scanning traversing loop calculations iteration elements context configuration execution control variables loop block path track.
        for section in sections:
            # Extract header before any splitting
            # Metadata title identification parsing subroutine execution trace pass components data fields maps returns variables lookup tracking parameters step trace call.
            header: str = _extract_section_header(section)

            # Target item text block character capacity length validation constraints checks dynamic evaluation scale rule checking branch pointer control logic layer statement.
            if len(section) > self.max_chunk_size:
                # Oversized — recursive split first, then inject header
                # Mechanical structural fallback parsing routine options invoke calculations chunks collections text arrays generated list outputs metrics run code block index data trace mapping.
                sub_chunks: list[str] = self._splitter.split_text(section)

                # Segmented components array items iteration scanning sequence loops evaluation tracking internal variables elements collection traversal workflow loop.
                for sub in sub_chunks:
                    # Dynamic metadata template payload inject handler subroutine call pass variables parameters transformations updates destination repository buffer array block context linear merge.
                    final_texts.append(_inject_header(sub, header, source))
            else:
                # Size OK — inject header directly
                # Bounded elements within safe capacity boundaries conditions direct parameters pass configuration dynamic prefix mapping method target layout context array list append updates path trace.
                final_texts.append(_inject_header(section, header, source))

        # Abstract standard inherited parent model abstract pattern structure method call dynamic orchestration properties configurations data elements factory mapping components structural dataclass lists return pipeline flow.
        return self._build_chunks(final_texts, doc, self.strategy_name)