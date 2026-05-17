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

# Har chunking operation ka lifecycle monitor karne ke liye logger setup kiya.
logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Har chunk ki maximum character length — embedding model limit ke andar safe rehne ke liye.
_MAX_CHUNK_CHARS: int = 1800

# Adjacent chunks ke beech shared characters — boundary pe context loss prevent karne ke liye.
_CHUNK_OVERLAP: int = 180  # 10% of max chunk size

# Section boundary patterns — priority order (most specific first).
# Aviation regulatory documents mein yeh patterns section boundaries identify karte hain.
_SECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?=§\s*\d+[\.\d]*)", re.MULTILINE),           # § 121.1 — FAA/DGCA standard
    re.compile(r"(?=^\d+\.\d+[\.\d]*\s+[A-Z])", re.MULTILINE), # 1.2.3 HEADING — multi-level
    re.compile(r"(?=^\d+\.\s+[A-Z])", re.MULTILINE),           # 1. HEADING — single level
    re.compile(r"(?=^[A-Z][A-Z\s]{8,}$)", re.MULTILINE),       # ALL CAPS — FAA AD headers
]

# Section header extractor — section text ki first meaningful line nikalta hai.
# Metadata injection prefix ke roop mein use hoga.
_HEADER_EXTRACTOR: re.Pattern[str] = re.compile(
    r"^(§\s*[\d\.]+[^\n]*|[\d\.]+\s+[A-Z][^\n]*|[A-Z][A-Z\s]{4,})",
    re.MULTILINE,
)

# Fallback splitter — oversized sections ke liye jab Stage 2 trigger hota hai.
# ".\n" deliberately exclude kiya — aviation text mein section numbers (§ 121.\n)
# aur list items pe yeh pattern aata hai jo actual sentence endings nahi hain.
_FALLBACK_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_MAX_CHUNK_CHARS,
    chunk_overlap=_CHUNK_OVERLAP,
    separators=["\n\n", ". ", "! ", "? ", "\n", " ", ""],
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_section_header(section_text: str) -> str:
    """
    Section text se first meaningful header line extract karo.
    Metadata injection prefix ke roop mein use hoga.
    Sirf pehle 500 chars scan karo — performance overhead avoid karne ke liye.
    """
    match = _HEADER_EXTRACTOR.search(section_text[:500])
    if match:
        return match.group(0).strip()[:120]  # Max 120 chars — reasonable prefix

    # Fallback — pehli non-empty line return karo
    for line in section_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]

    return ""


def _inject_header(chunk_text: str, header: str, source: str) -> str:
    """
    Chunk content mein parent section header inject karo as prefix.
    Format: "[SOURCE | HEADER]\\nchunk_text"

    Skip conditions:
      - Header empty hai — inject karne se noisy prefix banega
      - Chunk already is prefix se start hota hai — duplicate avoid karo
    """
    # Empty header pe inject mat karo — SKYBRARY jaise unstructured sources mein
    # meaningful header nahi hota, prefix inject karna embedding quality giraa deta hai.
    if not header:
        return chunk_text

    prefix: str = f"[{source} | {header}]\n"

    # Duplicate injection avoid karo
    if chunk_text.startswith(prefix):
        return chunk_text

    return prefix + chunk_text


def _split_on_boundaries(content: str) -> list[str]:
    """
    Regex patterns se section boundaries detect karo.
    Pehli successful split (sections > 1) return hoti hai.
    Koi boundary nahi mili toh full content single element list mein.
    """
    for pattern in _SECTION_PATTERNS:
        sections: list[str] = [
            s.strip() for s in pattern.split(content) if s.strip()
        ]
        if len(sections) > 1:
            logger.debug(
                "StructureAwareChunker: pattern matched — %d sections",
                len(sections),
            )
            return sections

    logger.debug("StructureAwareChunker: no boundary found — single chunk")
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

    strategy_name: str = "structure_aware"

    def __init__(
        self,
        max_chunk_size: int = _MAX_CHUNK_CHARS,
        chunk_overlap: int = _CHUNK_OVERLAP,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        # Stage 2 fallback splitter — sentence-boundary aware separators.
        # ".\n" exclude kiya — aviation regulatory text mein section numbers
        # aur list items pe yeh pattern aata hai jo sentence endings nahi hain.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ". ", "! ", "? ", "\n", " ", ""],
        )

    def _chunk_document(
        self,
        doc: dict[str, Any],
    ) -> list[ChunkedDocument]:
        content: str = doc.get("content", "")
        source: str = doc.get("source", "")

        # Stage 1 — Section boundary detection
        sections: list[str] = _split_on_boundaries(content)

        # Stage 2 + Stage 3 — Size control + header injection
        final_texts: list[str] = []

        for section in sections:
            # Header pehle extract karo — split karne se pehle
            # taaki sub-chunks mein bhi parent section ka context rahe
            header: str = _extract_section_header(section)

            if len(section) > self.max_chunk_size:
                # Oversized section — recursive split, phir har sub-chunk mein header inject
                sub_chunks: list[str] = self._splitter.split_text(section)
                for sub in sub_chunks:
                    final_texts.append(_inject_header(sub, header, source))
            else:
                # Size OK — directly header inject karo
                final_texts.append(_inject_header(section, header, source))

        return self._build_chunks(final_texts, doc, self.strategy_name)