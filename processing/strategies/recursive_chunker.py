"""
processing/strategies/recursive_chunker.py

Strategy 1: Recursive Character Splitter — Baseline chunking strategy.

LangChain RecursiveCharacterTextSplitter use karta hai jo separator hierarchy
follow karta hai: paragraph → line → sentence → word → character.
Har level pe split karne ki koshish hoti hai jab tak chunk_size enforce na ho jaye.

Pros : Fast, deterministic, zero API cost, size-controlled
Cons : Document structure (§ sections, headings) ignore karta hai
Use  : Baseline — doosri strategies isse compare karke evaluate hongi
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processing.base_chunker import BaseChunker, ChunkedDocument
from monitoring.logger import get_logger

# Current class aur structural algorithm processing run lifecycle events ko systematically trace aur debug karne k liye dynamic logging instance configure kiya.
logger = get_logger(__name__)


# ── Recursive Chunking Strategy Implementation ────────────────────────────────


class RecursiveChunker(BaseChunker):
    """
    LangChain RecursiveCharacterTextSplitter wrapper.

    Args:
        chunk_size    : Maximum characters per chunk (default: 1000)
        chunk_overlap : Shared characters between adjacent chunks (default: 150)
    """

    # Downstream processing evaluation engines, RAG pipeline orchestrator layer, A/B comparison tests aur data metrics evaluation k liye identifier name track register kiya.
    strategy_name: str = "recursive"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        # Incoming raw documentation streams ko split karte waqt har single element text chunk block ki maximum character scale range capacity property initialize ki.
        self.chunk_size = chunk_size
        # Continuity data gaps protection, chunk text split boundaries alignment shifts preserve karne k liye common sequence sliding window scope configure kiya.
        self.chunk_overlap = chunk_overlap

        # Underlying framework text splitter library parsing processor object compute execution initialization runtime handle structure define instantiate kiya.
        self._splitter = RecursiveCharacterTextSplitter(
            # Configured length size conditions limits target parameter pass mapping criteria value binding control execution sequence block setup.
            chunk_size=chunk_size,
            # Sliding memory parameters data collection matrix reference parameters setup rule integration structure map model assignment trace layout execution.
            chunk_overlap=chunk_overlap,
            # Hierarchical scanning tracking symbols execution lookup loop sequence fallback rules setup: sub-paragraphs, clear lines, strict sentences separation rules layout strings.
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _chunk_document(
        self,
        doc: dict[str, Any],
    ) -> list[ChunkedDocument]:
        # Dictionary dataset entities block configuration properties layout context metadata path inline reading operation process memory fetch key attributes execute wrapper logic setup.
        # String stream array parsing algorithm framework processing split engine pass mapping run text blocks collection tracking segmentation array list calculation complete execution indicator trace.
        texts: list[str] = self._splitter.split_text(doc.get("content", ""))

        # Base pattern abstract architecture template methods layout block processing pipeline call route run execution handler invoke.
        # Unstructured text strings sequence array items coordinate transform map targeted unified custom schema framework output data models direct list conversion process return flow.
        return self._build_chunks(texts, doc, self.strategy_name)