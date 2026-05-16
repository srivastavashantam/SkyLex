"""
processing/strategies/hybrid_chunker.py

Strategy 5: Hybrid Chunker — Hierarchical + Recursive fallback.

Two-stage pipeline:
  Stage 1 — HierarchicalChunker se section boundaries detect karo
  Stage 2 — Oversized sections ko RecursiveCharacterTextSplitter se further split karo

Structure-aware chunking + size guarantee — best of both worlds.

Pros : Regulatory structure respect karta hai + size-controlled chunks
Cons : HierarchicalChunker ki boundary detection limitations inherit karta hai
Use  : Default production strategy — structured aur unstructured dono handle karta hai
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processing.base_chunker import BaseChunker, ChunkedDocument
from processing.strategies.hierarchical_chunker import HierarchicalChunker
from monitoring.logger import get_logger

# Processing pipeline flows, system metrics diagnostics aur dynamic fallback triggers real-time trace aur monitor karne ke liye standard logger module setup configure kiya.
logger = get_logger(__name__)


# ── Hybrid Chunking Strategy Implementation ───────────────────────────────────


class HybridChunker(BaseChunker):
    """
    Two-stage chunker: HierarchicalChunker + RecursiveCharacterTextSplitter fallback.

    Args:
        max_chunk_size : Section size threshold — isse bade sections recursive se split honge
                         (default: 1500)
        chunk_overlap  : Overlap for recursive fallback splits (default: 150)
    """

    # Downstream indexing mechanisms, validation metrics dashboards, aur orchestration framework models RAG pipeline evaluations baseline testing tracking criteria key target registry design name.
    strategy_name: str = "hybrid"

    def __init__(
        self,
        max_chunk_size: int = 1500,
        chunk_overlap: int = 150,
    ) -> None:
        # Document partitions parsing check boundaries framework evaluation k liye parameter limits sizing conditions local property storage variable coordinate value map target assign kiya.
        self.max_chunk_size = max_chunk_size

        # Stage 1 execution workflow orchestration layout ready karne k liye system regulatory architecture matching expression logic internal dependency instance register object construct invoke runner setup.
        self._hierarchical = HierarchicalChunker(max_chunk_size=max_chunk_size)

        # Stage 2 emergency fallback context bounds recovery subroutines optimization layout criteria specifications parameters settings configure instantiate parsing setup model allocation tracking.
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            # Standard single block maximum boundary constraints rule allocation properties pass setup matrix tracking limit.
            chunk_size=max_chunk_size,
            # Continuous streams data consistency alignment buffer sliding window configuration rule assignment tracking scale mapping values layer.
            chunk_overlap=chunk_overlap,
            # Sub-level text formatting parsing tokens symbols validation sequence list delimiters structure scan prioritizer loop layout strings definitions context tracking logic.
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _chunk_document(
        self,
        doc: dict[str, Any],
    ) -> list[ChunkedDocument]:
        # Stage 1 — hierarchical boundaries detect karo

        # Input raw configurations dictionary collections dataset target structural text properties data reading safe validation lookup tracking key fetch context buffer step run.
        # Section structure boundary mapping algorithm processing subroutine execute invoke, regex partitions scanning arrays strings list extraction values record tracking trace pipeline execution indicators.
        sections: list[str] = self._hierarchical._split_on_boundaries(
            doc.get("content", "")
        )

        # Stage 2 — oversized sections ko recursive se further split karo

        # Strict scale threshold limits parameter verification completely passed output layout contents array container memory block initialization placeholder dataset target map allocation register array list.
        final_texts: list[str] = []

        # Structural partitions segments list sequence processing iteration control elements collections scanning traverse workflow loop context trace management execution layout tracking variables code block block path execution.
        for section in sections:
            # Segment content array sizes parameters scale threshold constraints compliance condition boundary dynamic rule processing checking validation evaluate step check control logic layer statement block layout.
            if len(section) > self.max_chunk_size:
                # Target underlying sub-level library processing routine parameters execution string division mapping calculations chunk fragments lists arrays return metrics logic execution run indicator trace model.
                sub_chunks: list[str] = self._recursive_splitter.split_text(
                    section
                )
                # Fragmented small scale data entities components list structure sequences elements directly unified collection destination repository buffer array block context seamless linear merge updates trace step runtime tracking mapping.
                final_texts.extend(sub_chunks)
            else:
                # Within safe capacity constraints ranges items matching elements context directly destination target storage directory container tracking array tracking operations memory address buffer map update element add list append block path trace layer.
                final_texts.append(section)

        # Abstract standard base models interface contract architecture template processing engine function mapping criteria parameters orchestration properties call route dynamic invoke execution pipeline standard dataset items records lists structure format output stream conversion return flow lifecycle complete indicator trace target data.
        return self._build_chunks(final_texts, doc, self.strategy_name)