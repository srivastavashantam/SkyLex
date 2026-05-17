"""
processing/strategies/improved_hybrid_chunker.py

Improved Hybrid Chunker — Larger max_chunk_size + better fallback separators.

Original HybridChunker se improvements:
  1. max_chunk_size 1500 → 2000 —
     DGCA_CAR mein 572 chunks from 6 docs (95/doc) — over-fragmentation.
     2000 chars better balance deta hai structure aur granularity ke beech.
  2. Fallback separators improved —
     ".\n", ". ", "! ", "? " explicitly added —
     Stage 2 recursive fallback sentence boundaries respect karega.
  3. chunk_overlap 150 → 200 (10% of max_chunk_size) —
     Context continuity at section boundaries.

Pros : Structure-aware + size controlled + better sentence boundaries
Cons : Hierarchical boundary detection irregular formatting pe miss kar sakti hai
Use  : FAA_CFR, FAA_AC, DGCA_CAR — structured sources with large docs
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processing.base_chunker import BaseChunker, ChunkedDocument
from processing.strategies.hierarchical_chunker import HierarchicalChunker
from monitoring.logger import get_logger

# Pipeline execution phases, validation traces aur dynamic chunks conversion events logs ko trace karne k liye system logger wrapper allocate kiya.
logger = get_logger(__name__)


# ── Two-Stage Improved Hybrid Chunker Implementation ──────────────────────────


class ImprovedHybridChunker(BaseChunker):
    """
    Improved two-stage chunker: HierarchicalChunker + better recursive fallback.

    Stage 1 — HierarchicalChunker se section boundaries detect karo.
    Stage 2 — Oversized sections ko improved RecursiveCharacterTextSplitter
              se split karo — sentence-boundary aware separators.

    Args:
        max_chunk_size : Section size threshold for recursive fallback (default: 2000)
        chunk_overlap  : Overlap for fallback splits (default: 200)
    """

    # Downstream indexing matrices, dynamic MLflow dashboards, aur RAG optimization evaluation k liye strategy unique identifier metadata register kiya.
    strategy_name: str = "improved_hybrid"

    def __init__(
        self,
        max_chunk_size: int = 2000,
        chunk_overlap: int = 200,
    ) -> None:
        # Source document layouts structural check parameters k liye size thresholds allocation local variables context data properties directory me assign kiye.
        self.max_chunk_size = max_chunk_size

        # Stage 1 core processing architecture build karne k liye baseline structural regulation matching system class runtime instance configure optimize kiya.
        self._hierarchical = HierarchicalChunker(max_chunk_size=max_chunk_size)

        # Stage 2 secondary breakdown optimization matrix parameters conditions mapping initialize setup engine model instance allocate run.
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            # Standard boundary target upper length limits parameter payload flow control direct assignment.
            chunk_size=max_chunk_size,
            # Sliding continuous segments context preservation overlap indices mapping reference check constraint data indicator scale.
            chunk_overlap=chunk_overlap,
            # Tokens formatting rules parsing constraints breakdown list setup: paragraphs priority first, specialized text termination symbols tracking, strict line limits, and backup space words layout control strings collections map trace.
            separators=["\n\n", ".\n", ". ", "! ", "? ", "\n", " ", ""],
        )

    def _chunk_document(
        self,
        doc: dict[str, Any],
    ) -> list[ChunkedDocument]:
        # Stage 1 — hierarchical boundary detection

        # Input data configuration map collections safely reading text properties lookup value safe retrieval parsing pipeline logic trace code checkpoint.
        # Master structural boundary analyzer logic subroutine call execute layout, regex section matching string lists data values sequence extraction returns.
        sections: list[str] = self._hierarchical._split_on_boundaries(
            doc.get("content", "")
        )

        # Stage 2 — improved recursive fallback for oversized sections

        # Clean capacity parameter checked targeted text items output processing lists array container allocation placeholder memory array variable register.
        final_texts: list[str] = []

        # Parsed text segments tracking listing iterative loop sequence scanning elements collections sequential metrics trace workflow loop code layout.
        for section in sections:
            # Current segment structural text character range evaluation constraints conditional boundary limit check rules checking code branch pointer.
            if len(section) > self.max_chunk_size:
                # Target underlying recursive sub-level parsing routine execution mapping calculations split chunks collection list arrays items output.
                sub_chunks: list[str] = self._recursive_splitter.split_text(
                    section
                )
                # Split components fragments blocks variables direct target storage main container array collection buffer data merge execution update trace step.
                final_texts.extend(sub_chunks)
            else:
                # Under safe limits items matching contexts data lists direct destination directory storage mapping sequence update list append tracking pointer line.
                final_texts.append(section)

        # Abstract standard inherited parent model abstract pattern contract method dynamic routing call configurations mapping properties factory dataset generation output standardization list trace return.
        return self._build_chunks(final_texts, doc, self.strategy_name)