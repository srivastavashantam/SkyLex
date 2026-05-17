"""
processing/strategies/improved_semantic_chunker.py

Improved Semantic Chunker — Lower threshold + hard post-processing size cap.

Original SemanticChunker se improvements:
  1. breakpoint_threshold_amount 95.0 → 85.0 —
     Top 15% similarity drops pe split (vs top 5% pehle).
     Zyada splits → chhote, focused chunks → better retrieval precision.
  2. Hard post-processing cap (max_chars=6000) —
     Original mein FAA_CFR pe 55,816 char chunks bane —
     text-embedding-3-small limit (8,191 tokens ≈ 32,764 chars) ke paas.
     6000 char cap safe buffer deta hai + consistent embedding quality.
     Oversized semantic chunks RecursiveCharacterTextSplitter se split hote hain.

Pros : Semantic topic boundaries + size controlled + embedding safe
Cons : Still API-bound — large sources pe latency high rahegi
Use  : SKYBRARY, DGCA_CAR — small sources jahan latency acceptable hai
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from processing.base_chunker import BaseChunker, ChunkedDocument
from monitoring.logger import get_logger
from config.settings import settings

# Multi-stage parsing runtime execution pipeline events tracking aur dynamic exceptions logic monitor karne k liye customized module logging setup analyze kiya.
logger = get_logger(__name__)

# LLM embedding extraction pipelines optimization framework configurations k liye safe token boundaries boundary parameter range setup limits check metrics numbers constant register kiya.
_HARD_CAP_CHARS: int = 6000

# Top-level global template fallback processing module allocation parameters delimiters priorities lookups checklist definition logic runtime setup mapping.
_FALLBACK_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_HARD_CAP_CHARS,
    chunk_overlap=300,  # 5% of hard cap — context preservation
    separators=["\n\n", ".\n", ". ", "! ", "? ", "\n", " ", ""],
)


# ── Two-Stage Improved Semantic Chunker Implementation ────────────────────────


class ImprovedSemanticChunker(BaseChunker):
    """
    Improved LangChain SemanticChunker with lower threshold + hard size cap.

    Pipeline:
        Stage 1 — SemanticChunker (threshold=85.0) → topic-boundary splits
        Stage 2 — Any chunk > 6000 chars → RecursiveCharacterTextSplitter fallback

    Args:
        breakpoint_threshold_type   : Similarity drop detection method
                                      (default: 'percentile')
        breakpoint_threshold_amount : Split threshold — lower = more splits
                                      85.0 → top 15% drops trigger split
                                      (default: 85.0, was 95.0)
        hard_cap_chars              : Maximum chars per chunk after post-processing
                                      (default: 6000)
    """

    # Downstream monitoring pipelines dashboards index comparison variants matrix evaluation analytics k liye explicit pipeline method strategy string keyword marking track kiya.
    strategy_name: str = "improved_semantic"

    def __init__(
        self,
        breakpoint_threshold_type: Literal[
            "percentile",
            "standard_deviation",
            "interquartile",
            "gradient",
        ] = "percentile",
        breakpoint_threshold_amount: float = 85.0,
        hard_cap_chars: int = _HARD_CAP_CHARS,
    ) -> None:
        # Paragraph arrays distance logic processing control configuration parameter selection analysis criteria model save type settings properties tracking local data map.
        self.breakpoint_threshold_type = breakpoint_threshold_type
        # Cosine distance variance calculation metrics checking boundary drop indices values register threshold configuration value control.
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        # Post-processing stage capacity threshold conditions rules layout allocation dynamic parameters limitations metrics save storage variable data pointer rules.
        self.hard_cap_chars = hard_cap_chars

        # Outbound dynamic calculations embedding processor vector constructor engine model instantiation initialization parameters config setup memory route mapping.
        self._embeddings = OpenAIEmbeddings(
            # Central specifications tracking framework settings profile variable lookup options structure reading key identification allocation value parameters model config map.
            model=settings.openai_embedding_model,
            # Data privacy layer protections structural conversion validations validation variables token allocation check annotations configurations override execution run trace.
            openai_api_key=SecretStr(settings.openai_api_key),  # type: ignore[call-arg]
        )
        # Stage 1 active semantic similarity analysis processing base engine infrastructure wrapper component configuration initialize mapping invoke context pipeline.
        self._splitter = LangChainSemanticChunker(
            # Vectors generation mapping target context metrics injection parameter code workflow setup variables trace reference tracking components call layer logic models.
            embeddings=self._embeddings,
            # Vector distance lookup selector methodology criteria verification properties checking rules matrix route setup rules dynamic branching tracking options variables.
            breakpoint_threshold_type=breakpoint_threshold_type,
            # Cutoff statistical drop limit metrics criteria register configuration limit boundary constraints checking condition evaluations tracking parameter values database run.
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )
        # Stage 2 dynamic overflow mitigation infrastructure execution workflow configurations settings specifications structural fallback setup allocation properties parameter array context.
        self._fallback = RecursiveCharacterTextSplitter(
            # Explicit single partition maximum capacity safety bounds parameters limits pass assignment criteria matrix processing control check value tracking.
            chunk_size=hard_cap_chars,
            # Sliding sliding overlap memory windows retention data uniformity synchronization rule configurations pass setup parameters data matrix indicator factor.
            chunk_overlap=300,
            # Sub-segment tokens characters scanning priority system list structure symbols configuration check priority parameters arrays layout definitions strings logic check formatting layers.
            separators=["\n\n", ".\n", ". ", "! ", "? ", "\n", " ", ""],
        )

    def _apply_hard_cap(self, texts: list[str]) -> list[str]:
        """
        Post-processing hard cap — oversized semantic chunks ko
        RecursiveCharacterTextSplitter se further split karo.
        text-embedding-3-small ke token limit ke andar rehna guaranteed.
        """
        # Verified capacity checked output datasets mapping array parameters clean isolated placeholder target list allocations variables memory context setup loop framework.
        final_texts: list[str] = []
        # Overflow elements index monitoring counters parameter evaluations conditions track calculations tracking limits variables increments runtime check loop context trace.
        oversized_count: int = 0

        # Structural partitions segments tracking listing iteration loop scanning data elements collection matrix sequential blocks scan run pipeline trace logic.
        for text in texts:
            # Segment character capacity length thresholds boundary checks evaluate dynamic condition matching properties rules validation control branching logic parameter execute check.
            if len(text) > self.hard_cap_chars:
                # Accumulator indicator metric statistics update calculations counters value update track processing loop statement runtime data state verification model.
                oversized_count += 1
                # Secondary mechanical breakdown algorithm processing parameters execution text division sub-chunks lists array configurations processing return data.
                sub_chunks: list[str] = self._fallback.split_text(text)
                # Split segments elements array values directly unified target tracking collection buffer array data blocks merge execution processing update trace step run.
                final_texts.extend(sub_chunks)
            else:
                # Under safe limits items matching arrays objects direct destination allocation storage sequence array container elements register tracking update list path trace execution.
                final_texts.append(text)

        # Ingest evaluations trace check alert notifications system level debugger parameters conditions checking rule log visibility console update event code tracking block.
        if oversized_count > 0:
            # Telemetry matrix log update trigger console display formatting metrics variables reporting pipeline instrumentation tracking info messaging run stream layout check trace.
            logger.debug(
                "ImprovedSemanticChunker: %d oversized chunks split via fallback",
                oversized_count,
            )

        # Yield completely clean isolated bounded textual components collection array mapping variables records trace database back to caller framework context pipeline.
        return final_texts

    def _chunk_document(
        self,
        doc: dict[str, Any],
    ) -> list[ChunkedDocument]:
        # Stage 1 — semantic topic boundary detection

        # Dynamic mapping components payload data lookup dictionary parameters read safe extraction tracking keys query memory reading data validation step value pipeline pointer config.
        # Computational vector cosine similarity calculations tracking sequence execution loop splits partitions strings extraction values arrays returning metadata trace run.
        texts: list[str] = self._splitter.split_text(doc.get("content", ""))

        # Stage 2 — hard cap post-processing

        # Outbound validation parsing array execution filtering subroutine method passing elements array parameter returns update variables mapping sequence track process indicator complete.
        texts = self._apply_hard_cap(texts)

        # Abstract standard inherited parent model processing pattern contract method dynamic routing call configurations mapping properties factory dataset generation standardized components schema lists return.
        return self._build_chunks(texts, doc, self.strategy_name)