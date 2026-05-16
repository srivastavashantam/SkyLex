"""
processing/strategies/agentic_chunker.py

Strategy 3: Agentic Proposition Chunker — LLM-based atomic extraction.

GPT-4o-mini use karke har passage se self-contained propositions extract karta hai.
Har chunk independently retrievable hota hai — no surrounding context needed.

Pros : Highest retrieval precision, context-independent chunks
Cons : OpenAI API calls per passage — highest cost aur latency
Use  : High-stakes retrieval — jahan precision > cost
"""

from __future__ import annotations

import json
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from processing.base_chunker import BaseChunker, ChunkedDocument
from monitoring.logger import get_logger
from config.settings import settings

# System logs maintenance, tracking telemetry metrics aur real-time processing operations monitor karne ke liye standard logger instance set kiya.
logger = get_logger(__name__)


# ── Agentic Chunking Strategy Implementation ──────────────────────────────────


class AgenticChunker(BaseChunker):
    """
    GPT-4o-mini based proposition extraction chunker.

    Large documents ko pehle passages mein todta hai (RecursiveCharacterTextSplitter),
    phir har passage pe GPT-4o-mini call karke atomic propositions extract karta hai.

    Args:
        max_tokens_per_call : Max tokens for each LLM response (default: 2000)
        passage_size        : Max chars per passage before LLM call (default: 3000)
        passage_overlap     : Overlap between passages (default: 100)
    """

    # Downstream evaluation layers, performance reporting dashboards, aur multi-model orchestration benchmark testing k liye unique framework identification tags registry name setup kiya.
    strategy_name: str = "agentic"

    # Core LLM prompt behavior orchestration instructions directive code layer setup: output payload control structures rule sets target configuration data schema boundary define mapping guidelines statement.
    _SYSTEM_PROMPT: str = (
        "You are an expert at extracting self-contained propositions from "
        "aviation regulatory documents. Given a passage, extract a list of "
        "atomic, self-contained statements. Each statement must be independently "
        "understandable without any surrounding context. "
        "Return ONLY a JSON array of strings. No explanation, no markdown."
    )

    def __init__(
        self,
        max_tokens_per_call: int = 2000,
        passage_size: int = 3000,
        passage_overlap: int = 100,
    ) -> None:
        # Secure api calls networks connections verify credentials mapping authorization token parameter configurations state setup client engine handler allocate run.
        self._client = OpenAI(api_key=settings.openai_api_key)
        # LLM system completion endpoints processing requests window context token limits parameters boundary condition parameter tracker save variables logic property value trace.
        self.max_tokens_per_call = max_tokens_per_call
        # Token window overflow crash aur payload sizes save handle rules structure check context limits safe range scale map recursive text pre-splitter pipeline config setup run.
        self._presplitter = RecursiveCharacterTextSplitter(
            # Large scale document data processing steps optimize parameters sizing boundaries limits conditions rules target maps allocation sequence.
            chunk_size=passage_size,
            # Continuous layout segments data retention overlaps parameter mappings tracker sequence check limits rule assignment setup layout tracking.
            chunk_overlap=passage_overlap,
            # System priority token formatting hierarchy rules string separation characters lookups evaluation loops index validation array collection setup logic text formatters.
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _extract_propositions(self, passage: str) -> list[str]:
        """
        Single passage se atomic propositions extract karo via GPT-4o-mini.
        LLM failure pe graceful fallback — original passage return karo.
        """
        try:
            # Global properties engine central management module properties reference model setup, parameters control pipeline runtime chat completions client API invocation call trigger code layer mapping.
            response = self._client.chat.completions.create(
                # System orchestration variable path dynamically mapped config template dynamic variable target string parameters mapping identifier token structure lookup logic trace state.
                model=settings.openai_model,
                # Outbound system resource generation validation tokens allocations lengths numerical constraints assignments mapping pipeline tracker boundary threshold configuration context block data run.
                max_tokens=self.max_tokens_per_call,
                # Deterministic outputs extraction validation control parameters rule context configuration strict zero-variance settings structure mapping trace.
                temperature=0.0,
                # Multi-level prompt context injection payload array parameters structure routing dictionary objects list properties mapping layer trace rules definitions flow stream process target blocks.
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": passage},
                ],
            )
            # Response validation structures schema lookup layers checking parameters logic check nested fallback tracking sequence evaluation checks execution result variable extraction handler block layout.
            raw: str = response.choices[0].message.content or "[]"
            # Unstructured text formatting strings representation object parse dynamic JSON deserialization validator module invoke extraction processing properties variable structure array setup trace map data blocks.
            propositions: Any = json.loads(raw)
            # Schema response data type integrity checks control check execution condition map rules evaluation validate trace flow structure block layout handling loop parameter configurations mapping runtime indicator type.
            if isinstance(propositions, list):
                # Array strings format validation mapping filter clean processing strings layout list comprehensive loop checks evaluation execution values records return context logic tracking execution parameters framework.
                return [str(p) for p in propositions if str(p).strip()]
            # Backup safety flow routing control structural mechanism configuration data type divergence errors verification layer default fallback return array state block trace execution tracker setup.
            return [passage]
        except Exception as e:
            # Network drops, schema validation checks failed ya structural exceptions interception reporting tracking alert indicators trigger logs warning tracking sequence system trace logging execution.
            logger.warning(
                "AgenticChunker LLM call failed: %s — falling back to passage",
                e,
            )
            # Graceful degraded quality assurance fallback pipeline state parameter tracking execution flow failure control array string wrap logic operations structure mapping safety blocks recovery run.
            return [passage]

    def _chunk_document(
        self,
        doc: dict[str, Any],
    ) -> list[ChunkedDocument]:
        # Input raw dynamic data container entity content retrieval schema lookups processing read operations safe parameters defaults validation lookup pipeline tracing context value map step.
        content: str = doc.get("content", "")
        # Internal formatting utilities algorithms layer execution step invoke run large documents content string multi-passages segment chunks matrix transformation split lists container data extraction arrays execution tracking.
        passages: list[str] = self._presplitter.split_text(content)
        # Atomic parsed text sequences items storage container allocation tracker flat array aggregation layer processing initialization matrix targets parameters variable trace.
        all_propositions: list[str] = []
        # Multi-segment iteration tracking dynamic block sequences collection traversal control execution loop state tracking process scanning operations.
        for passage in passages:
            # Extraction automation subroutines logic calls parameters pass evaluation responses returns arrays target flat container records structural aggregation updates sequence operations merge step.
            all_propositions.extend(self._extract_propositions(passage))
        # Abstract base pattern validation template structural rules engine mapping parameters properties orchestration call execution target components pipeline standard dataclass lists schema model generation return context stream flow.
        return self._build_chunks(
            all_propositions, doc, self.strategy_name
        )