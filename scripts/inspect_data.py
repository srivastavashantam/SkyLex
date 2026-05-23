"""
scripts/inspect_data.py

Data inventory script for SkyLex raw data.
Scans data/raw/ and produces a source-wise breakdown of:
- File types present
- Document counts (from bulk JSONs only)
- Content field stats (length distribution)
- Exclusion summary (hash registries, meta JSONs, non-data files)

Usage:
    python scripts/inspect_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# SkyLex centralized settings se directory paths aur configurations load ho rahi hain
from config.settings import settings
# Monitoring module ka use karke report ko console aur file dono jagah log kiya jayega
from monitoring.logger import get_logger, setup_logging

# Logging infrastructure initialization for systematic reporting
setup_logging()
logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Phase 1 Ingestion ka final destination jahan saara raw data curated folders mein rakha hai
RAW_DATA_DIR: Path = settings.data_raw_dir

# 📌 SYSTEM EXCLUSION REGISTRY:
# Ye files operational metadata hain (hashes/sidecars). Inme regulatory content nahi hota,
# isliye inke character counts ko audit metrics se bahar rakhna analytical accuracy ke liye zaroori hai.
EXCLUDED_JSON_PATTERNS: tuple[str, ...] = (
    "hash_registry.json",  # Incremental ingestion ki state file
    "_meta.json",          # Individual PDF metadata reference files
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_excluded_json(path: Path) -> bool:
    """
    Check karta hai ki di gayi JSON file actual document data hai ya pipeline maintenance file.
    Registry files ko skip karke hum sirf 'text-rich' data par focus karte hain.
    """
    return path.name == "hash_registry.json" or path.name.endswith("_meta.json")


def is_bulk_data_json(path: Path) -> bool:
    """
    Primary Data Identifier:
    Vaildate karta hai ki file .json extension ki hai aur system exclusion list mein nahi hai.
    Ye ensures karta hai ki hum sirf Ingester ke main output files ko hi process karein.
    """
    return path.suffix == ".json" and not is_excluded_json(path)


def load_bulk_json(path: Path) -> dict[str, Any] | None:
    """
    Safe JSON Loader with Schema Validation:
    File load karne ke baad check karta hai ki kya usme SkyLex standard 'documents' key hai.
    Agar schema match nahi hota (non-ingester JSON), toh use safely ignore karta hai.
    """
    try:
        with path.open(encoding="utf-8") as f:
            data: Any = json.load(f)
        # Structural check to ensure compatibility with Document dataclass format
        if isinstance(data, dict) and "documents" in data:
            return data
        return None
    except (json.JSONDecodeError, OSError) as e:
        # File corruption ya I/O errors ko warning level par log karta hai bina script crash kiye
        logger.warning("Could not load %s: %s", path, e)
        return None


def compute_content_stats(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """
    🔬 QUANTITATIVE DATA AUDIT:
    Ye function character-level analytics calculate karta hai. 
    Phase 2 mein 'chunk_size' decide karne ke liye ye stats (min/max/avg chars) 
    sabse bada factor hain, taaki semantics lose na hon.
    """
    # Raw text distribution collect kar rahe hain statistical analysis ke liye
    lengths: list[int] = [
        len(doc.get("content", ""))
        for doc in documents
        if isinstance(doc, dict)
    ]

    # Edge case: Agar source folder mein koi valid document data hi na ho
    if not lengths:
        return {
            "total_docs": 0,
            "min_chars": 0,
            "max_chars": 0,
            "avg_chars": 0,
            "total_chars": 0,
            "empty_content_count": 0,
        }

    return {
        "total_docs": len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "avg_chars": round(sum(lengths) / len(lengths)), # Window size tuning ke liye benchmark
        "total_chars": sum(lengths),
        "empty_content_count": sum(1 for length in lengths if length == 0), # Potential ingestion bugs flag karta hai
    }


# ── Core ─────────────────────────────────────────────────────────────────────

def inspect_source(source_dir: Path) -> dict[str, Any]:
    """
    Recursive Source Auditor:
    'rglob' pattern ka use karke ye DGCA jaise nested folder structures ko walk karta hai.
    Physical files (PDFs) aur logic files (JSONs) ka source-level breakdown provide karta hai.
    """
    # Directory tree walk: Saari files ko aggregate karna list mein
    all_files: list[Path] = [f for f in source_dir.rglob("*") if f.is_file()]

    # Disk inventory: Extensions ka breakdown (e.g., kitni PDFs vs kitni JSONs)
    ext_counts: dict[str, int] = {}
    for f in all_files:
        ext = f.suffix.lower() if f.suffix else "(no extension)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # Separation of concerns: Files ko unke role ke hisaab se filter karna
    excluded_jsons: list[Path] = [
        f for f in all_files
        if f.suffix == ".json" and is_excluded_json(f)
    ]
    bulk_jsons: list[Path] = [
        f for f in all_files
        if is_bulk_data_json(f)
    ]
    pdf_files: list[Path] = [f for f in all_files if f.suffix == ".pdf"]

    # In-memory aggregation: Saare bulk JSONs ko scan karke overall document stats nikalna
    all_documents: list[dict[str, Any]] = []
    bulk_json_details: list[dict[str, Any]] = []

    for bulk_json in sorted(bulk_jsons):
        data = load_bulk_json(bulk_json)
        if data is None:
            # Descriptive failure log for non-compliant JSON files
            bulk_json_details.append({
                "file": bulk_json.name,
                "status": "skipped — no 'documents' key",
                "document_count": 0,
                "size_kb": round(bulk_json.stat().st_size / 1024, 1),
            })
            continue

        docs: list[dict[str, Any]] = data.get("documents", [])
        bulk_json_details.append({
            "file": bulk_json.name,
            "status": "loaded",
            "document_count": data.get("document_count", len(docs)),
            "size_kb": round(bulk_json.stat().st_size / 1024, 1),
        })
        all_documents.extend(docs)

    return {
        "source": source_dir.name,
        "total_files": len(all_files),
        "file_types": ext_counts,
        "pdf_count": len(pdf_files),
        "excluded_json_count": len(excluded_jsons),
        "bulk_json_count": len(bulk_jsons),
        "bulk_jsons": bulk_json_details,
        "content_stats": compute_content_stats(all_documents),
    }


def run_inspection() -> None:
    """
    Master Orchestration Engine:
    'data/raw/' ke har folder ko ek independent data source maan kar audit shuru karta hai.
    Ye dashboard view deta hai ki total kitna corpus RAG pipeline ke liye ready ho chuka hai.
    """

    if not RAW_DATA_DIR.exists():
        logger.error("RAW_DATA_DIR does not exist: %s", RAW_DATA_DIR)
        return

    # Iterating through source folders (FAA, DGCA, eCFR, SKYbrary, etc.)
    source_dirs: list[Path] = sorted(
        d for d in RAW_DATA_DIR.iterdir() if d.is_dir()
    )

    if not source_dirs:
        logger.warning("No source directories found in %s", RAW_DATA_DIR)
        return

    # Report Header UI for terminal visibility
    logger.info("=" * 70)
    logger.info("SKYLEX DATA INVENTORY — POST-INGESTION AUDIT")
    logger.info("Scanning: %s", RAW_DATA_DIR)
    logger.info("Source directories found: %d", len(source_dirs))
    logger.info("=" * 70)

    # Project-wide metrics accumulators
    grand_total_docs: int = 0
    grand_total_chars: int = 0

    # Source-wise breakdown execution
    for source_dir in source_dirs:
        result = inspect_source(source_dir)
        stats = result["content_stats"]

        # Formatted Source Summary
        logger.info("")
        logger.info("── SOURCE: %s ──", result["source"].upper())
        logger.info("  Total files on disk  : %d", result["total_files"])
        logger.info("  File types           : %s", result["file_types"])
        logger.info("  PDF files            : %d", result["pdf_count"])
        logger.info("  Excluded JSONs       : %d  (hash_registry + _meta)", result["excluded_json_count"])
        logger.info("  Bulk data JSONs      : %d", result["bulk_json_count"])

        # Individual bulk file breakdown inside the source
        for detail in result["bulk_jsons"]:
            logger.info(
                "    ↳ %-50s | docs: %4d | %s KB | %s",
                detail["file"],
                detail["document_count"],
                detail["size_kb"],
                detail["status"],
            )

        # Content Analytics Reporting
        logger.info("  Documents loaded     : %d", stats["total_docs"])
        logger.info(
            "  Content (chars)      : min=%d | max=%d | avg=%d | total=%d",
            stats["min_chars"],
            stats["max_chars"],
            stats["avg_chars"],
            stats["total_chars"],
        )

        # 🚩 DATA QUALITY WARNING: Khali documents detect hone par developer ko immediate alert dena
        if stats["empty_content_count"] > 0:
            logger.warning(
                "  ⚠ ALERT: Empty content docs detected: %d", stats["empty_content_count"]
            )

        grand_total_docs += stats["total_docs"]
        grand_total_chars += stats["total_chars"]

    # Final Dashboard Summary: Tokens aur Costs estimate karne mein help karta hai
    logger.info("")
    logger.info("=" * 70)
    logger.info("SKYLEX FINAL AUDIT SUMMARY")
    logger.info("GRAND TOTAL — Documents  : %d", grand_total_docs)
    logger.info("GRAND TOTAL — Characters : %d", grand_total_chars)
    logger.info("=" * 70)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Script runtime execution shuru
    run_inspection()