# =============================================================================
# SkyLex — Full Ingestion Pipeline Runner
# Runs all 5 data source ingesters in sequence.
# Safe to re-run — hash-based change detection skips unchanged documents.
# Usage: python scripts/run_ingestion.py
#        python scripts/run_ingestion.py --sources ecfr faa_ad
#        python scripts/run_ingestion.py --dry-run
# =============================================================================

import argparse
import sys
import time
from pathlib import Path

# Add project root to path — allows running from any directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.logger import setup_logging, get_logger
from monitoring.mlflow_tracker import initialize_mlflow

logger = get_logger(__name__)

# =============================================================================
# Ingestion Configuration
# All sources with their ingester class and run parameters
# =============================================================================
INGESTION_CONFIG = {

    # ── eCFR — FAA 14 CFR Title 14 (All 226 Parts) ───────────────────────────
    "ecfr": {
        "description": "FAA 14 CFR Title 14 — All 226 Parts (Aeronautics and Space)",
        "class": "ingestion.sources.ecfr_ingester.ECFRIngester",
        "kwargs": {
            "parts": None,  # None = fetch ALL 226 parts
        },
    },

    # ── FAA ADs — Air India Fleet Relevant Aircraft Types ─────────────────────
    "faa_ad": {
        "description": "FAA Airworthiness Directives — 7 Air India fleet aircraft types (2024-present)",
        "class": "ingestion.sources.faa_ad_ingester.FAAAADIngester",
        "kwargs": {
            "aircraft_keys": None,  # None = all 7 aircraft types
            "from_date": "2024-01-01",
            "download_pdfs": False,  # Metadata only — PDFs optional
        },
    },

    # ── DGCA CARs — 6 Verified Indian Regulatory Documents ───────────────────
    "dgca": {
        "description": "DGCA Civil Aviation Requirements — 6 verified CARs (Air India relevant)",
        "class": "ingestion.sources.dgca_ingester.DGCAIngester",
        "kwargs": {
            "car_keys": None,  # None = all 6 verified CARs
        },
    },

    # ── FAA ACs — 20 Curated Advisory Circulars ───────────────────────────────
    "faa_ac": {
        "description": "FAA Advisory Circulars — 20 curated ACs (Air India operations relevant)",
        "class": "ingestion.sources.faa_ac_ingester.FAAAACIngester",
        "kwargs": {
            "ac_keys": None,  # None = all 20 ACs
        },
    },

    # ── SKYbrary — 20 Aviation Safety Articles ────────────────────────────────
    "skybrary": {
        "description": "SKYbrary Aviation Safety — 20 curated articles (EUROCONTROL/ICAO/FSF)",
        "class": "ingestion.sources.skybrary_ingester.SKYbraryIngester",
        "kwargs": {
            "article_keys": None,  # None = all 20 articles
        },
    },
}


def _load_ingester_class(class_path: str):
    """
    Dynamically load ingester class from dot-separated module path.

    Args:
        class_path: e.g., "ingestion.sources.ecfr_ingester.ECFRIngester"

    Returns:
        Ingester class
    """
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def run_single_source(
    source_key: str,
    config: dict,
    dry_run: bool = False,
) -> dict:
    """
    Run ingestion for a single data source.

    Args:
        source_key: Source identifier e.g. "ecfr"
        config:     Source configuration dict
        dry_run:    If True, only show what would be run — no actual ingestion

    Returns:
        Dict with source stats
    """
    logger.info("=" * 60)
    logger.info(f"🚀 Source: {source_key.upper()}")
    logger.info(f"   {config['description']}")
    logger.info("=" * 60)

    if dry_run:
        logger.info(f"   [DRY RUN] Would run: {config['class']}")
        logger.info(f"   [DRY RUN] With kwargs: {config['kwargs']}")
        return {"source": source_key, "status": "dry_run", "documents": 0}

    start_time = time.time()

    try:
        # Dynamically load and instantiate ingester
        IngesterClass = _load_ingester_class(config["class"])
        ingester = IngesterClass(**config["kwargs"])

        # Run ingestion
        documents = ingester.run()

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            f"✅ {source_key.upper()} complete | "
            f"Documents: {len(documents)} | "
            f"Time: {elapsed}s"
        )

        return {
            "source": source_key,
            "status": "success",
            "documents": len(documents),
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        logger.error(
            f"❌ {source_key.upper()} FAILED | "
            f"Error: {e} | "
            f"Time: {elapsed}s"
        )
        return {
            "source": source_key,
            "status": "failed",
            "documents": 0,
            "elapsed_seconds": elapsed,
            "error": str(e),
        }


def run_full_ingestion(
    sources: list[str],
    dry_run: bool = False,
) -> None:
    """
    Run full ingestion pipeline for specified sources.

    Args:
        sources: List of source keys to run
        dry_run: If True, only show what would run
    """
    setup_logging()
    initialize_mlflow()

    total_start = time.time()
    all_stats = []

    logger.info("=" * 60)
    logger.info("🛫 SkyLex — Full Ingestion Pipeline Starting")
    logger.info(f"   Sources: {sources}")
    logger.info(f"   Dry run: {dry_run}")
    logger.info("=" * 60)

    for source_key in sources:
        if source_key not in INGESTION_CONFIG:
            logger.warning(f"⚠️ Unknown source: {source_key} — skipping")
            continue

        config = INGESTION_CONFIG[source_key]
        stats = run_single_source(source_key, config, dry_run)
        all_stats.append(stats)

        # Small delay between sources — be respectful to APIs
        if not dry_run and source_key != sources[-1]:
            logger.info("⏳ Waiting 3 seconds before next source...")
            time.sleep(3)

    # Final summary
    total_elapsed = round(time.time() - total_start, 2)
    total_docs = sum(s["documents"] for s in all_stats)
    failed = [s for s in all_stats if s["status"] == "failed"]

    logger.info("=" * 60)
    logger.info("🏁 SkyLex — Full Ingestion Pipeline Complete")
    logger.info(f"   Total documents ingested: {total_docs}")
    logger.info(f"   Total time: {total_elapsed}s")
    logger.info(f"   Sources run: {len(all_stats)}")
    logger.info(f"   Failed: {len(failed)}")
    logger.info("=" * 60)

    # Per-source summary
    logger.info("📊 Per-Source Summary:")
    for stats in all_stats:
        status_icon = "✅" if stats["status"] == "success" else "❌" if stats["status"] == "failed" else "🔍"
        logger.info(
            f"   {status_icon} {stats['source'].upper():12} | "
            f"Docs: {stats.get('documents', 0):4} | "
            f"Time: {stats.get('elapsed_seconds', 0):6}s"
        )

    if failed:
        logger.warning(f"⚠️ Failed sources: {[s['source'] for s in failed]}")
        logger.warning("   Re-run with --sources to retry failed sources only")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SkyLex Full Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ingestion.py                          # Run all sources
  python scripts/run_ingestion.py --sources ecfr faa_ad   # Run specific sources
  python scripts/run_ingestion.py --dry-run                # Preview only
  python scripts/run_ingestion.py --sources dgca skybrary  # Re-run specific sources

Available sources:
  ecfr      — FAA 14 CFR Title 14 (226 parts)
  faa_ad    — FAA Airworthiness Directives (7 aircraft types)
  dgca      — DGCA Civil Aviation Requirements (6 verified CARs)
  faa_ac    — FAA Advisory Circulars (20 curated ACs)
  skybrary  — SKYbrary Aviation Safety (20 articles)
        """,
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(INGESTION_CONFIG.keys()),
        default=list(INGESTION_CONFIG.keys()),
        help="Sources to ingest (default: all)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be run without actually ingesting",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_ingestion(
        sources=args.sources,
        dry_run=args.dry_run,
    )