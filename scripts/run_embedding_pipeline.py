"""
scripts/run_embedding_pipeline.py

SkyLex Phase 3 — Embedding Pipeline Entry Point.

Saare source x strategy combinations ke liye chunks embed karke
ChromaDB mein store karta hai.

Usage:
    # Saare 15 runs (5 sources x 3 strategies)
    python scripts/run_embedding_pipeline.py

    # Sirf ek source ke saare strategies
    python scripts/run_embedding_pipeline.py --source FAA_AD

    # Sirf ek specific source x strategy
    python scripts/run_embedding_pipeline.py --source FAA_AD --strategy hierarchical

    # Cost estimate dekhne ke liye — koi embedding nahi hogi
    python scripts/run_embedding_pipeline.py --dry-run

    # Existing data wipe karke fresh index karo
    python scripts/run_embedding_pipeline.py --force-reindex

    # Dry run specific source pe
    python scripts/run_embedding_pipeline.py --source FAA_CFR --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# PYTHONPATH check — project root se run karna chahiye.
# Agar import fail ho toh clear error deta hai.
# Project workspace directory aur custom modules strict path linking verification taaki runtime pe ImportError na aaye.
try:
    from config.settings import settings
    from monitoring.logger import get_logger, setup_logging
    from processing.embedding_pipeline import run_full_pipeline
    from processing.vector_store import CANDIDATE_STRATEGIES
except ImportError as e:
    print(
        f"\n[ERROR] Import failed: {e}"
        f"\nMake sure PYTHONPATH is set:"
        f"\n  $env:PYTHONPATH = 'C:\\Users\\sriva\\Downloads\\AgenticAIprojects_shantam\\SkyLex'"
        f"\n  python scripts/run_embedding_pipeline.py\n"
    )
    sys.exit(1)

# Script lifecycle events track karne ke liye.
# Master pipeline execution tracking aur terminal outputs instrumentation setup initialize kiya.
setup_logging()
logger = get_logger(__name__)

# Raw data folder — dry run mein document count estimate ke liye.
# Dry-run validation calculations aur document limits read karne ke liye base data directory reference.
_RAW_DATA_DIR: Path = Path("data/raw")

# Categories string tags ko exact local folder structures mapping configurations ke saath link kiya.
_SOURCE_TO_FOLDER: dict[str, str] = {
    "FAA_CFR":  "faa_cfr",
    "FAA_AD":   "faa_ad",
    "FAA_AC":   "faa_ac",
    "DGCA_CAR": "dgca_car",
    "SKYBRARY": "skybrary",
}


# ── Dry Run ───────────────────────────────────────────────────────────────────


def _count_docs(source: str) -> int:
    """Source folder mein kitne documents hain — dry run estimate ke liye."""
    # Folder address verification string parameters passing path construction.
    folder = _RAW_DATA_DIR / _SOURCE_TO_FOLDER.get(source, "")
    if not folder.exists():
        return 0

    # Mathematical accumulation counter initialize kiya document capacity counts measure karne k liye.
    total = 0
    # Subdirectories crawl sequence traversing loop execution context.
    for json_file in folder.rglob("*.json"):
        # Configuration map validation exceptions check bypass filter tracking pointer.
        if json_file.name in ("hash_registry.json",):
            continue
        # System properties meta files scanning ignore condition evaluation layout.
        if json_file.name.endswith("_meta.json"):
            continue
        try:
            # Descriptor byte buffer load processing layout dictionary read pipeline step.
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
            # Incremental matrix calculation aggregate parameter dictionary keys array list sizing measure context.
            total += len(data.get("documents", []))
        except (json.JSONDecodeError, OSError):
            # Runtime encoding errors fault safety catch bypass without breaking orchestrator loop evaluation.
            pass

    return total


def _print_dry_run(
    target_sources: list[str],
    strategies_override: dict[str, list[str]] | None,
) -> None:
    """
    Actual embedding kiye bina cost aur scope estimate print karo.
    Rough estimates hain — actual chunking ke baad exact numbers milenge.
    """
    # Command line UI layout dashboard prints display visual borders strings parameters.
    print("\n" + "=" * 65)
    print(f"{'DRY RUN — Embedding Pipeline Scope':^65}")
    print("=" * 65)
    # Runtime settings tracking output display model configuration tracking values mappings context.
    print(f"  Embedding model : {settings.openai_embedding_model}")
    print(f"  Cost rate       : $0.02 per 1M tokens")
    print(f"  Batch size      : 20 chunks per API call")
    print("=" * 65)
    # Column formatting layout padding visual alignments metrics array headers display template.
    print(
        f"  {'Source':<12} │ {'Strategy':<22} │ {'Est. Docs':>9} │ {'Est. Cost':>10}"
    )
    print("─" * 65)

    # Baseline analytics variable metrics accumulators track numeric parameters memory.
    total_docs = 0
    total_runs = 0

    # Master iteration execution check target options configurations sequences variables list maps.
    for source in target_sources:
        # Source criteria filtering parameters conditional selections evaluation data paths routing lookup dictionary logic.
        strategies = (
            strategies_override.get(source, CANDIDATE_STRATEGIES[source])
            if strategies_override
            else CANDIDATE_STRATEGIES[source]
        )
        # Local evaluation parsing matrix value mappings dictionary loop counters update data fields.
        doc_count = _count_docs(source)
        total_docs += doc_count * len(strategies)
        total_runs += len(strategies)

        # Strategy dimensions parameter inner loops combinations scanning options map execution loop block.
        for strategy in strategies:
            # Rough estimate: avg 500 chunks per source, avg 300 tokens per chunk.
            # Actual numbers chunking ke baad pata chalenge.
            # Linear scale calculations math evaluations metrics options properties variables conversion tracking code limit values.
            est_chunks = doc_count * 3
            est_tokens = est_chunks * 300
            est_cost   = (est_tokens / 1_000_000) * 0.02

            # Dashboard metrics line rendering string parameters data combinations format layouts values updates print.
            print(
                f"  {source:<12} │ {strategy:<22} │ {doc_count:>9} │ ${est_cost:>9.4f}"
            )

    # Footer dashboard visual boundaries format lines display tracking metrics print statements.
    print("─" * 65)
    print(
        f"  {'TOTAL':<12} │ {total_runs} runs{'':<17} │ {total_docs:>9} │"
        f" (see above)"
    )
    print("=" * 65)
    # Explicit warning notices user guidance tracking prints documentation log statements screen values metrics visibility.
    print(
        "\n  NOTE: Yeh rough estimates hain."
        "\n        Actual cost chunking ke baad pata chalegi."
        "\n        API strategies (semantic, improved_semantic, double_pass)"
        "\n        chunking ke waqt bhi embeddings consume karte hain.\n"
    )


# ── Confirmation ──────────────────────────────────────────────────────────────


def _confirm(target_sources: list[str], strategies_override: dict[str, list[str]] | None) -> bool:
    """
    User se confirmation lo before actual embedding run.
    API calls aur cost involved hain — accidental run avoid karo.
    """
    # Aggregate logic iterations variables mapping tracking length conditions array variables evaluation configurations options math sequence sum loops.
    total_runs = sum(
        len(
            strategies_override.get(s, CANDIDATE_STRATEGIES[s])
            if strategies_override
            else CANDIDATE_STRATEGIES[s]
        )
        for s in target_sources
    )

    # Prompt UI interface metrics logs tracking string formats variables console read values view layout parameters maps check statements text updates.
    print(f"\n  Sources  : {', '.join(target_sources)}")
    print(f"  Total runs: {total_runs} (source × strategy combinations)")
    print(f"  Embedding : {settings.openai_embedding_model}")
    print("\n  Yeh OpenAI API calls karega aur cost lagegi.")

    # Real-time console terminal variables read loop input conditions standard properties conversions logic boolean paths options.
    answer = input("\n  Proceed? (yes/no): ").strip().lower()
    return answer in ("yes", "y")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    # CLI framework configuration instantiation structural details options variables memory setup parameters command options documentation templates text.
    parser = argparse.ArgumentParser(
        description="SkyLex Phase 3 — Embedding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_embedding_pipeline.py
  python scripts/run_embedding_pipeline.py --source FAA_AD
  python scripts/run_embedding_pipeline.py --source FAA_AD --strategy hierarchical
  python scripts/run_embedding_pipeline.py --dry-run
  python scripts/run_embedding_pipeline.py --force-reindex
        """,
    )

    # String target filter option argument flags definitions limit configurations tracking data logic path branches arrays constraints.
    parser.add_argument(
        "--source",
        type=str,
        choices=list(CANDIDATE_STRATEGIES.keys()),
        default=None,
        help="Sirf ek source run karo (default: saare sources)",
    )
    # Method selector strategy argument mapping parameters dynamic flags choices execution control structures evaluation conditions logic strings.
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Sirf ek strategy run karo (--source ke saath use karo)",
    )
    # Testing bypass option switches Boolean modifiers values configuration checks metrics constraints options loop properties check flags.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Cost estimate print karo — koi embedding nahi hogi",
    )
    # Database reset destructive flag arguments boolean options configurations mapping logic pipeline trigger code parameters.
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Existing collection data delete karke fresh index karo",
    )
    # Interaction bypass parameters flag argument values terminal interface variables constraints check bypass prompt options flow models.
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirmation prompt skip karo",
    )

    # Dictionary mappings configuration objects output list variables format memory structures parse setup.
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # User defined properties configuration argument parsing objects execution framework options.
    args = _parse_args()

    # Target sources determine karo.
    # Conditional lists arrays variable assignments logic checks definitions dynamic path routes parameters index.
    target_sources: list[str] = (
        [args.source] if args.source else list(CANDIDATE_STRATEGIES.keys())
    )

    # Strategy override determine karo.
    # Fallback values constraints variables initial empty object dictionary placeholders setup mappings framework trace code check values logic pointer.
    strategies_override: dict[str, list[str]] | None = None
    if args.strategy:
        # Condition validation missing inputs safety boundaries protection constraints parameter print text strings check rules validation options.
        if not args.source:
            print(
                "\n[ERROR] --strategy ke saath --source bhi specify karo."
                "\nExample: --source FAA_AD --strategy hierarchical\n"
            )
            sys.exit(1)

        # Validate karo ki strategy us source ke liye valid candidate hai.
        # Local reference dictionary variables check target condition bounds variables options tracking error handling code check lists.
        valid = CANDIDATE_STRATEGIES[args.source]
        if args.strategy not in valid:
            print(
                f"\n[ERROR] '{args.strategy}' source '{args.source}' ke liye"
                f" valid candidate nahi hai."
                f"\nValid candidates: {valid}\n"
            )
            sys.exit(1)

        # Dictionary payload mappings structure configuration initialization settings dictionary target mapping fields trace loop variable arrays check layout options.
        strategies_override = {args.source: [args.strategy]}

    # Dry run — sirf estimate print karo.
    # Bypass conditional branches parameters execution loop skip function invoke return sequence operations control evaluation variables check block.
    if args.dry_run:
        _print_dry_run(target_sources, strategies_override)
        return

    # Actual run — confirmation lo (unless --yes flag hai).
    # Dashboard UI visibility formatting string display indicators updates printing log tracking configuration code template options.
    print("\n" + "=" * 65)
    print(f"{'SkyLex Phase 3 — Embedding Pipeline':^65}")
    print("=" * 65)

    # User validation confirmation skip option variables conditions boolean checks mapping execution loops path tracing parameters evaluation check variables cancel routing loop track code stop.
    if not args.yes and not _confirm(target_sources, strategies_override):
        print("\n  Cancelled.\n")
        return

    # Process initialization logger track indicators updates pipeline monitoring parameters format values terminal trace code level updates parameters.
    logger.info(
        "Embedding pipeline starting | sources=%s | force_reindex=%s",
        target_sources,
        args.force_reindex,
    )

    # Pipeline run karo.
    # Master architecture subroutine function module call parameters dictionary structures values map outputs dynamic metrics variables object collections tracking.
    all_metrics = run_full_pipeline(
        sources             = target_sources,
        strategies_override = strategies_override,
        force_reindex       = args.force_reindex,
    )

    # Summary print karo.
    # Display rendering layout templates string borders drawing parameters dashboard reporting prints visual parameters text view terminal fields matrix layout options.
    print("\n" + "=" * 65)
    print(f"{'Pipeline Complete — Summary':^65}")
    print("=" * 65)
    print(
        f"  {'Source':<12} │ {'Strategy':<22} │ {'Chunks':>7} │ {'Status':<10}"
    )
    print("─" * 65)

    # Reporting accumulator total values sum numeric constants variables mapping configuration memory.
    total_chunks = 0
    total_cost   = 0.0

    # Iteration cycle scanning final results output arrays combinations strings dictionary values variables mapping fields parameters loops tracing metrics formatting array strings outputs check.
    for run_key, metrics in sorted(all_metrics.items()):
        # Dynamic dictionary keys lookup memory validations tracking variable defaults strings conversion properties extraction tracking value references arrays check values mapping variables data options framework logic fields assignment.
        status  = metrics.get("status", "unknown")
        chunks  = metrics.get("chunk_count", 0)
        cost    = metrics.get("cost_usd", 0.0)
        source  = metrics.get("source", "")
        strategy = metrics.get("strategy", "")

        # Math formulas counters update tracking sequence logic parameters integer increments data.
        total_chunks += chunks
        total_cost   += cost

        # Text alignment characters variable printing console row outputs configurations trace.
        print(
            f"  {source:<12} │ {strategy:<22} │ {chunks:>7} │ {status:<10}"
        )

    # Frame bounding bottom graphics line log analytics string trace variables numbers layout complete parameters output trace code.
    print("─" * 65)
    print(f"  {'TOTAL':<12} │ {'':<22} │ {total_chunks:>7} │")
    print(f"  Estimated cost: ${total_cost:.4f}")
    print("=" * 65 + "\n")

    # Telemetry final check logs update reporting variables trace line text display options code block parameter level details configurations check trace execution.
    logger.info(
        "Pipeline done | total_chunks=%d | total_cost=$%.4f",
        total_chunks,
        total_cost,
    )


if __name__ == "__main__":
    main()