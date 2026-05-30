"""
scripts/read_raw_data.py

Temporary script — golden queries likhne ke liye saare raw documents read karta hai.
Har source ke documents ka content print karta hai — tqdm se progress track hoti hai.

Usage:
    python scripts/read_raw_data.py > raw_data_output.txt 2>&1
"""

import io
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Windows terminal encoding fix — UTF-8 force karo
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RAW_DATA_DIR = Path("data/raw")

SOURCE_FOLDERS = {
    "FAA_CFR":  "faa_cfr",
    "FAA_AD":   "faa_ad",
    "FAA_AC":   "faa_ac",
    "DGCA_CAR": "dgca_car",
    "SKYBRARY": "skybrary",
}

# Har source se kitne docs print karne hain
DOCS_TO_PRINT = {
    "FAA_CFR":  15,
    "FAA_AD":   30,
    "FAA_AC":   5,
    "DGCA_CAR": 6,
    "SKYBRARY": 20,
}

# Content kitna print karna hai per doc (chars)
CONTENT_PREVIEW = {
    "FAA_CFR":  2000,
    "FAA_AD":   1500,
    "FAA_AC":   3000,
    "DGCA_CAR": 3000,
    "SKYBRARY": 2000,
}


def load_documents(source: str, folder: str) -> list[dict]:
    """Source folder se saare JSON files load karo."""
    source_dir = RAW_DATA_DIR / folder
    all_docs = []
    seen_ids = set()

    json_files = sorted(source_dir.rglob("*.json"))
    json_files = [
        f for f in json_files
        if f.name != "hash_registry.json"
        and not f.name.endswith("_meta.json")
    ]

    for json_file in tqdm(json_files, desc=f"  Loading {source}", leave=False):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            for doc in data.get("documents", []):
                doc_id = doc.get("doc_id", "")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)
        except Exception as e:
            print(f"    [ERROR] {json_file.name}: {e}")

    return all_docs


def main():
    sep = "=" * 80
    thin = "-" * 70

    print(f"\n{sep}")
    print(f"{'SkyLex Raw Data Reader - Golden Queries Preparation':^80}")
    print(f"{sep}")

    for source, folder in tqdm(SOURCE_FOLDERS.items(), desc="Sources", unit="source"):
        print(f"\n\n{sep}")
        print(f"SOURCE: {source}  |  Folder: data/raw/{folder}")
        print(f"{sep}")

        docs = load_documents(source, folder)
        print(f"Total documents loaded: {len(docs)}\n")

        sample_size = min(DOCS_TO_PRINT[source], len(docs))
        preview_chars = CONTENT_PREVIEW[source]

        for i, doc in enumerate(
            tqdm(docs[:sample_size], desc=f"  Printing {source}", leave=False)
        ):
            print(f"\n{thin}")
            print(f"DOC #{i+1}")
            print(f"  doc_id : {doc.get('doc_id', 'N/A')}")
            print(f"  title  : {doc.get('title', 'N/A')}")
            print(f"  url    : {doc.get('url', 'N/A')}")
            print(f"  chars  : {len(doc.get('content', ''))}")
            print(f"{thin}")

            content = doc.get("content", "")
            print(content[:preview_chars])

            if len(content) > preview_chars:
                print(
                    f"\n  ... [truncated - "
                    f"{len(content) - preview_chars} more chars]"
                )
            print()

    print(f"\n{sep}")
    print("Read complete - ab golden queries likhi ja sakti hain.")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()