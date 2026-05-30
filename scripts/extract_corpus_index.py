"""
scripts/extract_corpus_index_v2.py

Golden Queries banane ke liye deep corpus reference extractor.

Problem with v1:
  - Sirf 500 chars preview → FAA_CFR ke 57K char docs ka 0.9% hi dekh paate the
  - Golden queries likhne ke liye actual content + section structure chahiye

Is version mein source-specific extraction strategy:

  FAA_CFR  → Section-wise split karo (§ markers pe), top 15 sections ka
              title + first 2000 chars extract karo. Bahut bade docs hain,
              isliye puri content nahi — lekin har relevant section ka snapshot.

  FAA_AD   → Full content (avg 833 chars, already short)

  FAA_AC   → Table of contents + numbered headings + first 5000 chars of body.
              FAA ACs structured hote hain, headings se hi query ideas milte hain.

  DGCA_CAR → Full content (avg 34K chars, manageable)

  SKYBRARY → Full content (avg 1.6K chars, already short)

Output files:
  data/gq_reference_FAA_CFR.json   — FAA CFR section snapshots
  data/gq_reference_FAA_AD.json    — FAA AD full content
  data/gq_reference_FAA_AC.json    — FAA AC structured content
  data/gq_reference_DGCA_CAR.json  — DGCA CAR full content
  data/gq_reference_SKYBRARY.json  — SKYbrary full content
  data/gq_reference_SUMMARY.txt    — Human-readable summary for query writing

Yeh files golden_queries.py banate waqt reference ke roop mein use hongi.
"""

import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR   = Path("data")

SOURCE_FOLDERS = {
    "FAA_CFR":  "faa_cfr",
    "FAA_AD":   "faa_ad",
    "FAA_AC":   "faa_ac",
    "DGCA_CAR": "dgca_car",
    "SKYBRARY": "skybrary",
}

# Source-specific content extraction depth
CONTENT_CONFIG = {
    "FAA_CFR":  {"mode": "sectioned",  "section_chars": 2000, "max_sections": 15},
    "FAA_AD":   {"mode": "full"},
    "FAA_AC":   {"mode": "structured", "body_chars": 5000},
    "DGCA_CAR": {"mode": "full"},
    "SKYBRARY": {"mode": "full"},
}

# Section split pattern for FAA_CFR
SECTION_SPLIT_RE = re.compile(r'(?=§\s*\d+[\.\d]*\s)', re.MULTILINE)

# Regulatory identifier patterns
IDENTIFIER_PATTERNS = {
    "section_refs": re.compile(r'§\s*[\d]+\.[\d]+[a-z]?', re.IGNORECASE),
    "part_refs":    re.compile(r'\bPart\s+\d+\b', re.IGNORECASE),
    "ad_numbers":   re.compile(r'\bAD[-\s]\d{4}[-\s]\d+[-\s]\d+', re.IGNORECASE),
    "car_refs":     re.compile(r'\bCAR[-\s](?:Section\s*)?\d+|CAR[-\s]\d+[-\s][A-Z]+|CAR\s+\d{2,3}\b', re.IGNORECASE),
    "numbered_heads": re.compile(r'^\d+\.\d*\s+[A-Z][^\n]{10,80}', re.MULTILINE),
    "all_caps_heads": re.compile(r'^[A-Z][A-Z\s]{6,50}$', re.MULTILINE),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raw_docs(source: str, folder: str) -> list[dict]:
    """Source folder se saare documents load karo (dedup included)."""
    source_dir = RAW_DATA_DIR / folder
    all_docs   = []
    seen_ids   = set()

    json_files = sorted(source_dir.rglob("*.json"))
    json_files = [f for f in json_files
                  if f.name != "hash_registry.json"
                  and not f.name.endswith("_meta.json")]

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            for doc in data.get("documents", []):
                doc_id = doc.get("doc_id", "")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)
        except Exception as e:
            print(f"  [WARN] {jf.name}: {e}")

    return all_docs


def extract_identifiers(content: str) -> dict:
    """Content se regulatory identifiers extract karo (first 60K chars scan)."""
    scan_text = content[:60000]
    result = {}
    for name, pat in IDENTIFIER_PATTERNS.items():
        matches = pat.findall(scan_text)
        unique  = list(dict.fromkeys(m.strip() for m in matches))[:30]
        if unique:
            result[name] = unique
    return result


def split_into_sections(content: str) -> list[str]:
    """FAA_CFR content ko § markers pe split karo."""
    parts = SECTION_SPLIT_RE.split(content)
    return [p.strip() for p in parts if p.strip()]


def extract_toc_headings(content: str) -> list[str]:
    """FAA AC style numbered headings extract karo (TOC / section headers)."""
    headings = []
    # Numbered headings: e.g. "1.1 Purpose", "7.2 Type B Applications:"
    num_pat = re.compile(r'^\d+[\.\d]*\s+[A-Z][^\n]{5,80}', re.MULTILINE)
    headings.extend(num_pat.findall(content[:20000]))
    return list(dict.fromkeys(h.strip() for h in headings))[:40]


# ── Per-source extractors ─────────────────────────────────────────────────────

def process_faa_cfr(docs: list[dict]) -> list[dict]:
    """
    FAA_CFR: Har doc ko sections mein split karo.
    Top sections (by § number relevance) ka snapshot extract karo.
    Focus on Part 91, Part 121 docs — Air India operations ke liye relevant.
    """
    results = []
    cfg = CONTENT_CONFIG["FAA_CFR"]

    for doc in tqdm(docs, desc="  FAA_CFR", leave=False):
        content = doc.get("content", "")
        title   = doc.get("title", "")
        doc_id  = doc.get("doc_id", "")
        url     = doc.get("url", "")

        # Priority flag: Part 91, 121 docs are most relevant for Air India
        is_priority = any(
            kw in title for kw in ["Part 91", "Part 121", "Part 117", "Part 119",
                                    "Part 43", "Part 145", "Part 39", "Part 135"]
        )

        sections = split_into_sections(content)

        # Top sections ka snapshot
        section_snapshots = []
        for sec in sections[:cfg["max_sections"]]:
            # Section header (first line)
            first_line = sec.splitlines()[0][:200] if sec.splitlines() else ""
            section_snapshots.append({
                "header":  first_line,
                "content": sec[:cfg["section_chars"]],
                "chars":   len(sec),
            })

        identifiers = extract_identifiers(content)

        results.append({
            "doc_id":            doc_id,
            "title":             title,
            "url":               url,
            "total_chars":       len(content),
            "total_sections":    len(sections),
            "is_priority":       is_priority,
            "identifiers":       identifiers,
            "section_snapshots": section_snapshots,
        })

    return results


def process_faa_ad(docs: list[dict]) -> list[dict]:
    """FAA_AD: Full content (short abstracts, avg 833 chars)."""
    results = []
    for doc in tqdm(docs, desc="  FAA_AD", leave=False):
        content     = doc.get("content", "")
        identifiers = extract_identifiers(content)

        # Infer aircraft type from title/content
        aircraft_types = []
        for pattern in ["787", "777", "737", "A320", "A321", "A350", "A330",
                         "757", "767", "GE90", "GEnx", "CFM", "LEAP", "Trent"]:
            if pattern.lower() in content.lower():
                aircraft_types.append(pattern)

        results.append({
            "doc_id":         doc.get("doc_id", ""),
            "title":          doc.get("title", ""),
            "url":            doc.get("url", ""),
            "chars":          len(content),
            "aircraft_types": aircraft_types,
            "identifiers":    identifiers,
            "full_content":   content,  # Full — short enough
        })

    return results


def process_faa_ac(docs: list[dict]) -> list[dict]:
    """
    FAA_AC: Structured extraction.
    - Full TOC/headings (from first 20K chars)
    - First 5000 chars of body
    - All regulatory identifiers
    """
    results = []
    cfg = CONTENT_CONFIG["FAA_AC"]

    for doc in tqdm(docs, desc="  FAA_AC", leave=False):
        content     = doc.get("content", "")
        identifiers = extract_identifiers(content)
        headings    = extract_toc_headings(content)

        # AC number from title
        ac_match = re.search(r'AC\s+[\d\.-]+', doc.get("title", ""))
        ac_number = ac_match.group(0) if ac_match else ""

        results.append({
            "doc_id":       doc.get("doc_id", ""),
            "title":        doc.get("title", ""),
            "url":          doc.get("url", ""),
            "ac_number":    ac_number,
            "total_chars":  len(content),
            "identifiers":  identifiers,
            "headings":     headings,
            "body_excerpt": content[:cfg["body_chars"]],
        })

    return results


def process_dgca_car(docs: list[dict]) -> list[dict]:
    """DGCA_CAR: Full content (avg 34K chars — manageable)."""
    results = []
    for doc in tqdm(docs, desc="  DGCA_CAR", leave=False):
        content     = doc.get("content", "")
        identifiers = extract_identifiers(content)

        # Extract numbered sections (DGCA uses 1., 1.1, 2., etc.)
        section_headers = re.findall(
            r'^\d+\.\s+[A-Z][^\n]{5,80}', content, re.MULTILINE
        )[:30]

        results.append({
            "doc_id":          doc.get("doc_id", ""),
            "title":           doc.get("title", ""),
            "url":             doc.get("url", ""),
            "total_chars":     len(content),
            "identifiers":     identifiers,
            "section_headers": section_headers,
            "full_content":    content,  # Full — 34K is fine
        })

    return results


def process_skybrary(docs: list[dict]) -> list[dict]:
    """SKYBRARY: Full content (avg 1.6K chars — very short)."""
    results = []
    for doc in tqdm(docs, desc="  SKYBRARY", leave=False):
        content     = doc.get("content", "")
        identifiers = extract_identifiers(content)

        # Extract bullet points / key terms
        bullet_lines = [
            line.strip() for line in content.splitlines()
            if line.strip().startswith("-") or line.strip().startswith("•")
        ][:20]

        results.append({
            "doc_id":       doc.get("doc_id", ""),
            "title":        doc.get("title", ""),
            "url":          doc.get("url", ""),
            "total_chars":  len(content),
            "identifiers":  identifiers,
            "bullet_points": bullet_lines,
            "full_content": content,
        })

    return results


# ── Summary writer ────────────────────────────────────────────────────────────

def write_summary(all_data: dict, output_path: Path) -> None:
    """Human-readable summary text file for golden query writing."""
    lines = []
    lines.append("=" * 80)
    lines.append("SKYLEX CORPUS REFERENCE SUMMARY — For Golden Query Writing")
    lines.append("=" * 80)

    # FAA_CFR summary
    cfr_docs = all_data["FAA_CFR"]
    lines.append(f"\n{'─' * 60}")
    lines.append(f"FAA_CFR — {len(cfr_docs)} documents")
    lines.append("─" * 60)
    priority = [d for d in cfr_docs if d["is_priority"]]
    lines.append(f"Priority docs (Part 91/121/117/etc): {len(priority)}")
    lines.append("\nTOP PRIORITY DOCS:")
    for d in sorted(priority, key=lambda x: -x["total_chars"])[:15]:
        lines.append(f"  [{d['doc_id'][:12]}] {d['title'][:80]}")
        ids = d.get("identifiers", {})
        secs = ids.get("section_refs", [])[:8]
        if secs:
            lines.append(f"    Sections: {', '.join(secs)}")
        # First section snapshot
        snaps = d.get("section_snapshots", [])
        if snaps:
            snap = snaps[0]
            lines.append(f"    First section: {snap['header'][:100]}")
            lines.append(f"    Content: {snap['content'][:300].replace(chr(10), ' ')}")

    # FAA_AD summary
    ad_docs = all_data["FAA_AD"]
    lines.append(f"\n{'─' * 60}")
    lines.append(f"FAA_AD — {len(ad_docs)} documents")
    lines.append("─" * 60)
    all_ad_numbers = []
    for d in ad_docs:
        all_ad_numbers.extend(d["identifiers"].get("ad_numbers", []))
    all_ad_numbers = list(dict.fromkeys(all_ad_numbers))
    lines.append(f"Unique AD numbers found: {len(all_ad_numbers)}")
    lines.append(f"Sample ADs: {', '.join(all_ad_numbers[:15])}")
    lines.append("\nSAMPLE AD DOCS (full content):")
    # Boeing 787 ADs (most relevant for Air India)
    b787_ads = [d for d in ad_docs if "787" in d.get("full_content", "")][:5]
    for d in b787_ads:
        lines.append(f"\n  [{d['doc_id'][:12]}] {d['title'][:80]}")
        lines.append(f"  Aircraft: {', '.join(d['aircraft_types'][:5])}")
        lines.append(f"  Content: {d['full_content'][:400].replace(chr(10), ' ')}")

    # FAA_AC summary
    ac_docs = all_data["FAA_AC"]
    lines.append(f"\n{'─' * 60}")
    lines.append(f"FAA_AC — {len(ac_docs)} documents")
    lines.append("─" * 60)
    for d in ac_docs:
        lines.append(f"\n  [{d['ac_number'] or d['doc_id'][:12]}] {d['title'][:70]}")
        lines.append(f"  Chars: {d['total_chars']:,}")
        if d["headings"]:
            lines.append(f"  Key headings: {'; '.join(d['headings'][:5])}")
        ids = d.get("identifiers", {})
        secs = ids.get("section_refs", [])[:5]
        if secs:
            lines.append(f"  Sections cited: {', '.join(secs)}")
        lines.append(f"  Excerpt: {d['body_excerpt'][:400].replace(chr(10), ' ')}")

    # DGCA_CAR summary
    car_docs = all_data["DGCA_CAR"]
    lines.append(f"\n{'─' * 60}")
    lines.append(f"DGCA_CAR — {len(car_docs)} documents")
    lines.append("─" * 60)
    for d in car_docs:
        lines.append(f"\n  [{d['doc_id'][:12]}] {d['title'][:80]}")
        lines.append(f"  Chars: {d['total_chars']:,}")
        lines.append(f"  Sections: {', '.join(d['section_headers'][:6])}")
        lines.append(f"  Full content (first 1500 chars):")
        lines.append(f"  {d['full_content'][:1500].replace(chr(10), ' ')}")

    # SKYBRARY summary
    sky_docs = all_data["SKYBRARY"]
    lines.append(f"\n{'─' * 60}")
    lines.append(f"SKYBRARY — {len(sky_docs)} documents")
    lines.append("─" * 60)
    for d in sky_docs:
        lines.append(f"\n  [{d['doc_id'][:12]}] {d['title']}")
        lines.append(f"  {d['full_content'][:600].replace(chr(10), ' ')}")

    lines.append("\n" + "=" * 80)
    lines.append("END OF SUMMARY")
    lines.append("=" * 80)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Summary saved: {output_path} ({output_path.stat().st_size // 1024} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print(f"{'SkyLex Deep Corpus Extractor v2':^70}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    PROCESSORS = {
        "FAA_CFR":  process_faa_cfr,
        "FAA_AD":   process_faa_ad,
        "FAA_AC":   process_faa_ac,
        "DGCA_CAR": process_dgca_car,
        "SKYBRARY": process_skybrary,
    }

    all_data = {}

    for source, folder in SOURCE_FOLDERS.items():
        print(f"\nLoading {source}...")
        docs = load_raw_docs(source, folder)
        print(f"  Loaded {len(docs)} documents")

        processor = PROCESSORS[source]
        processed = processor(docs)

        # Save per-source JSON
        out_json = OUTPUT_DIR / f"gq_reference_{source}.json"
        out_json.write_text(
            json.dumps(processed, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        size_kb = out_json.stat().st_size // 1024
        print(f"  Saved: {out_json.name} ({size_kb} KB, {len(processed)} docs)")

        all_data[source] = processed

    # Human-readable summary
    print("\nWriting human-readable summary...")
    write_summary(all_data, OUTPUT_DIR / "gq_reference_SUMMARY.txt")

    # Print quick stats
    print("\n" + "=" * 70)
    print(f"{'EXTRACTION COMPLETE':^70}")
    print("=" * 70)
    for source, docs in all_data.items():
        total_chars = sum(
            d.get("total_chars", d.get("chars", 0)) for d in docs
        )
        print(f"  {source:<12} : {len(docs):3} docs | "
              f"{total_chars:>12,} chars extracted")

    print(f"\nOutput files in: {OUTPUT_DIR}/")
    print("  gq_reference_FAA_CFR.json")
    print("  gq_reference_FAA_AD.json")
    print("  gq_reference_FAA_AC.json")
    print("  gq_reference_DGCA_CAR.json")
    print("  gq_reference_SKYBRARY.json")
    print("  gq_reference_SUMMARY.txt  ← Yeh padhke golden queries likho")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()