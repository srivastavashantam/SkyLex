# =============================================================================
# SkyLex — FAA Advisory Circular (AC) Ingester
# Source: https://www.federalregister.gov/api/v1 (REST API — free, no auth)
# Strategy:
#   1. 14 aviation-relevant AC categories targeted fetch
#   2. Federal Register API — structured JSON, no PDF parsing needed
#   3. Pagination handling — fetch all pages per category
#   4. Hash-based change detection — skip unchanged ACs on re-runs
# Coverage: All 14 AC series relevant to Air India operations
# =============================================================================
# 📌 YE FILE KYA KARTI HAI?
# FAA Advisory Circulars (ACs) ingest karta hai — Federal Register API se.
# ACs mandatory nahi hote (ADs ki tarah) — ye FAA ke "recommended practices"
# aur interpretations hain. Lekin aviation compliance ke liye bahut important hain.
# e.g., AC 120-92B = Safety Management System implementation guidance.
#
# FAAAADIngester SE SIMILARITY:
# Ye file FAAAADIngester ka almost twin hai — same API, same pattern.
# Fark sirf itna hai:
#   - Search terms alag hain (ADs = aircraft-type, ACs = topic/category)
#   - AC categories 14 hain (AD aircraft types 7 the)
#   - ACs ka from_date 2020 hai (ADs ka 2024 tha — ACs older bhi relevant hain)
#   - EXTRA FEATURE: Cross-category deduplication — ek AC multiple categories
#     mein match kar sakta hai, isliye seen_doc_nums set rakha gaya hai
#
# FLOW:
#   fetch() → 14 categories mein se har ek ke liye paginate → raw metadata collect
#   parse() → deduplicate → hash check → Document banao
# =============================================================================

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from tqdm import tqdm  # Progress bar — categories parse hote waqt visual progress

from config.settings import settings
from exceptions.exceptions import SkyLexIngestionError
from ingestion.base_ingester import BaseIngester, Document
from monitoring.logger import get_logger

# Is module ka named logger — logs mein "ingestion.faa_ac_ingester" dikhega
logger = get_logger(__name__)

# ─── FEDERAL REGISTER API CONFIGURATION ──────────────────────────────────────
FAA_AC_BASE_URL = "https://www.federalregister.gov/api/v1"
FAA_AC_AGENCY = "federal-aviation-administration"  # Agency filter — sirf FAA documents

# ─── REQUEST HEADERS ─────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "SkyLex-ResearchBot/1.0 (Aviation Compliance RAG System)",
    "Accept": "application/json",
}

# ─── RATE LIMITING & RETRY ───────────────────────────────────────────────────
REQUEST_DELAY_SECONDS = 1.0  # Har successful request ke baad 1 second wait
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# ─── FETCH WINDOW ────────────────────────────────────────────────────────────
# ACs ke liye 2020 se fetch karte hain — ADs se zyada purana window isliye ki
# ACs guidance documents hain jo kaafi time tak relevant rehte hain
# (AD 2 saal purana outdated ho sakta hai, AC 5 saal purana bhi valid hota hai)
AC_FETCH_FROM_DATE = "2020-01-01"

# Federal Register API se sirf ye fields maango — bandwidth save karne ke liye
AC_FIELDS = [
    "title",
    "document_number",
    "pdf_url",
    "publication_date",
    "abstract",
    "type",
    "effective_on",
    "docket_ids",
    "html_url",
]

# =============================================================================
# 14 AC Categories — All Air India relevant series
# Key maps to search term targeting specific AC series
# =============================================================================
# 💡 14 CATEGORIES KYA HAIN?
# FAA ACs hazaaron hain — saare ingest karna impractical aur irrelevant hoga.
# Ye 14 categories Air India ke operations ke liye specifically chosen hain:
# maintenance, crew training, navigation, safety management, etc.
# Har category ka search term Federal Register API ke liye optimized hai.
AC_SEARCH_TERMS: dict[str, str] = {
    # General aviation topics — weather, turbulence, safety
    "ac_general": "advisory circular aviation weather turbulence",
    # Transport category aircraft — Part 25 (Boeing 787, 777, Airbus A350)
    "ac_transport_aircraft": "advisory circular transport category aircraft part 25",
    # Airworthiness certification procedures — Part 21
    "ac_airworthiness": "advisory circular airworthiness certification part 21",
    # Maintenance, inspection, repair — Part 43
    "ac_maintenance": "advisory circular maintenance inspection repair part 43",
    # General operating rules — Part 91
    "ac_flight_ops": "advisory circular general operating flight rules part 91",
    # Air carrier operations — Part 120 (CRITICAL for Air India)
    "ac_air_carrier": "advisory circular air carrier operations part 120",
    # Domestic and flag operations — Part 121 (CRITICAL for Air India)
    "ac_domestic_ops": "advisory circular domestic flag operations part 121",
    # Flight crew training and simulators
    "ac_crew_training": "advisory circular flight crew training simulator",
    # Icing conditions — operationally critical
    "ac_icing": "advisory circular icing conditions flight operations",
    # Navigation and avionics — RNAV, RNP, ADS-B
    "ac_navigation": "advisory circular navigation avionics RNAV RNP",
    # Safety Management Systems — SMS
    "ac_sms": "advisory circular safety management system SMS",
    # Dangerous goods and hazardous materials carriage
    "ac_dangerous_goods": "advisory circular dangerous goods hazardous materials",
    # Airport and ground operations
    "ac_airports": "advisory circular airport operations ground handling",
    # Electronic Flight Bags — EFB
    "ac_efb": "advisory circular electronic flight bag EFB",
}


class FAAAACIngester(BaseIngester):
    """
    Ingester for FAA Advisory Circulars (ACs).
    Fetches ACs from Federal Register API — 14 aviation-relevant categories.
    Each AC becomes one Document object with full metadata.
    Architecture mirrors FAAAADIngester — same API, different search terms.
    """

    def __init__(
        self,
        ac_categories: Optional[list[str]] = None,
        from_date: Optional[str] = None,
    ) -> None:
        """
        Args:
            ac_categories: Optional list of AC category keys to fetch.
                           e.g., ["ac_air_carrier", "ac_maintenance"]
                           None means fetch ALL 14 categories.
            from_date:     Fetch ACs published on or after this date (YYYY-MM-DD).
                           Default: AC_FETCH_FROM_DATE (2020-01-01)
        """
        super().__init__(source_name="FAA_AC")

        # ─── CATEGORY KEYS VALIDATE KARO ─────────────────────────────────────
        # FAAAADIngester ki tarah same pattern — upfront validation
        if ac_categories:
            invalid = [k for k in ac_categories if k not in AC_SEARCH_TERMS]
            if invalid:
                raise SkyLexIngestionError(
                    "Invalid AC category keys provided",
                    details=f"Invalid: {invalid} | Valid: {list(AC_SEARCH_TERMS.keys())}"
                )
            self.ac_categories = ac_categories
        else:
            # None = saari 14 categories fetch karo
            self.ac_categories = list(AC_SEARCH_TERMS.keys())

        self.from_date = from_date or AC_FETCH_FROM_DATE  # Default to 2020-01-01

        # Hash registry — FAAAADIngester jaisa same pattern
        self.registry_path = self.output_dir / "hash_registry.json"
        self.hash_registry = self._load_hash_registry()

        self.logger.info(
            f"FAAAACIngester initialized | "
            f"Categories: {self.ac_categories} | "
            f"From: {self.from_date}"
        )

    # -------------------------------------------------------------------------
    # Hash Registry
    # -------------------------------------------------------------------------

    def _load_hash_registry(self) -> dict:
        """Load previously stored content hashes from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_hash_registry(self) -> None:
        """Save updated hash registry to disk."""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.hash_registry, f, indent=2)

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content string for change detection."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    # HTTP Helpers
    # -------------------------------------------------------------------------

    def _get_json(
        self,
        client: httpx.Client,
        url: str,
        params: Optional[dict] = None
    ) -> Any:
        """
        GET request returning JSON — with retry logic.

        Args:
            client: httpx client instance
            url:    Full URL to fetch
            params: Optional query parameters

        Returns:
            Parsed JSON response

        Raises:
            SkyLexIngestionError: If request fails after MAX_RETRIES
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.get(
                    url,
                    params=params,    # Dict automatically URL-encoded hota hai
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()         # 4xx/5xx pe exception
                time.sleep(REQUEST_DELAY_SECONDS)   # Rate limiting — 1 req/sec
                return response.json()

            except httpx.HTTPStatusError as e:
                # Server ne refuse kiya — retry se theek nahi hoga
                raise SkyLexIngestionError(
                    f"HTTP error fetching {url}",
                    details=f"Status: {e.response.status_code}"
                )
            except httpx.HTTPError as e:
                # Network error — retry karo
                self.logger.warning(
                    f"⚠️ Network error | Attempt: {attempt}/{MAX_RETRIES} | {e}"
                )
                if attempt == MAX_RETRIES:
                    raise SkyLexIngestionError(
                        f"Network error after {MAX_RETRIES} retries",
                        details=str(e)
                    )
                time.sleep(2 ** attempt)  # Exponential backoff — 2s, 4s, 8s

    # -------------------------------------------------------------------------
    # fetch() — required by BaseIngester
    # -------------------------------------------------------------------------

    def fetch(self) -> dict:
        """
        Fetch FAA ACs from Federal Register API — category by category.

        Strategy:
        - For each of 14 AC categories: paginate through all results
        - Collect metadata for each AC
        - Return all raw AC data grouped by category

        Returns:
            Dict with raw AC data grouped by category key

        Raises:
            SkyLexIngestionError: If fetch fails completely
        """
        # Return structure FAAAADIngester jaisa:
        # {"ac_air_carrier": [ac1, ac2, ...], "ac_maintenance": [...]}
        all_raw_data: dict[str, list[dict]] = {}

        with httpx.Client(follow_redirects=True) as client:
            for category_key in self.ac_categories:
                search_term = AC_SEARCH_TERMS[category_key]
                self.logger.info(
                    f"📥 Fetching ACs | Category: {category_key} | "
                    f"From: {self.from_date}"
                )

                # Har category ke liye paginated fetch
                acs = self._fetch_category_acs(client, category_key, search_term)
                all_raw_data[category_key] = acs

                self.logger.info(
                    f"✅ {category_key} — {len(acs)} ACs fetched"
                )

        return all_raw_data

    def _fetch_category_acs(
        self,
        client: httpx.Client,
        category_key: str,
        search_term: str,
    ) -> list[dict]:
        """
        Fetch all ACs for one category — handles pagination.

        Args:
            client:       httpx client instance
            category_key: e.g., "ac_air_carrier"
            search_term:  Federal Register search query

        Returns:
            List of raw AC metadata dicts
        """
        all_acs: list[dict] = []
        page = 1
        total_pages: Optional[int] = None  # Pehli response mein pata chalega

        # ─── PAGINATION LOOP ──────────────────────────────────────────────────
        # FAAAADIngester._fetch_aircraft_ads() jaisa exactly same pattern
        # Federal Register API max 100 results per page deta hai
        while True:
            params = {
                "conditions[agencies][]": FAA_AC_AGENCY,
                "conditions[term]": search_term,                     # Category-specific search
                "conditions[publication_date][gte]": self.from_date, # From date filter
                "per_page": 100,                                      # Max per page
                "page": page,
                "order": "newest",                                    # Latest ACs pehle
                "fields[]": AC_FIELDS,                                # Sirf zaroori fields
            }

            try:
                data = self._get_json(
                    client,
                    f"{FAA_AC_BASE_URL}/documents.json",
                    params=params
                )
            except SkyLexIngestionError as e:
                # Ek page fail = yahan tak ka data rakh lo, aage mat jao
                self.logger.warning(
                    f"⚠️ Page {page} fetch failed for {category_key} — stopping | {e}"
                )
                break

            results = data.get("results", [])
            if not results:
                break  # Koi results nahi — is category ka pagination khatam

            all_acs.extend(results)

            # Pehli page pe total info log karo — ek baar hi
            if total_pages is None:
                total_pages = data.get("total_pages", 1)
                total_count = data.get("count", 0)
                self.logger.info(
                    f"   {category_key}: {total_count} total ACs | "
                    f"{total_pages} pages"
                )

            self.logger.debug(
                f"   Page {page}/{total_pages} — {len(results)} ACs"
            )

            # Last page check
            if page >= (total_pages or 1):
                break

            page += 1  # Agla page

        return all_acs

    # -------------------------------------------------------------------------
    # parse() — required by BaseIngester
    # -------------------------------------------------------------------------

    def parse(self, raw_data: Any) -> list[Document]:
        """
        Parse raw AC metadata into Document objects.
        Each AC becomes one Document.
        Hash-based change detection skips unchanged ACs.
        Deduplication — same AC may appear in multiple categories.

        Args:
            raw_data: Dict from fetch() — {category_key: [ac_dicts]}

        Returns:
            List of Document objects — one per new/changed AC
        """
        documents: list[Document] = []
        total_skipped = 0

        # 🔑 CROSS-CATEGORY DEDUPLICATION —
        # Ye FAAAADIngester mein nahi tha — AC-specific feature hai.
        # Problem: Ek AC multiple categories mein match kar sakta hai.
        # e.g., AC 120-92B (SMS) "ac_sms" aur "ac_air_carrier" dono mein aayega.
        # seen_doc_nums set mein document_number store karte hain —
        # pehli baar process karo, dobara same category se aaye toh skip karo.
        seen_doc_nums: set[str] = set()

        for category_key, acs in raw_data.items():
            self.logger.info(
                f"🔍 Parsing {len(acs)} ACs | Category: {category_key}"
            )

            for ac in tqdm(acs, desc=f"Parsing {category_key}", unit="AC"):
                doc_num = ac.get("document_number", "unknown")

                # ─── DEDUPLICATION CHECK ──────────────────────────────────────
                # Pehle deduplication check karo — hash check se pehle
                # (Hash compute karna expensive hai — duplicate ko pehle hi skip karo)
                if doc_num in seen_doc_nums:
                    total_skipped += 1
                    continue
                seen_doc_nums.add(doc_num)  # Pehli baar dekha — register karo

                # ─── HASH CHECK ───────────────────────────────────────────────
                # FAAAADIngester ki tarah same pattern
                meta_str = json.dumps(ac, sort_keys=True)  # Deterministic JSON string
                content_hash = self._compute_hash(meta_str)
                registry_key = f"faa_ac_{doc_num}"

                if self.hash_registry.get(registry_key) == content_hash:
                    # Unchanged AC — skip karo
                    total_skipped += 1
                    continue

                # ─── CONTENT BUILD KARO ───────────────────────────────────────
                # Title + Abstract = RAG ke liye sufficient context
                content_parts = []
                if ac.get("title"):
                    content_parts.append(f"Title: {ac['title']}")
                if ac.get("abstract"):
                    content_parts.append(f"Abstract: {ac['abstract']}")

                content = "\n\n".join(content_parts)
                if not content.strip():
                    continue  # Koi text nahi — empty AC skip karo

                # ─── DOC ID GENERATE KARO ────────────────────────────────────
                doc_id = hashlib.md5(
                    f"FAA_AC_{doc_num}".encode()
                ).hexdigest()[:12]

                doc = Document(
                    content=content,
                    source="FAA_AC",
                    doc_id=f"ac-{doc_id}",
                    url=ac.get("html_url", ""),
                    title=ac.get("title", f"AC {doc_num}"),
                    metadata={
                        "document_number": doc_num,
                        "category_key": category_key,              # Kaunsi category mein PEHLI BAAR mila
                        "publication_date": ac.get("publication_date", ""),
                        "effective_on": ac.get("effective_on", ""),
                        "pdf_url": ac.get("pdf_url", ""),
                        "docket_ids": ac.get("docket_ids", []),
                        "regulation_type": "AC",
                        "jurisdiction": "USA",
                        "authority": "FAA",
                    },
                )
                documents.append(doc)

                # Memory mein hash update karo
                self.hash_registry[registry_key] = content_hash

        # Saari categories process ho gayi — ek baar disk pe save karo
        self._save_hash_registry()

        self.logger.info(
            f"✅ Parse complete | "
            f"New/changed: {len(documents)} | "
            f"Skipped (duplicates + no change): {total_skipped}"  # Dono reasons ek saath count mein
        )
        return documents