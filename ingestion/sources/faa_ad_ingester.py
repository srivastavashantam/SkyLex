# =============================================================================
# SkyLex — FAA AD (Airworthiness Directive) Ingester
# Source: https://www.federalregister.gov/api/v1 (REST API — free, no auth)
# Strategy:
#   1. Aircraft-type targeted fetch — Air India fleet relevant only
#   2. Federal Register API — structured JSON, no PDF parsing needed
#   3. Pagination handling — fetch all pages per aircraft type
#   4. Hash-based change detection — skip unchanged ADs on re-runs
#   5. PDF download — optional, saves raw PDF alongside metadata
# =============================================================================
# 📌 YE FILE KYA KARTI HAI?
# FAA Airworthiness Directives (ADs) ingest karta hai — Federal Register API se.
# ADs mandatory safety instructions hote hain jo FAA issue karta hai specific
# aircraft types ke liye. e.g., "Boeing 787 ke fuel pump X ko 500 flight hours
# mein replace karo."
#
# AIR INDIA FLEET FOCUS:
# Saari ADs fetch nahi karta — sirf Air India ke relevant aircraft types ke liye:
# Boeing 787, 777, 737 | Airbus A320, A321, A350 | GE Engines
#
# KEY FEATURES:
#   1. Aircraft-type targeted search — irrelevant ADs skip hote hain
#   2. Pagination — ek aircraft ke 100+ ADs ho sakte hain, sab fetch hote hain
#   3. Hash-based change detection — re-run pe unchanged ADs skip hote hain
#   4. Optional PDF download — default off (fast) — on karo toh raw PDF bhi save hoga
#
# FLOW:
#   fetch() → har aircraft type ke liye paginate → raw AD metadata collect karo
#   parse() → hash check → Document banao → optionally PDF download karo
# =============================================================================

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from tqdm import tqdm  # Progress bar — kai ADs parse hote waqt visual progress dikhata hai

from config.settings import settings
from exceptions.exceptions import SkyLexIngestionError
from ingestion.base_ingester import BaseIngester, Document
from monitoring.logger import get_logger

# Is module ka named logger — logs mein "ingestion.faa_ad_ingester" dikhega
logger = get_logger(__name__)

# ─── FEDERAL REGISTER API CONFIGURATION ──────────────────────────────────────
FAA_AD_BASE_URL = "https://www.federalregister.gov/api/v1"
FAA_AD_AGENCY = "federal-aviation-administration"  # API mein agency filter ke liye

# ─── REQUEST HEADERS ─────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "SkyLex-ResearchBot/1.0 (Aviation Compliance RAG System)",
    "Accept": "application/json",
}

# ─── RATE LIMITING & RETRY ───────────────────────────────────────────────────
REQUEST_DELAY_SECONDS = 1.0  # Har successful request ke baad 1 second ruko
REQUEST_TIMEOUT = 30          # Per request timeout
MAX_RETRIES = 3               # Network errors pe retry count

# ─── FETCH CONFIGURATION ─────────────────────────────────────────────────────
# 2 years back se fetch karo — operationally relevant window
# Bahut purane ADs already complied ho chuke hote hain — unhe ingest karna wasteful hai
AD_FETCH_FROM_DATE = "2024-01-01"

# Federal Register API se sirf ye fields maango — bandwidth save karna ke liye
# Agar sab fields maango toh response bahut bada hoga, useful data kam
AD_FIELDS = [
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

# ─── AIR INDIA FLEET SEARCH TERMS ────────────────────────────────────────────
# Har aircraft family ke liye ek search term — Federal Register API ka search query
# Key = internal identifier, Value = actual search string jo API ko bheja jaata hai
AIRCRAFT_SEARCH_TERMS: dict[str, str] = {
    "boeing_787": "airworthiness directive boeing 787",
    "boeing_777": "airworthiness directive boeing 777",
    "boeing_737": "airworthiness directive boeing 737",
    "airbus_a320": "airworthiness directive airbus a320",
    "airbus_a321": "airworthiness directive airbus a321",
    "airbus_a350": "airworthiness directive airbus a350",
    "ge_engines": "airworthiness directive general electric engine",
}


class FAAAADIngester(BaseIngester):
    """
    Ingester for FAA Airworthiness Directives (ADs).
    Fetches ADs from Federal Register API — targeted by Air India fleet types.
    Each AD becomes one Document object with full metadata.
    Supports optional PDF download alongside metadata.
    """

    def __init__(
        self,
        aircraft_keys: Optional[list[str]] = None,
        from_date: Optional[str] = None,
        download_pdfs: bool = False,
    ) -> None:
        """
        Args:
            aircraft_keys: Optional list of aircraft type keys to fetch.
                           e.g., ["boeing_787", "airbus_a320"]
                           None means fetch ALL aircraft types.
            from_date:     Fetch ADs published on or after this date (YYYY-MM-DD).
                           Default: AD_FETCH_FROM_DATE (2024-01-01)
            download_pdfs: Whether to download PDF files alongside metadata.
                           Default: False — metadata only for faster ingestion.
        """
        super().__init__(source_name="FAA_AD")

        # ─── AIRCRAFT KEYS VALIDATE KARO ─────────────────────────────────────
        # User ne koi invalid aircraft key diya (typo etc.) toh startup pe hi fail karo
        # Baad mein koi valid aur koi invalid key ke saath 50% kaam karna
        # worse hoga — isliye upfront validation
        if aircraft_keys:
            invalid = [k for k in aircraft_keys if k not in AIRCRAFT_SEARCH_TERMS]
            if invalid:
                raise SkyLexIngestionError(
                    "Invalid aircraft keys provided",
                    details=f"Invalid: {invalid} | Valid: {list(AIRCRAFT_SEARCH_TERMS.keys())}"
                )
            self.aircraft_keys = aircraft_keys
        else:
            # None diya = saare aircraft types fetch karo
            self.aircraft_keys = list(AIRCRAFT_SEARCH_TERMS.keys())

        # from_date None diya toh default constant use karo
        self.from_date = from_date or AD_FETCH_FROM_DATE
        self.download_pdfs = download_pdfs  # PDF download flag — default False

        # Hash registry — ECFRIngester ki tarah same pattern
        self.registry_path = self.output_dir / "hash_registry.json"
        self.hash_registry = self._load_hash_registry()

        self.logger.info(
            f"FAAAADIngester initialized | "
            f"Aircraft: {self.aircraft_keys} | "
            f"From: {self.from_date} | "
            f"PDFs: {self.download_pdfs}"
        )

    # -------------------------------------------------------------------------
    # Hash Registry
    # -------------------------------------------------------------------------

    def _load_hash_registry(self) -> dict:
        """Load previously stored content hashes from disk."""
        # Disk pe purana registry hai toh load karo — warna fresh start (empty dict)
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

    def _get_json(self, client: httpx.Client, url: str, params: Optional[dict] = None) -> Any:
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
                    params=params,    # Dict automatically URL-encoded ho jaata hai — ?key=val&key2=val2
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()  # 4xx/5xx pe exception raise hoga
                time.sleep(REQUEST_DELAY_SECONDS)  # Rate limiting — 1 req/sec
                return response.json()

            except httpx.HTTPStatusError as e:
                # Server ne clearly refuse kiya — retry se theek nahi hoga
                raise SkyLexIngestionError(
                    f"HTTP error fetching {url}",
                    details=f"Status: {e.response.status_code}"
                )
            except httpx.HTTPError as e:
                # Network level error — retry karo
                self.logger.warning(
                    f"⚠️ Network error | Attempt: {attempt}/{MAX_RETRIES} | {e}"
                )
                if attempt == MAX_RETRIES:
                    raise SkyLexIngestionError(
                        f"Network error after {MAX_RETRIES} retries",
                        details=str(e)
                    )
                time.sleep(2 ** attempt)  # Exponential backoff — 2s, 4s, 8s

    def _download_pdf(self, client: httpx.Client, pdf_url: str) -> Optional[bytes]:
        """
        Download PDF content — non-fatal, returns None if download fails.

        Args:
            client:  httpx client instance
            pdf_url: Direct URL to PDF file

        Returns:
            PDF content as bytes, or None if download fails.
        """
        # 🔕 NON-FATAL DESIGN —
        # PDF download fail hona acceptable hai — metadata Document toh ban chuka hai.
        # Agar raise karte toh ek PDF ki wajah se poora AD skip ho jaata.
        # Warning log karo aur None return karo — caller handle karega.
        try:
            response = client.get(
                pdf_url,
                headers={**HEADERS, "Accept": "application/pdf"},  # PDF accept header
                timeout=60,               # PDFs bade hote hain — zyada timeout
                follow_redirects=True,    # Federal Register PDF links redirect karte hain
            )
            response.raise_for_status()
            time.sleep(REQUEST_DELAY_SECONDS)
            return response.content  # bytes return karo — disk pe likhne ke liye
        except httpx.HTTPStatusError as e:
            self.logger.warning(
                f"⚠️ PDF download failed | URL: {pdf_url} | "
                f"Status: {e.response.status_code}"
            )
            return None
        except httpx.HTTPError as e:
            self.logger.warning(f"⚠️ PDF network error | URL: {pdf_url} | {e}")
            return None

    # -------------------------------------------------------------------------
    # fetch() — required by BaseIngester
    # -------------------------------------------------------------------------

    def fetch(self) -> dict:
        """
        Fetch FAA ADs from Federal Register API — aircraft type by type.

        Strategy:
        - For each aircraft type: paginate through all results
        - Collect metadata + optionally download PDFs
        - Return all raw AD data grouped by aircraft type

        Returns:
            Dict with raw AD data grouped by aircraft key

        Raises:
            SkyLexIngestionError: If fetch fails completely
        """
        # Return structure: {"boeing_787": [ad1, ad2, ...], "airbus_a320": [...]}
        all_raw_data: dict[str, list[dict]] = {}

        with httpx.Client(follow_redirects=True) as client:
            for aircraft_key in self.aircraft_keys:
                search_term = AIRCRAFT_SEARCH_TERMS[aircraft_key]
                self.logger.info(
                    f"📥 Fetching ADs | Aircraft: {aircraft_key} | "
                    f"From: {self.from_date}"
                )

                # Har aircraft ke liye paginated fetch — sab pages collect karo
                ads = self._fetch_aircraft_ads(client, aircraft_key, search_term)
                all_raw_data[aircraft_key] = ads

                self.logger.info(
                    f"✅ {aircraft_key} — {len(ads)} ADs fetched"
                )

        return all_raw_data

    def _fetch_aircraft_ads(
        self,
        client: httpx.Client,
        aircraft_key: str,
        search_term: str,
    ) -> list[dict]:
        """
        Fetch all ADs for one aircraft type — handles pagination.

        Args:
            client:      httpx client instance
            aircraft_key: e.g., "boeing_787"
            search_term: Federal Register search query

        Returns:
            List of raw AD metadata dicts
        """
        all_ads: list[dict] = []
        page = 1
        total_pages: Optional[int] = None  # Pehli response mein pata chalega total pages

        # ─── PAGINATION LOOP ──────────────────────────────────────────────────
        # Federal Register API max 100 results per page deta hai.
        # Ek aircraft ke 200+ ADs ho sakte hain — sab ke liye multiple pages chahiye.
        while True:
            params = {
                "conditions[agencies][]": FAA_AD_AGENCY,       # Sirf FAA ke documents
                "conditions[term]": search_term,                # e.g., "airworthiness directive boeing 787"
                "conditions[publication_date][gte]": self.from_date,  # From date filter
                "per_page": 100,                                # Max allowed per page
                "page": page,
                "order": "newest",                              # Latest ADs pehle
                "fields[]": AD_FIELDS,                          # Sirf zarooori fields maango
            }

            try:
                data = self._get_json(
                    client,
                    f"{FAA_AD_BASE_URL}/documents.json",
                    params=params
                )
            except SkyLexIngestionError as e:
                # Ek page fail hua — yahan tak jo mila woh rakh lo, aage mat jao
                # Poora aircraft skip karne se better hai partial data
                self.logger.warning(
                    f"⚠️ Page {page} fetch failed for {aircraft_key} — stopping | {e}"
                )
                break

            results = data.get("results", [])
            if not results:
                break  # Koi results nahi — pagination khatam

            all_ads.extend(results)

            # Pehli page pe total pages aur count log karo — ek baar hi calculate hota hai
            if total_pages is None:
                total_pages = data.get("total_pages", 1)
                total_count = data.get("count", 0)
                self.logger.info(
                    f"   {aircraft_key}: {total_count} total ADs | "
                    f"{total_pages} pages"
                )

            self.logger.debug(
                f"   Page {page}/{total_pages} — {len(results)} ADs"
            )

            # Last page pe pahunch gaye — loop tod do
            if page >= (total_pages or 1):
                break

            page += 1  # Agla page fetch karo

        return all_ads

    # -------------------------------------------------------------------------
    # parse() — required by BaseIngester
    # -------------------------------------------------------------------------

    def parse(self, raw_data: Any) -> list[Document]:
        """
        Parse raw AD metadata into Document objects.
        Each AD becomes one Document.
        Hash-based change detection skips unchanged ADs.

        Args:
            raw_data: Dict from fetch() — {aircraft_key: [ad_dicts]}

        Returns:
            List of Document objects — one per new/changed AD
        """
        documents: list[Document] = []
        total_skipped = 0

        # PDF download ke liye httpx client chahiye — parse() ke andar bhi use hoga
        with httpx.Client(follow_redirects=True) as client:
            for aircraft_key, ads in raw_data.items():
                self.logger.info(
                    f"🔍 Parsing {len(ads)} ADs | Aircraft: {aircraft_key}"
                )

                # tqdm = progress bar — terminal mein dikhta hai kitne ADs process hue
                # desc = bar ke saath text, unit = har item ko kya bolein
                for ad in tqdm(ads, desc=f"Parsing {aircraft_key}", unit="AD"):
                    doc_num = ad.get("document_number", "unknown")

                    # ─── HASH CHECK ───────────────────────────────────────────
                    # Poori AD dict ko sorted JSON string mein convert karo — deterministic
                    # sort_keys=True zaroori hai warna same dict har baar alag hash de sakti hai
                    meta_str = json.dumps(ad, sort_keys=True)
                    content_hash = self._compute_hash(meta_str)
                    registry_key = f"faa_ad_{doc_num}"

                    if self.hash_registry.get(registry_key) == content_hash:
                        # AD unchanged hai — skip karo, Document mat banao
                        total_skipped += 1
                        continue

                    # ─── CONTENT BUILD KARO ───────────────────────────────────
                    # Federal Register API se structured metadata milta hai — PDF parse nahi karna
                    # Title + Abstract = RAG ke liye sufficient context
                    content_parts = []
                    if ad.get("title"):
                        content_parts.append(f"Title: {ad['title']}")
                    if ad.get("abstract"):
                        content_parts.append(f"Abstract: {ad['abstract']}")

                    content = "\n\n".join(content_parts)
                    if not content.strip():
                        continue  # Koi text nahi — empty AD skip karo

                    # ─── OPTIONAL PDF DOWNLOAD ────────────────────────────────
                    # download_pdfs=True ho toh PDF bhi disk pe save karo
                    # Non-fatal — PDF fail hone pe Document toh banega hi
                    pdf_content: Optional[bytes] = None
                    if self.download_pdfs and ad.get("pdf_url"):
                        pdf_content = self._download_pdf(client, ad["pdf_url"])
                        if pdf_content:
                            self._save_pdf(pdf_content, doc_num, aircraft_key)

                    # ─── DOC ID GENERATE KARO ────────────────────────────────
                    # Document number unique hota hai Federal Register mein
                    # MD5 hash se short unique ID banao
                    doc_id = hashlib.md5(
                        f"FAA_AD_{doc_num}".encode()
                    ).hexdigest()[:12]

                    doc = Document(
                        content=content,
                        source="FAA_AD",
                        doc_id=f"ad-{doc_id}",
                        url=ad.get("html_url", ""),
                        title=ad.get("title", f"AD {doc_num}"),
                        metadata={
                            "document_number": doc_num,
                            "aircraft_key": aircraft_key,              # Kaunsa fleet type
                            "publication_date": ad.get("publication_date", ""),
                            "effective_on": ad.get("effective_on", ""),  # Kab se mandatory hai
                            "pdf_url": ad.get("pdf_url", ""),
                            "docket_ids": ad.get("docket_ids", []),    # FAA docket reference
                            "regulation_type": "AD",
                            "jurisdiction": "USA",
                            "authority": "FAA",
                        },
                    )
                    documents.append(doc)

                    # Memory mein hash update karo — disk pe baad mein save hoga (efficient)
                    self.hash_registry[registry_key] = content_hash

        # Saare aircraft types process ho gaye — ek baar disk pe save karo
        # Loop ke andar save karna har AD pe disk write karega — unnecessary I/O
        self._save_hash_registry()

        self.logger.info(
            f"✅ Parse complete | "
            f"New/changed: {len(documents)} | "
            f"Skipped (no change): {total_skipped}"
        )
        return documents

    # -------------------------------------------------------------------------
    # PDF Save Helper
    # -------------------------------------------------------------------------

    def _save_pdf(
        self,
        pdf_content: bytes,
        doc_num: str,
        aircraft_key: str,
    ) -> None:
        """
        Save downloaded PDF to disk.

        Args:
            pdf_content: PDF bytes
            doc_num:     AD document number
            aircraft_key: Aircraft type key for folder organization
        """
        # 📁 FOLDER STRUCTURE:
        # data/raw/faa_ad/boeing_787/2025-AD-123-45.pdf
        # data/raw/faa_ad/airbus_a320/2025-AD-456-78.pdf
        # Aircraft key se alag alag folders — organized aur searchable
        pdf_dir = self.output_dir / aircraft_key
        pdf_dir.mkdir(parents=True, exist_ok=True)  # Folder nahi hai toh banao
        pdf_path = pdf_dir / f"{doc_num}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)  # "wb" = write binary — PDF bytes ke liye zaroori
        self.logger.debug(f"💾 PDF saved | {pdf_path.name}")