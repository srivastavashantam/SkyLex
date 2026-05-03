# =============================================================================
# SkyLex — DGCA (Directorate General of Civil Aviation) CAR Ingester
# Source: https://www.dgca.gov.in (Direct PDF URLs — curated registry)
# Strategy:
#   1. Curated URL registry — DGCA has no public API, JS-rendered portal
#   2. Direct PDF download — stable attachId URLs work without scraping
#   3. PDF validity check — DGCA sometimes returns HTML error pages as PDF
#   4. pdfplumber text extraction — handles regulatory PDF layouts
#   5. Hash-based change detection — skip unchanged CARs on re-runs
# Coverage: Air India operations-relevant CARs only
# =============================================================================
# 📌 YE FILE KYA KARTI HAI?
# DGCA CARs (Civil Aviation Requirements) ingest karta hai — India ka aviation
# regulatory framework. FAA ki tarah DGCA ka koi public REST API nahi hai —
# unka portal JavaScript-rendered hai, scraping possible nahi.
#
# SOLUTION — CURATED REGISTRY:
# Maine manually verified PDF URLs ka ek registry banaya hai (DGCA_CAR_REGISTRY).
# Ye stable "attachId" URLs hain jo directly PDF download karte hain bina
# JS rendering ke. Naya CAR add karna ho toh sirf registry mein entry daalo.
#
# PDF KE SAATH EK CHALLENGE:
# DGCA portal kabhi kabhi PDF ki jagah HTML error page return karta hai —
# wahi same Content-Type ke saath. Isliye magic bytes check kiya jata hai:
# Real PDF hamesha "%PDF" se start hota hai.
#
# KEY FEATURES:
#   1. Curated URL registry — har entry mein section, series, part, relevance hai
#   2. PDF magic bytes validation — HTML error pages reject hote hain
#   3. pdfplumber extraction — text + tables dono extract hote hain
#   4. Hash-based change detection — PDF bytes ka SHA-256 compare hota hai
#   5. PDF + metadata JSON dono disk pe save hote hain — audit trail ke liye
# =============================================================================

import hashlib
import io          # PDF bytes ko file-like object mein wrap karne ke liye (pdfplumber ke liye)
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import pdfplumber  # PDF text + table extraction — regulatory PDFs ke liye best
from tqdm import tqdm

from config.settings import settings
from exceptions.exceptions import SkyLexIngestionError
from ingestion.base_ingester import BaseIngester, Document
from monitoring.logger import get_logger

# Is module ka named logger — logs mein "ingestion.dgca_ingester" dikhega
logger = get_logger(__name__)

# ─── DGCA PORTAL CONFIGURATION ───────────────────────────────────────────────
DGCA_BASE_URL = "https://www.dgca.gov.in"

# ─── REQUEST HEADERS ─────────────────────────────────────────────────────────
# Accept mein application/pdf daala — server ko batao hume PDF chahiye
# */* fallback isliye ki kuch URLs redirect karte hain
HEADERS = {
    "User-Agent": "SkyLex-ResearchBot/1.0 (Aviation Compliance RAG System)",
    "Accept": "application/pdf,*/*",
}

# ─── RATE LIMITING ───────────────────────────────────────────────────────────
# Government portal hai — 2 seconds delay rakhna respectful hai
# Zyada aggressive requests se IP block ho sakta hai
REQUEST_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT = 60   # PDFs bade hote hain — zyada timeout
MAX_RETRIES = 3

# ─── PDF VALIDATION ──────────────────────────────────────────────────────────
# Har valid PDF file ke pehle 4 bytes "%PDF" hote hain — ye PDF magic bytes hain.
# DGCA portal kabhi kabhi HTML error page return karta hai same Content-Type ke saath.
# Pehle 4 bytes check karo — HTML se clearly alag hota hai ("%PDF" vs "<htm")
PDF_MAGIC_BYTES = b"%PDF"

# =============================================================================
# DGCA CAR Curated URL Registry
# -----------------------------------------------------------------------------
# Why curated? DGCA portal is JavaScript-rendered — no public API exists.
# PDF URLs use stable attachId parameters — verified accessible 2026-04-27.
# Add new CARs here as they become operationally relevant for Air India.
# Structure per entry:
#   key:       Unique identifier
#   section:   DGCA CAR section number
#   series:    Series letter (A, B, C...)
#   part:      Part number (I, II, III...)
#   title:     Human-readable title
#   url:       Direct PDF download URL
#   relevance: Why this CAR matters for Air India operations
# =============================================================================
DGCA_CAR_REGISTRY: list[dict] = [

    # ── Section 3: Air Transport ──────────────────────────────────────────────
    # VERIFIED: Search results confirm Air Transport Series M content
    {
        "key": "s3_air_transport_m",
        "section": "3",
        "series": "M",
        "part": "VI",
        "title": "Air Transport Series M — Passenger Rights and Operations",
        "url": "https://www.dgca.gov.in/digigov-portal/Upload?flag=iframeAttachView&attachId=tCyTvbxJNxKAIaYounFBfw%3D%3D",
        "relevance": "Air transport operations requirements for scheduled Indian carriers",
    },
    # VERIFIED: Revision 01 June 2025 — scheduled operations
    {
        "key": "s3_scheduled_ops",
        "section": "3",
        "series": "C",
        "part": "I",
        "title": "Requirements for Scheduled Air Transport Operations — Rev 01 Jun 2025",
        "url": "https://www.dgca.gov.in/digigov-portal/Upload?flag=iframeAttachView&attachId=Zt7JQued3L0VAUVkAER6nA%3D%3D",
        "relevance": "Core operating requirements for airlines like Air India — latest revision",
    },

    # ── Section 5: Air Safety ─────────────────────────────────────────────────
    # VERIFIED: Search results confirm Breath Analyzer / Alcohol Testing content
    {
        "key": "s5_alcohol_testing",
        "section": "5",
        "series": "F",
        "part": "III",
        "title": "Breath-Analyzer Examination — Alcohol Testing Procedures",
        "url": "https://www.dgca.gov.in/digigov-portal/Upload?flag=iframeAttachView&attachId=YPPvmAUI/XKRXSX6YN5FfA%3D%3D",
        "relevance": "Safety-critical — crew alcohol testing procedures",
    },

    # ── Section 7: Flight Crew Standards ─────────────────────────────────────
    # VERIFIED: Search results confirm FDTL Flight Crew Rev1 Jan 2024
    {
        "key": "s7_fdtl_flight_crew",
        "section": "7",
        "series": "J",
        "part": "III",
        "title": "Flight Duty Time Limitations (FDTL) — Flight Crew, Scheduled Operations Rev1 Jan 2024",
        "url": "https://www.dgca.gov.in/digigov-portal/Upload?flag=iframeAttachView&attachId=YUKKrHOBzBr0RZyThH2OXw%3D%3D",
        "relevance": "CRITICAL — FDTL rules directly impact Air India crew scheduling. Major revision Jan 2024.",
    },
    # VERIFIED: DGCA Technical Centre document — cabin crew related
    {
        "key": "s7_fdtl_cabin_crew",
        "section": "7",
        "series": "J",
        "part": "I",
        "title": "Flight Duty Time Limitations (FDTL) — Cabin Crew",
        "url": "https://www.dgca.gov.in/digigov-portal/Upload?flag=iframeAttachView&attachId=il6xIB9QE6ehTbKQ7uUHyQ%3D%3D",
        "relevance": "FDTL rules for cabin crew — mandatory for all scheduled operators",
    },
    # VERIFIED: Search results confirm CAR-147 Maintenance Training content
    {
        "key": "s7_maintenance_training",
        "section": "7",
        "series": "H",
        "part": "I",
        "title": "Maintenance Training Organisation (CAR-147) Requirements",
        "url": "https://www.dgca.gov.in/digigov-portal/Upload?flag=iframeAttachView&attachId=I91vHr7SPMfsWLJUhRmOgg%3D%3D",
        "relevance": "Requirements for approved AME training organisations",
    },
]
# 💡 NAYA CAR ADD KARNA HO TOH:
# Upar registry mein ek naya dict entry daalo — baaki sab automatically handle hoga.
# URL DGCA portal se manually copy karna padega (attachId URL dhundo developer tools mein).


class DGCAIngester(BaseIngester):
    """
    Ingester for DGCA Civil Aviation Requirements (CARs).

    Strategy:
    - Curated URL registry — DGCA portal has no public API
    - Direct PDF download via stable attachId URLs
    - PDF validity check — rejects HTML error pages disguised as PDFs
    - pdfplumber text extraction — handles regulatory PDF layouts
    - Hash-based change detection — skips unchanged CARs on re-runs
    """

    def __init__(self, car_keys: Optional[list[str]] = None) -> None:
        """
        Args:
            car_keys: Optional list of specific CAR keys to ingest.
                      e.g., ["s2_mel", "s7_fdtl_flight_crew"]
                      None means ingest ALL CARs in registry.
        """
        super().__init__(source_name="DGCA_CAR")

        # ─── CAR KEYS VALIDATE KARO ───────────────────────────────────────────
        # Set comprehension isliye — O(1) lookup, list se zyada fast
        if car_keys:
            valid_keys = {c["key"] for c in DGCA_CAR_REGISTRY}
            invalid = [k for k in car_keys if k not in valid_keys]
            if invalid:
                # Upfront fail karo — invalid keys ke saath aadha kaam karna worse hoga
                raise SkyLexIngestionError(
                    "Invalid CAR keys provided",
                    details=f"Invalid: {invalid} | Valid: {sorted(valid_keys)}"
                )
            # Sirf requested CARs filter karo registry se
            self.target_cars = [c for c in DGCA_CAR_REGISTRY if c["key"] in car_keys]
        else:
            # None = poori registry ingest karo
            self.target_cars = DGCA_CAR_REGISTRY

        # Hash registry — ECFRIngester aur FAAAADIngester jaisa same pattern
        self.registry_path = self.output_dir / "hash_registry.json"
        self.hash_registry = self._load_hash_registry()

        self.logger.info(
            f"DGCAIngester initialized | "
            f"CARs to process: {len(self.target_cars)} | "
            f"Keys: {[c['key'] for c in self.target_cars]}"
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

    def _compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of PDF bytes for change detection."""
        # DGCA ke liye content bytes hai (string nahi) —
        # PDF binary content ka hash directly compute karo
        # ECFRIngester mein string.encode() kiya tha, yahan bytes direct pass hote hain
        return hashlib.sha256(content).hexdigest()

    # -------------------------------------------------------------------------
    # PDF Validation
    # -------------------------------------------------------------------------

    def _is_valid_pdf(self, content: bytes) -> bool:
        """
        Validate that downloaded content is actually a PDF.
        DGCA portal sometimes returns HTML error pages instead of PDFs.

        Args:
            content: Downloaded bytes

        Returns:
            True if content starts with PDF magic bytes (%PDF)
        """
        # 🔍 MAGIC BYTES CHECK —
        # PDF specification ke according, har valid PDF "%PDF" se start hota hai.
        # HTML page "<htm" ya "<!DO" se start hota hai — clearly alag hai.
        # content[:4] = sirf pehle 4 bytes dekho — fast check, poora file read nahi karna
        return content[:4] == PDF_MAGIC_BYTES

    # -------------------------------------------------------------------------
    # HTTP Helpers
    # -------------------------------------------------------------------------

    def _download_pdf(self, client: httpx.Client, url: str, car_key: str) -> Optional[bytes]:
        """
        Download PDF from DGCA portal — with retry logic.

        Args:
            client:  httpx client instance
            url:     Direct PDF URL from DGCA_CAR_REGISTRY
            car_key: CAR key for logging context

        Returns:
            PDF bytes if valid, None if download fails or invalid PDF.

        Raises:
            SkyLexIngestionError: If HTTP error occurs
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.get(
                    url,
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                    follow_redirects=True,  # DGCA URLs kabhi kabhi redirect karte hain
                )
                response.raise_for_status()
                time.sleep(REQUEST_DELAY_SECONDS)  # Government portal — respectful delay

                content = response.content  # bytes mein save karo

                # ✅ PDF VALIDITY CHECK —
                # Download successful tha, par content actual PDF hai ya HTML error page?
                # Magic bytes se verify karo — invalid ho toh None return karo
                if not self._is_valid_pdf(content):
                    self.logger.warning(
                        f"⚠️ Invalid PDF received for {car_key} — "
                        f"Content starts with: {content[:20]}"  # First 20 bytes log karo — debugging ke liye
                    )
                    return None

                return content  # Valid PDF bytes — caller use karega

            except httpx.HTTPStatusError as e:
                # Server ne clearly refuse kiya — retry se theek nahi hoga
                raise SkyLexIngestionError(
                    f"HTTP error downloading {car_key}",
                    details=f"Status: {e.response.status_code} | URL: {url}"
                )
            except httpx.HTTPError as e:
                # Network level error — retry karo
                self.logger.warning(
                    f"⚠️ Network error | {car_key} | "
                    f"Attempt: {attempt}/{MAX_RETRIES} | {e}"
                )
                if attempt == MAX_RETRIES:
                    raise SkyLexIngestionError(
                        f"Network error downloading {car_key} after {MAX_RETRIES} retries",
                        details=str(e)
                    )
                time.sleep(2 ** attempt)  # Exponential backoff — 2s, 4s, 8s

        return None

    # -------------------------------------------------------------------------
    # PDF Text Extraction
    # -------------------------------------------------------------------------

    def _extract_text_from_pdf(self, pdf_bytes: bytes, car_key: str) -> str:
        """
        Extract text from PDF bytes using pdfplumber.
        Handles multi-page regulatory PDFs with tables and headers.

        Args:
            pdf_bytes: Raw PDF content as bytes
            car_key:   CAR key for logging context

        Returns:
            Extracted text — all pages joined with newlines.
            Empty string if extraction fails.
        """
        try:
            text_parts: list[str] = []

            # io.BytesIO — bytes ko file-like object mein wrap karo
            # pdfplumber file path ya file object accept karta hai — bytes directly nahi
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                total_pages = len(pdf.pages)
                self.logger.debug(
                    f"   Extracting text | {car_key} | Pages: {total_pages}"
                )

                for page in pdf.pages:
                    # ─── PLAIN TEXT EXTRACT ───────────────────────────────────
                    # extract_text() = normal paragraphs, headings, sentences
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())

                    # ─── TABLES EXTRACT ───────────────────────────────────────
                    # Regulatory PDFs mein tables common hain — MEL tables, FDTL tables etc.
                    # extract_tables() = list of tables, har table = list of rows
                    # Inhe readable text mein convert karo
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = self._table_to_text(table)
                            if table_text:
                                text_parts.append(table_text)

            # Saare pages ka text double newline se join karo — clear page separation
            full_text = "\n\n".join(text_parts)
            self.logger.debug(
                f"   Extracted {len(full_text):,} chars | {car_key}"
            )
            return full_text

        except Exception as e:
            # PDF extraction fail hua — warning log karo, empty string return karo
            # Caller check karega empty string — Document skip hoga
            self.logger.warning(
                f"⚠️ PDF text extraction failed | {car_key} | {e}"
            )
            return ""

    def _table_to_text(self, table: list) -> str:
        """
        Convert pdfplumber table (list of lists) to readable text.

        Args:
            table: List of rows, each row is a list of cell values

        Returns:
            Tab-separated table as string
        """
        # pdfplumber table structure:
        # [[col1, col2, col3], [val1, val2, val3], ...]
        # Har row ko " | " se join karo — human-readable aur RAG ke liye parseable
        rows: list[str] = []
        for row in table:
            if row:
                # None cells ko empty string mein convert karo — join ke liye
                cleaned = [str(cell).strip() if cell else "" for cell in row]
                rows.append(" | ".join(cleaned))
        return "\n".join(rows)

    # -------------------------------------------------------------------------
    # fetch() — required by BaseIngester
    # -------------------------------------------------------------------------

    def fetch(self) -> list[dict]:
        """
        Download all DGCA CAR PDFs from curated registry.

        Returns:
            List of dicts — each with car_info + pdf_bytes

        Raises:
            SkyLexIngestionError: If all downloads fail
        """
        raw_data: list[dict] = []
        failed_count = 0  # Counter — kitne CARs download fail hue

        self.logger.info(
            f"📥 Downloading {len(self.target_cars)} DGCA CAR PDFs..."
        )

        with httpx.Client(follow_redirects=True) as client:
            for car_info in tqdm(
                self.target_cars,
                desc="Downloading DGCA CARs",
                unit="CAR"  # Progress bar mein "5 CAR/s" dikhega
            ):
                car_key = car_info["key"]
                self.logger.info(
                    f"📥 Downloading | {car_key} | {car_info['title']}"
                )

                try:
                    pdf_bytes = self._download_pdf(
                        client, car_info["url"], car_key
                    )

                    if pdf_bytes is None:
                        # Download hua par valid PDF nahi tha — skip karo
                        self.logger.warning(
                            f"⚠️ Skipping {car_key} — invalid or empty PDF"
                        )
                        failed_count += 1
                        continue

                    # car_info aur pdf_bytes saath rakh do — parse() mein dono chahiye
                    raw_data.append({
                        "car_info": car_info,
                        "pdf_bytes": pdf_bytes,
                    })
                    self.logger.info(
                        f"✅ Downloaded | {car_key} | "
                        f"Size: {len(pdf_bytes):,} bytes"
                    )

                except SkyLexIngestionError as e:
                    # Ek CAR fail = warning + continue — poori run abort nahi
                    self.logger.warning(
                        f"⚠️ Failed to download {car_key} — skipping | {e}"
                    )
                    failed_count += 1
                    continue

        self.logger.info(
            f"📥 Download complete | "
            f"Success: {len(raw_data)} | Failed: {failed_count}"
        )
        return raw_data

    # -------------------------------------------------------------------------
    # parse() — required by BaseIngester
    # -------------------------------------------------------------------------

    def parse(self, raw_data: Any) -> list[Document]:
        """
        Parse downloaded DGCA CAR PDFs into Document objects.
        Each CAR PDF becomes one Document.
        Saves PDF to disk alongside extracted text.
        Hash-based change detection skips unchanged CARs.

        Args:
            raw_data: List of dicts from fetch() — {car_info, pdf_bytes}

        Returns:
            List of Document objects — one per new/changed CAR
        """
        documents: list[Document] = []
        skipped = 0

        for item in raw_data:
            car_info: dict = item["car_info"]
            pdf_bytes: bytes = item["pdf_bytes"]
            car_key = car_info["key"]

            # ─── HASH CHECK ───────────────────────────────────────────────────
            # PDF bytes ka directly hash karo — string encode nahi karna (already bytes hain)
            # Agar PDF server pe update hua toh bytes alag honge → hash alag → re-ingest
            content_hash = self._compute_hash(pdf_bytes)
            registry_key = f"dgca_car_{car_key}"

            if self.hash_registry.get(registry_key) == content_hash:
                # Same hash = unchanged PDF — skip karo
                self.logger.debug(f"⏭️ {car_key} — no change, skipping")
                skipped += 1
                continue

            # ─── PDF DISK PE SAVE KARO ────────────────────────────────────────
            # Text extract karne se pehle disk pe save karo —
            # Taaki raw PDF always available rahe (debugging + audit ke liye)
            self._save_pdf_to_disk(car_info, pdf_bytes)

            # ─── TEXT EXTRACT KARO ────────────────────────────────────────────
            text_content = self._extract_text_from_pdf(pdf_bytes, car_key)

            if not text_content.strip():
                # PDF download hua, par text extract nahi hua (scanned image PDF?)
                # Warning log karo — Document mat banao (empty content useless hoga RAG mein)
                self.logger.warning(
                    f"⚠️ No text extracted from {car_key} — skipping document"
                )
                continue

            # ─── DOC ID GENERATE KARO ────────────────────────────────────────
            doc_id = hashlib.md5(
                f"DGCA_CAR_{car_key}".encode()
            ).hexdigest()[:12]

            doc = Document(
                content=text_content,
                source="DGCA_CAR",
                doc_id=f"dgca-{doc_id}",
                url=car_info["url"],
                # Title mein section + series + part + title — fully qualified reference
                title=f"DGCA CAR Section {car_info['section']} Series {car_info['series']} Part {car_info['part']} — {car_info['title']}",
                metadata={
                    "car_key": car_key,
                    "section": car_info["section"],    # e.g., "7"
                    "series": car_info["series"],      # e.g., "J"
                    "part": car_info["part"],          # e.g., "III"
                    "relevance": car_info["relevance"], # Why this CAR matters for Air India
                    "regulation_type": "CAR",
                    "jurisdiction": "India",           # FAA se alag — India jurisdiction
                    "authority": "DGCA",
                    "pdf_size_bytes": len(pdf_bytes),  # File size — monitoring ke liye
                },
            )
            documents.append(doc)

            # Memory mein hash update karo — loop end pe disk write hoga (efficient)
            self.hash_registry[registry_key] = content_hash

        # Saare CARs process ho gaye — ek baar disk pe save karo
        self._save_hash_registry()

        self.logger.info(
            f"✅ Parse complete | "
            f"New/changed: {len(documents)} | "
            f"Skipped (no change): {skipped}"
        )
        return documents

    # -------------------------------------------------------------------------
    # PDF Save Helper
    # -------------------------------------------------------------------------

    def _save_pdf_to_disk(self, car_info: dict, pdf_bytes: bytes) -> None:
        """
        Save downloaded PDF to organized folder structure.

        Args:
            car_info:  CAR metadata dict from DGCA_CAR_REGISTRY
            pdf_bytes: Raw PDF bytes
        """
        car_key = car_info["key"]

        # 📁 FOLDER STRUCTURE:
        # data/raw/dgca_car/section_7/s7_fdtl_flight_crew/s7_fdtl_flight_crew.pdf
        # data/raw/dgca_car/section_7/s7_fdtl_flight_crew/s7_fdtl_flight_crew_meta.json
        # Section-wise folders — organized browsing ke liye
        pdf_dir = (
            self.output_dir
            / f"section_{car_info['section']}"  # e.g., section_7
            / car_key                            # e.g., s7_fdtl_flight_crew
        )
        pdf_dir.mkdir(parents=True, exist_ok=True)

        # ─── PDF SAVE ─────────────────────────────────────────────────────────
        pdf_path = pdf_dir / f"{car_key}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)  # "wb" = write binary — PDF bytes ke liye zaroori

        # ─── METADATA JSON SAVE ───────────────────────────────────────────────
        # PDF ke saath metadata JSON bhi rakhte hain —
        # Taaki PDF dekhe bina pata chale yahan kya hai, kab ingest hua, kitna bada hai
        # **car_info = registry ki existing fields unpack karo, upar extra fields add karo
        meta_path = pdf_dir / f"{car_key}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **car_info,                                              # Registry ki saari fields
                    "source": "dgca_gov_in",
                    "file_size_bytes": len(pdf_bytes),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),  # UTC timestamp
                },
                f,
                indent=2,
            )

        self.logger.debug(
            f"💾 PDF saved | {car_key} | {len(pdf_bytes):,} bytes"
        )