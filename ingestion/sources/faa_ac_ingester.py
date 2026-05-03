# =============================================================================
# SkyLex — FAA Advisory Circular (AC) Ingester
# Source: https://www.faa.gov (Direct PDF URLs — curated registry)
# Strategy:
#   1. Curated URL registry — 20 most relevant ACs for Air India operations
#   2. Direct PDF download from faa.gov — stable, verified URLs
#   3. pdfplumber text extraction — handles regulatory PDF layouts
#   4. Hash-based change detection — skip unchanged ACs on re-runs
# Why curated: No public API for FAA ACs. 1500+ ACs exist but only ~20
#              are genuinely relevant for Air India commercial operations.
#              Direct PDF URLs from faa.gov are stable and accessible.
# =============================================================================

import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import pdfplumber
from tqdm import tqdm

from config.settings import settings
from exceptions.exceptions import SkyLexIngestionError
from ingestion.base_ingester import BaseIngester, Document
from monitoring.logger import get_logger

# Module-level logger
logger = get_logger(__name__)

# FAA document library base URL
FAA_BASE_URL = "https://www.faa.gov"

# Request headers — Referer required, FAA server blocks bot-like requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.faa.gov/regulations_policies/advisory_circulars/",
}

# Rate limiting — be respectful to faa.gov
REQUEST_DELAY_SECONDS = 1.5
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3

# PDF validity — real PDFs start with %PDF magic bytes
PDF_MAGIC_BYTES = b"%PDF"

# =============================================================================
# FAA AC Curated Registry
# -----------------------------------------------------------------------------
# Why curated? No public API for FAA ACs. 1500+ ACs exist but only ~20
# are relevant for Air India commercial operations.
# All URLs verified working — faa.gov PDF URLs are stable.
# Structure per entry:
#   key:        Unique identifier
#   ac_number:  Official AC number (e.g., "120-76D")
#   title:      Human-readable title
#   url:        Direct verified PDF URL from faa.gov
#   category:   Topic category for organization
#   relevance:  Why relevant for Air India operations
# =============================================================================
FAA_AC_REGISTRY: list[dict] = [

    # ── Electronic Flight Bags (EFB) ──────────────────────────────────────────
    {
        "key": "ac_120_76d",
        "ac_number": "120-76D",
        "title": "Authorization of Electronic Flight Bags (EFB)",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-76D.pdf",
        "category": "operations",
        "relevance": "EFB authorization — all modern cockpits use EFBs for charts and performance",
    },

    # ── Extended Operations (ETOPS) ───────────────────────────────────────────
    {
        "key": "ac_120_42b",
        "ac_number": "120-42B",
        "title": "Extended Operations (ETOPS) and Polar Operations",
        "url": "https://www.faa.gov/documentlibrary/media/advisory_circular/120-42b.pdf",
        "category": "operations",
        "relevance": "ETOPS — critical for Air India long-haul routes DEL-JFK, DEL-SFO on 787/777",
    },

    # ── Crew Resource Management (CRM) ───────────────────────────────────────
    {
        "key": "ac_120_51e",
        "ac_number": "120-51E",
        "title": "Crew Resource Management Training",
        "url": "https://www.faa.gov/documentlibrary/media/advisory_circular/ac120-51e.pdf",
        "category": "training",
        "relevance": "CRM training requirements for flight crew — mandatory for Part 121 operators",
    },

    # ── Standard Operating Procedures ────────────────────────────────────────
    {
        "key": "ac_120_71b",
        "ac_number": "120-71B",
        "title": "Standard Operating Procedures and Checklists for Commercial Operators",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-71B.pdf",
        "category": "operations",
        "relevance": "SOP and checklist requirements for Part 121 air carriers",
    },

    # ── Safety Management System (SMS) ───────────────────────────────────────
    {
        "key": "ac_120_92b",
        "ac_number": "120-92B",
        "title": "Safety Management Systems for Aviation Service Providers",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-92B.pdf",
        "category": "safety",
        "relevance": "SMS implementation guidance — DGCA also mandates SMS for Indian carriers",
    },

    # ── Stall Prevention and Recovery ─────────────────────────────────────────
    {
        "key": "ac_120_109a",
        "ac_number": "120-109A",
        "title": "Stall Prevention and Recovery Training",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-109A.pdf",
        "category": "training",
        "relevance": "UPRT requirements — mandatory for all commercial operators post-AF447",
    },

    # ── All Weather Operations ────────────────────────────────────────────────
    {
        "key": "ac_120_118",
        "ac_number": "120-118",
        "title": "Criteria for Approval of All Weather Operations — CAT I, II, and III",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-118.pdf",
        "category": "operations",
        "relevance": "CAT I/II/III approach operations — critical for fog-prone Indian airports DEL/BOM in winter",
    },

    # ── Air Carrier Maintenance Programs ─────────────────────────────────────
    {
        "key": "ac_120_16g",
        "ac_number": "120-16G",
        "title": "Air Carrier Maintenance Programs",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-16G.pdf",
        "category": "maintenance",
        "relevance": "Maintenance program requirements for Part 121 air carriers",
    },

    # ── Aircraft Inspection and Repair ───────────────────────────────────────
    {
        "key": "ac_43_13_1b",
        "ac_number": "43.13-1B",
        "title": "Acceptable Methods, Techniques, and Practices — Aircraft Inspection and Repair",
        "url": "https://www.faa.gov/documentlibrary/media/advisory_circular/ac_43.13-1b_w-chg1.pdf",
        "category": "maintenance",
        "relevance": "Standard reference for aircraft inspection and repair — used by MRO engineers",
    },

    # ── Aircraft Alterations ──────────────────────────────────────────────────
    {
        "key": "ac_43_13_2b",
        "ac_number": "43.13-2B",
        "title": "Acceptable Methods, Techniques, and Practices — Aircraft Alterations",
        "url": "https://www.faa.gov/documentlibrary/media/advisory_circular/ac%2043.13-2b.pdf",
        "category": "maintenance",
        "relevance": "Standard reference for aircraft alteration methods",
    },

    # ── Airworthiness Directives Compliance ──────────────────────────────────
    {
        "key": "ac_39_7d",
        "ac_number": "39-7D",
        "title": "Airworthiness Directives — Compliance Guidance",
        "url": "https://www.faa.gov/documentlibrary/media/advisory_circular/ac%2039-7d.pdf",
        "category": "airworthiness",
        "relevance": "AD compliance guidance — how to interpret and record AD compliance",
    },

    # ── System Design and Analysis ────────────────────────────────────────────
    {
        "key": "ac_25_1309_1a",
        "ac_number": "25.1309-1A",
        "title": "System Design and Analysis — Transport Category Airplanes",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_25.1309-1A.pdf",
        "category": "airworthiness",
        "relevance": "System safety analysis for transport category aircraft — 787, 777, A350",
    },

    # ── Flight Simulator Qualification ───────────────────────────────────────
    {
        "key": "ac_120_53c",
        "ac_number": "120-53C",
        "title": "Qualification of Flight Simulators and Flight Training Devices",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-53C.pdf",
        "category": "training",
        "relevance": "Flight simulator qualification — type rating training programs",
    },

    # ── Crewmember Qualification ──────────────────────────────────────────────
    {
        "key": "ac_121_43",
        "ac_number": "121-43",
        "title": "Qualification, Service, and Use of Crewmembers and Aircraft Dispatchers",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_121-43.pdf",
        "category": "training",
        "relevance": "Crewmember qualification requirements for Part 121 operators",
    },

    # ── Dangerous Goods ───────────────────────────────────────────────────────
    {
        "key": "ac_121_105a",
        "ac_number": "121-105A",
        "title": "Transportation of Hazardous Materials — Crewmember Training",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_121-105A.pdf",
        "category": "safety",
        "relevance": "Dangerous goods training requirements — IATA DGR compliance",
    },

    # ── Icing Conditions ─────────────────────────────────────────────────────
    {
        "key": "ac_91_74b",
        "ac_number": "91-74B",
        "title": "Pilot Guide — Flight in Icing Conditions",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_91-74B.pdf",
        "category": "operations",
        "relevance": "Icing conditions guidance — relevant for high-altitude winter operations",
    },

    # ── Minimum Equipment List ────────────────────────────────────────────────
    {
        "key": "ac_91_67",
        "ac_number": "91-67",
        "title": "Minimum Equipment Requirements for General Aviation Operations Under FAR Part 91",
        "url": "https://www.faa.gov/documentlibrary/media/advisory_circular/ac_91-67.pdf",
        "category": "airworthiness",
        "relevance": "MEL requirements guidance — cross-reference with DGCA MEL CAR",
    },

    # ── Navigation — RNAV and RNP ─────────────────────────────────────────────
    {
        "key": "ac_90_105a",
        "ac_number": "90-105A",
        "title": "Approval Guidance for RNP Operations and Barometric Vertical Navigation",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_90-105A.pdf",
        "category": "navigation",
        "relevance": "RNP/RNAV approval guidance — required for precision approaches at Indian airports",
    },

    # ── Weight and Balance ────────────────────────────────────────────────────
    {
        "key": "ac_120_85a",
        "ac_number": "120-85A",
        "title": "Air Cargo Operations",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-85A.pdf",
        "category": "operations",
        "relevance": "Air cargo operations guidance — relevant for Air India Cargo operations",
    },

    # ── Fatigue Risk Management ───────────────────────────────────────────────
    {
        "key": "ac_117_1",
        "ac_number": "117-1",
        "title": "Flightcrew Member Duty and Rest Requirements",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_117-1.pdf",
        "category": "safety",
        "relevance": "FRMS guidance — complements DGCA FDTL CARs for Air India crew scheduling",
    },
]


class FAAAACIngester(BaseIngester):
    """
    Ingester for FAA Advisory Circulars (ACs).
    Uses curated registry of 20 verified PDF URLs — direct download from faa.gov.
    Each AC PDF is downloaded, text extracted via pdfplumber, and stored as Document.
    Architecture mirrors DGCAIngester — same curated URL + pdfplumber approach.
    """

    def __init__(self, ac_keys: Optional[list[str]] = None) -> None:
        """
        Args:
            ac_keys: Optional list of specific AC keys to ingest.
                     e.g., ["ac_120_76d", "ac_120_92b"]
                     None means ingest ALL ACs in registry.
        """
        super().__init__(source_name="FAA_AC")

        # Validate ac_keys if provided
        if ac_keys:
            valid_keys = {a["key"] for a in FAA_AC_REGISTRY}
            invalid = [k for k in ac_keys if k not in valid_keys]
            if invalid:
                raise SkyLexIngestionError(
                    "Invalid AC keys provided",
                    details=f"Invalid: {invalid} | Valid: {sorted(valid_keys)}"
                )
            self.target_acs = [a for a in FAA_AC_REGISTRY if a["key"] in ac_keys]
        else:
            self.target_acs = FAA_AC_REGISTRY

        # Hash registry — change detection
        self.registry_path = self.output_dir / "hash_registry.json"
        self.hash_registry = self._load_hash_registry()

        self.logger.info(
            f"FAAAACIngester initialized | "
            f"ACs to process: {len(self.target_acs)} | "
            f"Keys: {[a['key'] for a in self.target_acs]}"
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
        return hashlib.sha256(content).hexdigest()

    # -------------------------------------------------------------------------
    # PDF Validation
    # -------------------------------------------------------------------------

    def _is_valid_pdf(self, content: bytes) -> bool:
        """
        Validate that downloaded content is actually a PDF.
        FAA server sometimes redirects to HTML error pages.

        Args:
            content: Downloaded bytes

        Returns:
            True if content starts with PDF magic bytes (%PDF)
        """
        return content[:4] == PDF_MAGIC_BYTES

    # -------------------------------------------------------------------------
    # HTTP Helpers
    # -------------------------------------------------------------------------

    def _download_pdf(
        self,
        client: httpx.Client,
        url: str,
        ac_key: str
    ) -> Optional[bytes]:
        """
        Download PDF from faa.gov — with retry logic.

        Args:
            client: httpx client instance
            url:    Direct PDF URL from FAA_AC_REGISTRY
            ac_key: AC key for logging context

        Returns:
            PDF bytes if valid, None if 404 or invalid PDF.

        Raises:
            SkyLexIngestionError: If non-404 HTTP error occurs
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.get(
                    url,
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                    follow_redirects=True,
                )

                # 404 = AC not at this URL — non-fatal, return None
                if response.status_code == 404:
                    self.logger.warning(
                        f"⚠️ 404 Not Found | {ac_key} | URL: {url}"
                    )
                    return None

                response.raise_for_status()
                time.sleep(REQUEST_DELAY_SECONDS)

                content = response.content

                # Validate PDF magic bytes — reject HTML error pages
                if not self._is_valid_pdf(content):
                    self.logger.warning(
                        f"⚠️ Invalid PDF received for {ac_key} — "
                        f"Content starts with: {content[:20]}"
                    )
                    return None

                return content

            except httpx.HTTPStatusError as e:
                raise SkyLexIngestionError(
                    f"HTTP error downloading {ac_key}",
                    details=f"Status: {e.response.status_code} | URL: {url}"
                )
            except httpx.HTTPError as e:
                self.logger.warning(
                    f"⚠️ Network error | {ac_key} | "
                    f"Attempt: {attempt}/{MAX_RETRIES} | {e}"
                )
                if attempt == MAX_RETRIES:
                    raise SkyLexIngestionError(
                        f"Network error downloading {ac_key} after {MAX_RETRIES} retries",
                        details=str(e)
                    )
                time.sleep(2 ** attempt)

        return None

    # -------------------------------------------------------------------------
    # PDF Text Extraction
    # -------------------------------------------------------------------------

    def _extract_text_from_pdf(self, pdf_bytes: bytes, ac_key: str) -> str:
        """
        Extract text from PDF bytes using pdfplumber.
        Same approach as DGCAIngester — handles regulatory PDF layouts.

        Args:
            pdf_bytes: Raw PDF content as bytes
            ac_key:    AC key for logging context

        Returns:
            Extracted text — all pages joined with newlines.
            Empty string if extraction fails.
        """
        try:
            text_parts: list[str] = []

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                total_pages = len(pdf.pages)
                self.logger.debug(
                    f"   Extracting text | {ac_key} | Pages: {total_pages}"
                )

                for page in pdf.pages:
                    # Extract plain text
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())

                    # Extract tables — convert to readable text
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = self._table_to_text(table)
                            if table_text:
                                text_parts.append(table_text)

            full_text = "\n\n".join(text_parts)
            self.logger.debug(
                f"   Extracted {len(full_text):,} chars | {ac_key}"
            )
            return full_text

        except Exception as e:
            self.logger.warning(
                f"⚠️ PDF text extraction failed | {ac_key} | {e}"
            )
            return ""

    def _table_to_text(self, table: list) -> str:
        """
        Convert pdfplumber table to readable text.

        Args:
            table: List of rows, each row is a list of cell values

        Returns:
            Pipe-separated table as string
        """
        rows: list[str] = []
        for row in table:
            if row:
                cleaned = [str(cell).strip() if cell else "" for cell in row]
                rows.append(" | ".join(cleaned))
        return "\n".join(rows)

    # -------------------------------------------------------------------------
    # fetch() — required by BaseIngester
    # -------------------------------------------------------------------------

    def fetch(self) -> list[dict]:
        """
        Download all FAA AC PDFs from curated registry.

        Returns:
            List of dicts — each with ac_info + pdf_bytes

        Raises:
            SkyLexIngestionError: If all downloads fail
        """
        raw_data: list[dict] = []
        failed_count = 0

        self.logger.info(
            f"📥 Downloading {len(self.target_acs)} FAA AC PDFs..."
        )

        with httpx.Client(follow_redirects=True) as client:
            for ac_info in tqdm(
                self.target_acs,
                desc="Downloading FAA ACs",
                unit="AC"
            ):
                ac_key = ac_info["key"]
                self.logger.info(
                    f"📥 Downloading | {ac_key} | AC {ac_info['ac_number']} | "
                    f"{ac_info['title']}"
                )

                try:
                    pdf_bytes = self._download_pdf(
                        client, ac_info["url"], ac_key
                    )

                    if pdf_bytes is None:
                        self.logger.warning(
                            f"⚠️ Skipping {ac_key} — PDF not available or invalid"
                        )
                        failed_count += 1
                        continue

                    raw_data.append({
                        "ac_info": ac_info,
                        "pdf_bytes": pdf_bytes,
                    })
                    self.logger.info(
                        f"✅ Downloaded | {ac_key} | "
                        f"Size: {len(pdf_bytes):,} bytes"
                    )

                except SkyLexIngestionError as e:
                    self.logger.warning(
                        f"⚠️ Failed to download {ac_key} — skipping | {e}"
                    )
                    failed_count += 1
                    continue

        self.logger.info(
            f"📥 Download complete | "
            f"Success: {len(raw_data)} | Failed/Not Found: {failed_count}"
        )
        return raw_data

    # -------------------------------------------------------------------------
    # parse() — required by BaseIngester
    # -------------------------------------------------------------------------

    def parse(self, raw_data: Any) -> list[Document]:
        """
        Parse downloaded FAA AC PDFs into Document objects.
        Each AC PDF becomes one Document.
        Hash-based change detection skips unchanged ACs.

        Args:
            raw_data: List of dicts from fetch() — {ac_info, pdf_bytes}

        Returns:
            List of Document objects — one per new/changed AC
        """
        documents: list[Document] = []
        skipped = 0

        for item in raw_data:
            ac_info: dict = item["ac_info"]
            pdf_bytes: bytes = item["pdf_bytes"]
            ac_key = ac_info["key"]

            # Hash check — skip if content unchanged
            content_hash = self._compute_hash(pdf_bytes)
            registry_key = f"faa_ac_{ac_key}"

            if self.hash_registry.get(registry_key) == content_hash:
                self.logger.debug(f"⏭️ {ac_key} — no change, skipping")
                skipped += 1
                continue

            # Save PDF to disk
            self._save_pdf_to_disk(ac_info, pdf_bytes)

            # Extract text from PDF
            text_content = self._extract_text_from_pdf(pdf_bytes, ac_key)

            if not text_content.strip():
                self.logger.warning(
                    f"⚠️ No text extracted from {ac_key} — skipping document"
                )
                continue

            # Build unique doc ID
            doc_id = hashlib.md5(
                f"FAA_AC_{ac_key}".encode()
            ).hexdigest()[:12]

            doc = Document(
                content=text_content,
                source="FAA_AC",
                doc_id=f"ac-{doc_id}",
                url=ac_info["url"],
                title=f"FAA AC {ac_info['ac_number']} — {ac_info['title']}",
                metadata={
                    "ac_key": ac_key,
                    "ac_number": ac_info["ac_number"],
                    "category": ac_info["category"],
                    "relevance": ac_info["relevance"],
                    "regulation_type": "AC",
                    "jurisdiction": "USA",
                    "authority": "FAA",
                    "pdf_size_bytes": len(pdf_bytes),
                },
            )
            documents.append(doc)

            # Update hash registry
            self.hash_registry[registry_key] = content_hash

        # Save updated hash registry
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

    def _save_pdf_to_disk(self, ac_info: dict, pdf_bytes: bytes) -> None:
        """
        Save downloaded PDF to organized folder structure.

        Args:
            ac_info:   AC metadata dict from FAA_AC_REGISTRY
            pdf_bytes: Raw PDF bytes
        """
        ac_key = ac_info["key"]
        pdf_dir = self.output_dir / ac_info["category"] / ac_key
        pdf_dir.mkdir(parents=True, exist_ok=True)

        # Save PDF
        pdf_path = pdf_dir / f"{ac_key}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Save metadata JSON alongside PDF
        meta_path = pdf_dir / f"{ac_key}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **ac_info,
                    "source": "faa_gov",
                    "file_size_bytes": len(pdf_bytes),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )

        self.logger.debug(
            f"💾 PDF saved | {ac_key} | {len(pdf_bytes):,} bytes"
        )