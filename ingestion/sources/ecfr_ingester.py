# =============================================================================
# SkyLex — eCFR (Electronic Code of Federal Regulations) Ingester
# Source: https://www.ecfr.gov (REST API — free, no auth required)
# Strategy:
#   1. Fetch latest published date from titles.json — never use today's date
#   2. Fetch Title 14 structure — get all part numbers
#   3. Fetch each part individually using ?part= param
#   4. Hash-based change detection — skip unchanged parts
# =============================================================================
# 📌 YE FILE KYA KARTI HAI?
# Pehle wale ECFRIngester ka improved version hai — do bade upgrades hain:
#
# UPGRADE 1 — SMARTER FETCH STRATEGY:
#   Pehle: Ek saath poora Title 14 XML download (bahut bada, slow)
#   Ab:    Pehle structure fetch karo → har Part alag alag fetch karo (?part= param)
#          Isse ek Part fail ho toh baaki continue hote hain
#
# UPGRADE 2 — HASH-BASED CHANGE DETECTION:
#   Har Part ka SHA-256 hash disk pe store hota hai (hash_registry.json)
#   Re-run pe: agar hash same hai → Part skip, agar alag hai → re-ingest
#   Isse unnecessary re-processing band hoti hai — sirf changed Parts process hote hain
#
# FLOW:
#   fetch() → titles.json (latest date) → structure.json (all parts) → har part ka XML
#   parse() → hash check → XML parse → Document object → hash registry update
# =============================================================================

import hashlib      # SHA-256 hashes change detection ke liye, MD5 doc_id ke liye
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx        # requests ki jagah httpx — better timeout handling, follow_redirects support

from config.settings import settings
from exceptions.exceptions import SkyLexIngestionError
from ingestion.base_ingester import BaseIngester, Document
from monitoring.logger import get_logger

# Is module ka named logger — logs mein "ingestion.ecfr_ingester" dikhega
logger = get_logger(__name__)

# ─── eCFR API CONFIGURATION ──────────────────────────────────────────────────
ECFR_BASE_URL = "https://www.ecfr.gov"
ECFR_TITLE = 14     # Title 14 = Aeronautics and Space (integer — API mein number field hai)

# ─── REQUEST HEADERS ─────────────────────────────────────────────────────────
# User-Agent se server ko pata chalta hai kaun call kar raha hai
# Public APIs ke liye ye good practice hai — polite identification
HEADERS = {
    "User-Agent": "SkyLex-ResearchBot/1.0 (Aviation Compliance RAG System)",
    "Accept": "application/json",
}

# ─── RATE LIMITING & RETRY CONFIG ────────────────────────────────────────────
REQUEST_DELAY_SECONDS = 1.0  # Har successful request ke baad 1 second ruko — server overwhelm na ho
REQUEST_TIMEOUT = 60          # Ek request ke liye max wait time
MAX_RETRIES = 3               # Network errors pe itni baar retry karo


class ECFRIngester(BaseIngester):
    """
    Ingester for FAA 14 CFR (Code of Federal Regulations Title 14).

    Strategy:
    - Fetches latest published date from eCFR titles.json API
    - Fetches Title 14 structure to get all part numbers
    - Fetches each part individually — avoids huge single XML download
    - Hash-based change detection — skips unchanged parts on re-runs
    """

    def __init__(self, parts: Optional[list[str]] = None) -> None:
        """
        Args:
            parts: Optional list of specific CFR part numbers to ingest.
                   e.g., ["91", "121", "25"] — if None, ingests ALL parts.
                   Note: strings, not ints — e.g., "91" not 91
        """
        super().__init__(source_name="FAA_CFR")
        self.parts = parts  # None = saare parts, list = sirf ye parts (testing ke liye useful)

        # Hash registry disk pe store hota hai — program restart ke baad bhi persist karta hai
        self.registry_path = self.output_dir / "hash_registry.json"
        self.hash_registry = self._load_hash_registry()  # Startup pe purana registry load karo

        self.logger.info(
            f"ECFRIngester initialized | "
            f"Parts filter: {parts if parts else 'ALL'}"
        )

    # -------------------------------------------------------------------------
    # Hash Registry — change detection
    # -------------------------------------------------------------------------

    def _load_hash_registry(self) -> dict:
        """Load previously stored content hashes from disk."""
        # 📂 Agar file exist karti hai toh load karo — warna empty dict return karo
        # (pehli baar run ho raha hai — koi purana registry nahi hoga)
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_hash_registry(self) -> None:
        """Save updated hash registry to disk."""
        # 💾 Parse complete hone ke baad updated hashes disk pe likhte hain
        # Taaki next run mein in hashes se compare kar sakein
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.hash_registry, f, indent=2)

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content for change detection."""
        # SHA-256 isliye — MD5 se zyada collision-resistant
        # encode("utf-8") zaroori hai — hashlib bytes accept karta hai, string nahi
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    # HTTP helpers
    # -------------------------------------------------------------------------

    def _get_json(self, client: httpx.Client, url: str) -> Any:
        """
        GET request returning JSON — with retry logic.

        Args:
            client: httpx client instance
            url: Full URL to fetch

        Returns:
            Parsed JSON response

        Raises:
            SkyLexIngestionError: If request fails after MAX_RETRIES
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.get(
                    url,
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()  # 4xx/5xx pe HTTPStatusError raise hoga
                time.sleep(REQUEST_DELAY_SECONDS)  # Rate limiting — server ko breathe karne do
                return response.json()  # Parsed dict/list return karo

            except httpx.HTTPStatusError as e:
                # 4xx/5xx = server ne clearly refuse kiya — retry se theek nahi hoga
                raise SkyLexIngestionError(
                    f"HTTP error fetching {url}",
                    details=f"Status: {e.response.status_code}"
                )
            except httpx.HTTPError as e:
                # Network level error — timeout, DNS fail, connection reset etc.
                # Ye retry se theek ho sakta hai
                self.logger.warning(
                    f"⚠️ Network error | Attempt: {attempt}/{MAX_RETRIES} | {e}"
                )
                if attempt == MAX_RETRIES:
                    # Saare retries khatam — ab raise karo
                    raise SkyLexIngestionError(
                        f"Network error fetching {url} after {MAX_RETRIES} retries",
                        details=str(e)
                    )
                # EXPONENTIAL BACKOFF — 2^1=2s, 2^2=4s, 2^3=8s
                # Jitni baar fail ho, utna zyada wait karo — server recover karne ka time do
                time.sleep(2 ** attempt)

    def _get_xml(self, client: httpx.Client, url: str, params: dict) -> Optional[str]:

        """
        GET request returning XML content — with retry logic.

        Args:
            client: httpx client instance
            url: Full URL to fetch
            params: Query parameters e.g. {"part": "91"}

        Returns:
            XML content as string

        Raises:
            SkyLexIngestionError: If request fails after MAX_RETRIES
        """
        # JSON headers ko override karke XML maango
        # **HEADERS = existing headers unpack karo, phir Accept override karo
        xml_headers = {**HEADERS, "Accept": "application/xml"}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.get(
                    url,
                    headers=xml_headers,
                    params=params,       # {"part": "91"} → URL mein ?part=91 ban jaata hai
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                time.sleep(REQUEST_DELAY_SECONDS)
                return response.text  # XML string return karo (bytes nahi — httpx decode kar deta hai)

            except httpx.HTTPStatusError as e:
                raise SkyLexIngestionError(
                    f"HTTP error fetching XML {url}",
                    details=f"Status: {e.response.status_code} | Params: {params}"
                )
            except httpx.HTTPError as e:
                self.logger.warning(
                    f"⚠️ Network error | Attempt: {attempt}/{MAX_RETRIES} | {e}"
                )
                if attempt == MAX_RETRIES:
                    raise SkyLexIngestionError(
                        f"Network error fetching XML after {MAX_RETRIES} retries",
                        details=str(e)
                    )
                time.sleep(2 ** attempt)  # Exponential backoff — same as _get_json mein
                return None

    # -------------------------------------------------------------------------
    # fetch() — required by BaseIngester
    # -------------------------------------------------------------------------

    def fetch(self) -> dict:
        """
        Fetch eCFR Title 14 data — part by part.

        Strategy:
        1. Get latest published date from titles.json
        2. Get Title 14 structure — extract all part numbers
        3. For each part: fetch XML content

        Returns:
            Dict with issue_date and list of raw part data dicts

        Raises:
            SkyLexIngestionError: If fetch fails
        """
        # httpx.Client context manager — connection pooling aur proper cleanup
        # follow_redirects=True — agar eCFR server redirect kare toh automatically follow karo
        with httpx.Client(follow_redirects=True) as client:

            # ─── STEP 1: LATEST PUBLISHED DATE FETCH KARO ───────────────────
            # Aaj ki date use karna GALAT hoga — eCFR pe sirf published dates valid hain.
            # titles.json mein har title ka "latest_issue_date" field hota hai — woh use karo.
            self.logger.info("📅 Fetching latest eCFR issue date...")
            titles_data = self._get_json(
                client,
                f"{ECFR_BASE_URL}/api/versioner/v1/titles.json"
            )
            # titles list mein se sirf Title 14 ka entry dhundo
            # next() with default None — agar Title 14 na mile toh crash mat karo
            title_info = next(
                (t for t in titles_data["titles"] if t["number"] == ECFR_TITLE),
                None
            )
            if not title_info:
                raise SkyLexIngestionError(
                    f"Title {ECFR_TITLE} not found in eCFR titles list"
                )
            issue_date = title_info["latest_issue_date"]  # e.g., "2025-01-10"
            self.logger.info(f"📅 Latest issue date: {issue_date}")

            # ─── STEP 2: TITLE 14 STRUCTURE FETCH KARO ──────────────────────
            # Structure JSON mein Title 14 ka poora hierarchy hota hai —
            # Chapters → Subchapters → Parts → Subparts → Sections
            # Hume sirf Part level chahiye — _extract_parts() recursively nikaalega
            self.logger.info("🗂️ Fetching Title 14 structure...")
            structure = self._get_json(
                client,
                f"{ECFR_BASE_URL}/api/versioner/v1/structure/{issue_date}/title-{ECFR_TITLE}.json"
            )
            all_parts = self._extract_parts(structure)
            self.logger.info(f"🗂️ Total parts found: {len(all_parts)}")

            # Parts filter apply karo — agar specific parts maange hain toh baaki skip
            if self.parts:
                all_parts = [p for p in all_parts if p["part_number"] in self.parts]
                self.logger.info(f"🔍 After filter: {len(all_parts)} parts")

            # ─── STEP 3: HAR PART KA XML FETCH KARO ─────────────────────────
            # ?part=91 parameter se sirf us ek Part ka XML milta hai — poora Title 14 nahi
            # Isse ek Part fail hone pe baaki Parts continue hote hain
            xml_url = f"{ECFR_BASE_URL}/api/versioner/v1/full/{issue_date}/title-{ECFR_TITLE}.xml"
            raw_parts = []

            for i, part_info in enumerate(all_parts, 1):  # enumerate(start=1) → 1-based counter
                part_num = part_info["part_number"]
                self.logger.info(
                    f"📥 [{i}/{len(all_parts)}] Fetching Part {part_num}: "
                    f"{part_info['part_title']}"
                )
                try:
                    xml_content = self._get_xml(
                        client,
                        xml_url,
                        params={"part": part_num}  # Sirf is part ka XML maango
                    )
                    raw_parts.append({
                        "part_number": part_num,
                        "part_title": part_info["part_title"],
                        "content": xml_content,   # Raw XML string — parse() mein process hoga
                        "issue_date": issue_date,
                    })
                except SkyLexIngestionError as e:
                    # ⚠️ EK PART FAIL = CONTINUE, ABORT NAHI —
                    # Agar Part 91 fail hua toh Part 121, 135 waghera continue karein
                    # Poori run ek Part ki wajah se fail nahi honi chahiye
                    self.logger.warning(f"⚠️ Part {part_num} failed — skipping | {e}")
                    continue

            return {"issue_date": issue_date, "parts": raw_parts}

    # -------------------------------------------------------------------------
    # parse() — required by BaseIngester
    # -------------------------------------------------------------------------

    def parse(self, raw_data: Any) -> list[Document]:
        """
        Parse fetched part data into Document objects.
        Each CFR Part becomes one Document — preserving full XML for chunking later.
        Hash-based change detection skips unchanged parts.

        Args:
            raw_data: Dict from fetch() — {issue_date, parts}

        Returns:
            List of Document objects — one per changed/new CFR part
        """
        # XML parsing yahan import kiya — sirf parse() mein chahiye, top-level nahi
        import xml.etree.ElementTree as ET

        documents = []
        issue_date = raw_data["issue_date"]
        parts = raw_data["parts"]
        skipped = 0  # Counter — kitne parts unchanged the

        for part_data in parts:
            part_num = part_data["part_number"]
            content = part_data["content"]  # Raw XML string

            # ─── HASH CHECK — CHANGE DETECTION ──────────────────────────────
            # Registry key unique hona chahiye — title aur part number dono include karo
            registry_key = f"ecfr_title{ECFR_TITLE}_part{part_num}"
            content_hash = self._compute_hash(content)

            if self.hash_registry.get(registry_key) == content_hash:
                # Hash same hai — content change nahi hua last run ke baad
                # Re-process karna time waste hoga — skip karo
                self.logger.debug(f"⏭️ Part {part_num} — no change, skipping")
                skipped += 1
                continue  # Agla part dekho

            # ─── XML PARSE KARO ──────────────────────────────────────────────
            try:
                root = ET.fromstring(content)  # XML string → Element tree
                text_content = self._extract_text(root)  # Saara text nikalo
                # _get_heading se title nikalo, agar nahi mila toh fetch ka title use karo
                part_title = self._get_heading(root) or part_data["part_title"]
            except ET.ParseError as e:
                # XML malformed — warning log karo, is part ko skip karo
                # Continue isliye — ek corrupt part se poori run fail nahi honi chahiye
                self.logger.warning(f"⚠️ XML parse error for Part {part_num}: {e}")
                continue

            # Empty content wale parts skip karo
            if not text_content.strip():
                continue

            # ─── DOC_ID GENERATE KARO ────────────────────────────────────────
            # Pehle wale ingester se change — ab Part level pe ek Document banta hai
            # (Section level nahi) kyunki chunking baad mein alag se hogi
            doc_id = hashlib.md5(
                f"FAA_CFR_title14_part{part_num}".encode()
            ).hexdigest()[:12]

            doc = Document(
                content=text_content,
                source="FAA_CFR",
                doc_id=f"cfr-{doc_id}",
                url=f"https://www.ecfr.gov/current/title-14/part-{part_num}",
                title=f"14 CFR Part {part_num} — {part_title}",
                metadata={
                    "title_number": str(ECFR_TITLE),  # String mein convert — metadata consistent hone chahiye
                    "part_number": part_num,
                    "part_title": part_title,
                    "issue_date": issue_date,           # Kab ki regulation hai — traceability ke liye
                    "regulation_type": "CFR",
                    "jurisdiction": "USA",
                    "authority": "FAA",
                },
            )
            documents.append(doc)

            # ✅ HASH REGISTRY UPDATE KARO —
            # Abhi memory mein update karo — loop khatam hone ke baad disk pe save hoga
            self.hash_registry[registry_key] = content_hash

        # Saare parts process ho gaye — updated registry disk pe save karo
        # Loop ke andar nahi kiya — har part ke baad disk write expensive hota hai
        self._save_hash_registry()

        self.logger.info(
            f"✅ Parse complete | "
            f"New/changed: {len(documents)} | "
            f"Skipped (no change): {skipped}"
        )
        return documents

    # -------------------------------------------------------------------------
    # XML helpers
    # -------------------------------------------------------------------------

    def _extract_parts(self, structure: dict) -> list[dict]:
        """
        Recursively extract all Part-level nodes from Title structure.

        Args:
            structure: Title structure dict from eCFR API

        Returns:
            List of dicts with part_number and part_title
        """
        parts = []

        def _traverse(node: dict) -> None:
            # 🌳 RECURSIVE TREE WALK —
            # eCFR structure JSON ek nested dict hai — Title → Chapter → Subchapter → Part
            # Hume sirf "part" type nodes chahiye — baaki sab intermediate containers hain
            if node.get("type") == "part":
                parts.append({
                    "part_number": node.get("identifier", ""),  # e.g., "91"
                    "part_title": node.get("label", ""),         # e.g., "General Operating and Flight Rules"
                })
            # Chahe current node part ho ya nahi — children mein bhi dhundo
            # (parts ke andar subparts hote hain — unhe skip karna hai isliye type check zaroori hai)
            for child in node.get("children", []):
                _traverse(child)

        _traverse(structure)  # Root node se shuru karo — baaki recursion handle kar lega
        return parts

    def _get_heading(self, element: Any) -> str:
        """Extract heading text from XML element."""
        # eCFR XML mein HEAD tag hota hai — Part/Section ka title wahan hota hai
        heading = element.find("HEAD")
        if heading is not None and heading.text:
            return heading.text.strip()
        return ""  # HEAD nahi mila — empty string, caller fallback use karega

    def _extract_text(self, element: Any) -> str:
        """Recursively extract all text from XML element."""
        # node.text  = opening tag ke baad ka text — <P>YE TEXT</P>
        # node.tail  = closing tag ke baad ka text — </P>YE TAIL TEXT<P>
        # Dono collect karna zaroori hai — warna kuch content miss ho jaayega
        texts = []
        for node in element.iter():
            if node.text and node.text.strip():
                texts.append(node.text.strip())
            if node.tail and node.tail.strip():
                texts.append(node.tail.strip())
        return "\n".join(texts)