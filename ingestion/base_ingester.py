# =============================================================================
# SkyLex — Abstract Base Ingester
# Defines the common interface and shared functionality for all data ingesters.
# Every source-specific ingester (eCFR, FAA AD, DGCA, etc.) inherits from this.
# Pattern: Template Method — base defines the flow, subclasses fill the steps.
# =============================================================================
# 📌 YE FILE KYA KARTI HAI?
# SkyLex kai jagahon se data ingest karta hai — FAA CFR, FAA ADs, DGCA CARs, etc.
# Har source ka apna alag logic hoga (koi API call, koi PDF download, koi scraping),
# lekin kuch cheezein SABI sources mein common hain:
#   - Documents JSON mein save karna
#   - Timing aur logging karna
#   - MLflow mein metrics log karna
#
# YE FILE WO COMMON LOGIC define karti hai — ek baar yahan likho, sab use karein.
#
# TEMPLATE METHOD PATTERN:
# BaseIngester.run() pura flow define karta hai: fetch → parse → save
# Subclasses sirf fetch() aur parse() implement karte hain apne source ke liye.
# run() aur save() inherited milte hain — dobara likhne ki zaroorat nahi.
# =============================================================================

import json
import time
from abc import ABC, abstractmethod          # Abstract class banane ke liye
from dataclasses import dataclass, field, asdict  # Clean data structure + JSON conversion
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone      # UTC timestamps ke liye

from config.settings import settings
from monitoring.logger import get_logger
from monitoring.mlflow_tracker import SkyLexMLflowTracker
from exceptions.exceptions import SkyLexIngestionError

# Is module ka named logger — logs mein "ingestion.base_ingester" dikhega
logger = get_logger(__name__)


# =============================================================================
# Document Dataclass — Standard structure for all ingested documents
# Every ingester returns a list of Document objects — no exceptions.
# =============================================================================

@dataclass
class Document:
    """
    Standard document structure used across the entire SkyLex pipeline.
    All ingesters, chunkers, and vector stores work with this structure.

    Attributes:
        content:     The actual text content of the document.
        source:      Source identifier e.g. "FAA_CFR", "FAA_AD", "DGCA_CAR".
        doc_id:      Unique identifier for this document.
        url:         Original URL or file path where document was fetched from.
        title:       Document title or heading.
        metadata:    Flexible dict for source-specific fields (part, section, date etc.)
        ingested_at: UTC timestamp when document was ingested.
    """
    # 📦 YE DATACLASS KYA HAI?
    # Poore SkyLex pipeline mein ek common "currency" hai — Document.
    # Ingester Document banata hai → Chunker Document leta hai → VectorStore Document store karta hai.
    # Sab ek hi structure use karte hain toh koi confusion nahi.

    content: str       # Document ka actual text — yahi RAG mein retrieve hoga
    source: str        # Kahan se aaya — "FAA_CFR", "FAA_AD", "DGCA_CAR"
    doc_id: str        # Unique ID — duplicates detect karne ke liye useful

    url: str = ""      # Original URL ya file path — traceability ke liye
    title: str = ""    # Document ka heading — display aur search ke liye

    # dict isliye — har source ke apne alag fields hote hain
    # (CFR mein "part" aur "section", AD mein "effective_date" aur "aircraft_type")
    # field(default_factory=dict) isliye ki mutable default value safe ho
    metadata: dict = field(default_factory=dict)

    # UTC timestamp automatically set hota hai jab Document object banta hai
    # lambda isliye use kiya — agar directly datetime.now() likhte toh
    # class define hote waqt ek baar evaluate hota, har instance ke liye nahi
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# =============================================================================
# Abstract Base Ingester
# =============================================================================

class BaseIngester(ABC):
    """
    Abstract base class for all SkyLex data ingesters.

    Defines the Template Method pattern:
        run() → fetch() → parse() → save()

    Subclasses MUST implement:
        - fetch(): Pull raw data from the source
        - parse(): Convert raw data into Document objects

    Subclasses INHERIT for free:
        - save(): Save documents to data/raw/ as JSON
        - run(): Orchestrate the full ingestion with logging + MLflow tracking
    """
    # 💡 ABC (Abstract Base Class) KYA HOTA HAI?
    # ABC se inherit karne ke baad, agar koi subclass @abstractmethod wale
    # methods implement nahi karta, toh Python INSTANTIATION PE hi error dega.
    # Matlab galti compile time pe pakdi jaayegi, runtime pe nahi.

    def __init__(self, source_name: str) -> None:
        """
        Args:
            source_name: Identifier for this source e.g. "FAA_CFR", "FAA_AD".
        """
        self.source_name = source_name

        # Har source ka apna named logger — "ingestion.faa_cfr", "ingestion.dgca" etc.
        # Isse logs filter karna aasaan hota hai — sirf ek source ke logs dekhne ho toh
        self.logger = get_logger(f"ingestion.{source_name.lower()}")

        # Output directory — data/raw/faa_cfr/, data/raw/dgca/ etc.
        # parents=True matlab parent folders bhi banao agar nahi hain
        # exist_ok=True matlab already hai toh error mat do
        self.output_dir: Path = settings.data_raw_dir / source_name.lower()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self) -> Any:
        """
        Fetch raw data from the source.
        Each ingester implements this differently:
            - eCFR: HTTP GET to eCFR API
            - FAA AD: scrape DRS portal
            - DGCA: download PDFs from portal

        Returns:
            Raw data in source-specific format (XML, PDF bytes, HTML, etc.)

        Raises:
            SkyLexIngestionError: If fetch fails after retries.
        """
        # @abstractmethod hone ki wajah se ye method subclass mein
        # ZAROOR implement karna padega — warna Python object banate waqt error dega.
        # Yahan `pass` sirf isliye hai kyunki base class mein koi logic nahi hota.
        pass

    @abstractmethod
    def parse(self, raw_data: Any) -> list[Document]:
        """
        Parse raw data into standardized Document objects.

        Args:
            raw_data: Output from fetch() — format depends on source.

        Returns:
            List of Document objects ready for chunking and embedding.

        Raises:
            SkyLexIngestionError: If parsing fails.
        """
        # fetch() ka output har source ke liye alag hoga:
        # eCFR → XML string, DGCA → PDF bytes, FAA AD → HTML
        # Har subclass apne format ko Document list mein convert karna jaanta hai.
        pass

    def save(self, documents: list[Document]) -> Path:
        """
        Save ingested documents to data/raw/<source_name>/ as JSON.
        This method is SHARED across all ingesters — no need to override.

        Args:
            documents: List of Document objects from parse().

        Returns:
            Path: Path to the saved JSON file.

        Raises:
            SkyLexIngestionError: If save fails.
        """
        try:
            # 🕐 TIMESTAMP-BASED FILENAME —
            # Har run ka alag file banta hai — "faa_cfr_20250115_143201.json"
            # Isse purane runs overwrite nahi hote, versioning milti hai free mein
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{self.source_name.lower()}_{timestamp}.json"

            # 📄 JSON STRUCTURE BANAO —
            # asdict() dataclass ko dict mein convert karta hai — JSON serialization ke liye
            # ensure_ascii=False — Hindi/special characters JSON mein escape nahi honge
            # indent=2 — human-readable formatting
            data = {
                "source": self.source_name,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "document_count": len(documents),          # Quick count — file khole bina pata chale
                "documents": [asdict(doc) for doc in documents],  # Har Document ko dict mein convert karo
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(
                f"💾 Documents saved | "
                f"Source: {self.source_name} | "
                f"Count: {len(documents)} | "
                f"File: {output_file.name}"
            )
            return output_file  # Caller ko path return karo — MLflow mein log karne ke liye

        except Exception as e:
            # Save fail hua — SkyLexIngestionError mein wrap karo
            # details mein original error daal do — debugging ke liye
            raise SkyLexIngestionError(
                f"Failed to save documents for {self.source_name}",
                details=str(e)
            )

    def run(self) -> list[Document]:
        """
        Orchestrate the full ingestion pipeline: fetch → parse → save.
        Handles logging, timing, and MLflow tracking automatically.
        Subclasses do NOT need to override this method.

        Returns:
            List of Document objects that were ingested and saved.

        Raises:
            SkyLexIngestionError: If any step fails.
        """
        # 🎯 YE TEMPLATE METHOD HAI —
        # Subclass sirf fetch() aur parse() likhta hai.
        # Timing, logging, MLflow tracking — sab yahan automatically handle hota hai.
        # Ek baar yahan likha, har ingester ko free mein milta hai.

        self.logger.info(f"🚀 Ingestion started | Source: {self.source_name}")
        start_time = time.time()  # Timer shuru — total ingestion time calculate karni hai

        # MLflow run shuru karo — with block khatam hote hi auto-end hoga
        with SkyLexMLflowTracker(run_name=f"ingestion_{self.source_name.lower()}") as tracker:
            try:
                # ─── STEP 1: FETCH ───────────────────────────────────────────
                # Subclass ka fetch() call karo — raw data milega
                self.logger.info(f"📥 Fetching data | Source: {self.source_name}")
                raw_data = self.fetch()

                # ─── STEP 2: PARSE ───────────────────────────────────────────
                # Raw data ko Document list mein convert karo
                self.logger.info(f"🔍 Parsing data | Source: {self.source_name}")
                documents = self.parse(raw_data)

                # ─── STEP 3: SAVE ────────────────────────────────────────────
                # Documents JSON file mein save karo
                output_file = self.save(documents)

                # ⏱️ Total time calculate karo — 2 decimal places tak
                elapsed = round(time.time() - start_time, 2)

                # 📊 MLFLOW MEIN LOG KARO —
                # Params = config values (kya ingest kiya, kahan save hua)
                # Metrics = numeric results (kitne documents, kitna time laga)
                tracker.log_params({
                    "source": self.source_name,
                    "output_file": output_file.name,
                })
                tracker.log_metrics({
                    "document_count": len(documents),
                    "ingestion_time_seconds": elapsed,
                })

                self.logger.info(
                    f"✅ Ingestion complete | "
                    f"Source: {self.source_name} | "
                    f"Documents: {len(documents)} | "
                    f"Time: {elapsed}s"
                )
                return documents

            except SkyLexIngestionError:
                # Already logged by exception class
                # 🔕 DOBARA LOG NAHI KARTE —
                # SkyLexIngestionError raise hote hi SkyLexBaseError.__init__ mein
                # logger.error() already call ho chuka hai.
                # Sirf re-raise karo taaki caller tak error pahunche
                raise

            except Exception as e:
                # ⚠️ UNEXPECTED ERROR —
                # fetch() ya parse() mein koi aisi exception aayi jo anticipate nahi ki thi.
                # Use SkyLexIngestionError mein wrap karo — consistent error type ke liye —
                # aur details mein original error message daal do
                raise SkyLexIngestionError(
                    f"Unexpected error during ingestion of {self.source_name}",
                    details=str(e)
                )