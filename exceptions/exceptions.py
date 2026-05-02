# =============================================================================
# SkyLex — Custom Exception Hierarchy
# All project-specific exceptions are defined here.
# This allows precise error handling and better debugging across all modules.
# Usage: from exceptions.exceptions import SkyLexIngestionError
# =============================================================================
# 📌 YE FILE KYA KARTI HAI?
# Python ke built-in exceptions (ValueError, RuntimeError, etc.) generic hote hain —
# unhe dekh ke pata nahi chalta ki error SkyLex ke kis part mein aayi.
#
# Yahan hum apne CUSTOM exceptions define karte hain jo:
#   1. Clearly batate hain ERROR KAHAAN AAYI (ingestion? retrieval? embedding?)
#   2. Raise hote hi AUTOMATICALLY LOG ho jaate hain — alag se logger.error() nahi likhna
#   3. Ek common base class se inherit karte hain — toh "any SkyLex error" bhi
#      easily catch kiya ja sakta hai: except SkyLexBaseError
# =============================================================================

from monitoring.logger import get_logger

# Is module ka named logger — log lines mein "exceptions.exceptions" dikhega
logger = get_logger(__name__)


class SkyLexBaseError(Exception):
    """
    Base exception for all SkyLex errors.
    All custom exceptions inherit from this class.
    Automatically logs the error when raised.
    """

    def __init__(self, message: str, details: str = "") -> None:
        """
        Args:
            message: Human-readable error description.
            details: Optional extra context (URL, file path, etc.)
        """
        # Error ka main message store karo — baad mein programmatically access ke liye
        self.message = message

        # Extra context store karo — jaise konsi file fail hui, konsa URL timeout hua
        self.details = details

        # Agar details diye hain toh combined message banao, warna sirf message
        # Example: "Fetch failed | Details: https://faa.gov/timeout"
        full_message = f"{message} | Details: {details}" if details else message

        # Python ke built-in Exception ko initialize karo full_message ke saath
        # Isse str(error) ya traceback mein complete message dikhega
        super().__init__(full_message)

        # 🔥 YE SABSE IMPORTANT PART HAI —
        # Exception raise hote hi automatically log ho jaata hai.
        # self.__class__.__name__ dynamically class ka naam deta hai —
        # matlab agar SkyLexIngestionError raise hua, toh log mein
        # "❌ SkyLexIngestionError: ..." dikhega, na ki "SkyLexBaseError"
        logger.error(f"❌ {self.__class__.__name__}: {full_message}")


# =============================================================================
# Module-Specific Exceptions
# =============================================================================
# 💡 DESIGN PATTERN: Har module ka apna exception class hai.
# Sab SkyLexBaseError se inherit karte hain, toh:
#   - `except SkyLexBaseError` se SAARE SkyLex errors ek saath pakad sakte ho
#   - `except SkyLexIngestionError` se SIRF ingestion errors pakad sakte ho
#
# `pass` isliye likha hai kyunki in classes ko koi extra logic nahi chahiye —
# sirf naam alag hai taaki pata chale error kahan aayi. Baaki sab
# SkyLexBaseError ka __init__ handle kar leta hai (auto-logging sameet).
# =============================================================================

class SkyLexConfigError(SkyLexBaseError):
    """Raised when configuration or settings are invalid or missing."""
    # Example: .env mein OPENAI_API_KEY missing ho, ya log_level invalid ho
    pass


class SkyLexIngestionError(SkyLexBaseError):
    """Raised when data ingestion fails — fetch, parse, or save errors."""
    # Example: FAA website se PDF download fail hua, ya HTML parse nahi hua
    pass


class SkyLexProcessingError(SkyLexBaseError):
    """Raised when document processing or chunking fails."""
    # Example: PDF corrupt hai, ya chunking ke waqt encoding error aayi
    pass


class SkyLexEmbeddingError(SkyLexBaseError):
    """Raised when embedding generation fails."""
    # Example: OpenAI embedding API rate limit hit hua ya connection timeout
    pass


class SkyLexVectorStoreError(SkyLexBaseError):
    """Raised when vector store operations fail — Qdrant or ChromaDB."""
    # Example: Qdrant server down hai, ya collection create karna fail hua
    pass


class SkyLexRetrievalError(SkyLexBaseError):
    """Raised when document retrieval or search fails."""
    # Example: Similarity search mein koi result nahi aaya, ya query malformed thi
    pass


class SkyLexRAGError(SkyLexBaseError):
    """Raised when the agentic RAG pipeline fails."""
    # Example: LLM ne response generate nahi kiya, ya chain execution fail hua
    pass


class SkyLexAPIError(SkyLexBaseError):
    """Raised when FastAPI layer encounters an error."""
    # Example: Invalid request payload aaya, ya downstream service ne error diya
    pass