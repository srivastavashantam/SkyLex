# =============================================================================
# SkyLex — LangSmith Configuration
# LangSmith traces every LangChain pipeline call — LLM, retrieval, embeddings.
# Records latency, token usage, cost, and errors for every query.
# Usage: call initialize_langsmith() once at application startup.
# =============================================================================
# 📌 YE FILE KYA KARTI HAI?
# Pehli wali langsmith config file ka upgraded version hai ye —
# ab isme API key validation bhi hai aur custom SkyLexConfigError bhi use hota hai.
#
# Naya kya hai pehle wale se:
#   1. API key missing ho toh pehle hi SkyLexConfigError raise karo (fail fast)
#   2. SkyLexConfigError alag se catch hota hai — kyunki wo already log ho chuka hota hai
#   3. Unexpected errors ke liye alag except block — dono cases alag treat hote hain
# =============================================================================

import os
from monitoring.logger import get_logger
from exceptions.exceptions import SkyLexConfigError  # API key missing ho toh ye raise hoga
from config.settings import settings  # Saari config values yahan se aati hain

# Is module ka named logger — logs mein "monitoring.langsmith_config" dikhega
logger = get_logger(__name__)


def initialize_langsmith() -> bool:
    """
    Initialize LangSmith tracing for the SkyLex project.
    Sets required environment variables that LangChain reads automatically.
    Call this function ONCE at application startup.
    Returns:
        bool: True if initialized successfully, False otherwise.
    """
    try:
        # 🛡️ FAIL FAST VALIDATION —
        # Pehle check karo ki API key hai ya nahi.
        # Agar nahi hai toh abhi hi SkyLexConfigError raise karo —
        # ye error automatically log ho jaayega (SkyLexBaseError ka __init__ karta hai),
        # toh alag se logger.error() likhne ki zaroorat nahi
        if not settings.langchain_api_key:
            raise SkyLexConfigError(
                "LangSmith API key is missing",
                details="Set LANGCHAIN_API_KEY in .env file"  # Developer ko exactly batao kya karna hai
            )

        # ✅ Key valid hai — ab environment variables set karo.
        # LangChain internally in 4 variables ko read karta hai automatically.
        # Hume manually kuch "connect" nahi karna — bas ye set karo.

        # bool ko lowercase string mein convert karna zaroori hai — env vars strings hote hain
        os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()

        # Actual API key — LangSmith server authenticate karne ke liye use karta hai
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key

        # Dashboard mein is naam se saare traces ek group mein dikhenge
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

        # LangSmith ka server URL — default: "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

        # Sab kuch set ho gaya — confirmation log karo
        logger.info(
            f"✅ LangSmith initialized | "
            f"Project: {settings.langchain_project} | "
            f"Tracing: {settings.langchain_tracing_v2}"
        )
        return True  # Caller ko batao — initialization successful rahi

    except SkyLexConfigError:
        # Already logged by exception class — just return False
        # 🔕 YAHAN DOBARA LOG NAHI KARTE —
        # SkyLexConfigError raise hote hi SkyLexBaseError.__init__ mein
        # logger.error() already call ho chuka hai.
        # Dobara log karte toh same error do baar dikhta terminal mein.
        return False  # Caller ko batao — config issue ki wajah se fail hua

    except Exception as e:
        # Unexpected error — non-fatal, project can run without tracing
        # ⚠️ YE BLOCK UNEXPECTED ERRORS KE LIYE HAI —
        # jaise network issue, ya koi aur runtime error jo anticipate nahi kiya tha.
        # Warning level isliye — critical nahi hai, app bina tracing ke bhi chalti hai
        logger.warning(f"⚠️ LangSmith initialization failed: {e}")
        return False  # Caller ko batao — unexpected error ki wajah se fail hua


def get_langsmith_status() -> dict:
    """
    Returns current LangSmith configuration status.
    Useful for health check endpoint in FastAPI.
    Returns:
        dict: LangSmith configuration details.
    """
    # 🩺 HEALTH CHECK FUNCTION —
    # FastAPI ka /health endpoint is dict ko return kar sakta hai.
    # Isse bina code mein ghuse instantly pata chalta hai ki
    # LangSmith properly configured hai ya nahi — especially useful
    # jab production mein deploy karo aur verify karna ho.

    return {
        # Tracing on hai ya off — "true" ya "false" string milegi
        "tracing_enabled": os.environ.get("LANGCHAIN_TRACING_V2", "false"),

        # Kaun sa LangSmith project use ho raha hai abhi
        "project": os.environ.get("LANGCHAIN_PROJECT", "not set"),

        # Kis server par traces ja rahe hain
        "endpoint": os.environ.get("LANGCHAIN_ENDPOINT", "not set"),

        # API key SET hai ya nahi — actual key expose nahi karte (security risk hoga),
        # sirf True/False batate hain. bool("") = False, bool("sk-...") = True
        "api_key_set": bool(os.environ.get("LANGCHAIN_API_KEY", "")),
    }