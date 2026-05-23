# =============================================================================
# SkyLex — Centralized Logging Configuration
# Sets up logging for the entire project with console and file handlers.
# Import and call setup_logging() once at application startup.
# Usage: from monitoring.logger import get_logger
#        logger = get_logger(__name__)
# =============================================================================
# 📌 YE FILE KYA KARTI HAI?
# Poore SkyLex project ka logging system yahan se control hota hai.
# Ek baar startup pe setup_logging() call karo — uske baad project ka
# har module get_logger(__name__) se apna logger le sakta hai.
#
# Logs DO jagah jaate hain simultaneously:
#   1. Terminal (console) — development mein real-time dekhne ke liye
#   2. logs/skylex.log file — production mein audit trail ke liye
# =============================================================================

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler  # File ko infinitely bada hone se rokta hai
from config.settings import settings  # log_level yahan se aata hai (e.g., "INFO" ya "DEBUG")

# 📁 Log file path — project root ke andar logs/ folder mein
# __file__ = ye current file (logger.py), .parent.parent = project root
LOG_FILE = Path(__file__).parent.parent / "logs" / "skylex.log"


def setup_logging() -> None:
    """
    Configure logging for the entire SkyLex project.

    Sets up two handlers:
    - Console handler: colored output to terminal
    - File handler: rotating log file (max 10MB, 5 backups)

    Call this function ONCE at application startup before anything else.
    """
    # ⚙️ settings.log_level ek string hoga jaise "INFO" ya "DEBUG"
    # getattr(logging, "INFO") = logging.INFO (numeric constant 20)
    # Agar invalid level diya toh fallback logging.INFO pe ho jaata hai
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # 📋 Har log line ka format — example output:
    # 2025-01-15 14:32:01 | INFO     | services.ingestion | Ingestion started
    # %-8s matlab "levelname" ko 8 characters mein pad karo — alignment ke liye
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Formatter object — yahi format string ko actual log lines mein convert karta hai
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # -------------------------------------------------------------------------
    # Console Handler — prints to terminal
    # -------------------------------------------------------------------------
    # sys.stdout = normal terminal output (stderr nahi, taki errors alag rakhe)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)       # Sirf is level aur upar ke logs print hone
    console_handler.setFormatter(formatter)   # Upar wala format apply karo

    # -------------------------------------------------------------------------
    # File Handler — rotating log file (max 10MB per file, keep 5 backups)
    # -------------------------------------------------------------------------
    # 🔄 ROTATING FILE HANDLER KYA HOTA HAI?
    # Jab skylex.log 10MB ho jaata hai, automatically rename hota hai skylex.log.1
    # Naya skylex.log banta hai. Aise max 5 backup files rakhta hai:
    # skylex.log, skylex.log.1, skylex.log.2 ... skylex.log.5
    # 6th rotate pe sabse purani file delete ho jaati hai — disk full hone se bacha

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)  # logs/ folder nahi hai toh banao, exist_ok=True matlab already hai toh error mat do

    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB = 10 × 1024 × 1024 bytes (explicitly calculate kiya clarity ke liye)
        backupCount=5,               # Purani 5 files rakhni hain, 6th pe oldest delete
        encoding="utf-8",            # Unicode support — special characters aur emojis log mein safe rahein
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # -------------------------------------------------------------------------
    # Root Logger — all loggers in project inherit from this
    # -------------------------------------------------------------------------
    # 🌳 ROOT LOGGER KYA HOTA HAI?
    # Python ka logging ek tree structure follow karta hai.
    # logging.getLogger("services.ingestion") ka parent hai logging.getLogger("services"),
    # jiska parent hai root logger (logging.getLogger() — no name).
    # Root logger pe level aur handlers set karo — sab inherit kar lete hain automatically.
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 🛡️ DUPLICATE HANDLER GUARD
    # Agar koi galti se setup_logging() do baar call kare (jaise testing mein),
    # toh bina is check ke har log line DOUBLE print hogi.
    # handlers list empty hai = pehli baar call ho raha hai — tab hi add karo
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    # 🔕 THIRD-PARTY LIBRARIES KO CHUP KARAO
    # httpx, openai, langchain ye sab internally bahut verbose logging karte hain —
    # har HTTP request, retry, header sab print karte hain.
    # Hum unka level WARNING pe set kar dete hain taki sirf serious issues dikhein,
    # warna hamara actual SkyLex logs in library logs mein dub jaayenge
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)

    # ✅ Sab kuch setup ho gaya — ek confirmation log print karo
    logging.info(
        f"✅ Logging initialized | Level: {settings.log_level} | "
        f"File: {LOG_FILE}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for any module in the project.

    Args:
        name: Module name — always pass __name__ here.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        from monitoring.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Ingestion started")
    """
    # 🎯 YE FUNCTION SIRF EK WRAPPER HAI — par bahut zaroori hai
    #
    # __name__ Python ka built-in variable hai jo current module ka full naam deta hai.
    # Jaise agar services/ingestion.py mein call karo, toh __name__ = "services.ingestion"
    #
    # Iska fayda: log line mein clearly dikhta hai KIS FILE se message aaya —
    # "2025-01-15 14:32:01 | INFO | services.ingestion | Ingestion started"
    #
    # Seedha logging.getLogger() call karne ke bajaye ye function isliye banaya
    # taaki agar future mein logging setup change ho, sirf yahan badalna pade
    return logging.getLogger(name)