# =============================================================================
# SkyLex — MLflow Experiment Tracking
# MLflow tracks all experiments — chunking, embedding, RAG pipeline runs.
# Records parameters, metrics, and artifacts for every experiment.
# Usage: from monitoring.mlflow_tracker import SkyLexMLflowTracker
# =============================================================================
# 📌 YE FILE KYA KARTI HAI?
# LangSmith real-time tracing karta hai (har query ka live trace).
# MLflow experiment tracking karta hai (offline experiments compare karne ke liye).
# Dono alag cheezein hain — dono saath chalte hain SkyLex mein.
#
# MLflow use karte ho jab:
#   - Alag alag chunk sizes try karo aur results compare karo
#   - Embedding models compare karo (text-embedding-3-small vs large)
#   - RAG pipeline ka retrieval score track karo across runs
#
# SkyLexMLflowTracker ek wrapper class hai jo:
#   - Context manager hai (with block mein use karo — run auto start/end hoti hai)
#   - Har log operation mein try-except hai — MLflow fail ho toh app nahi rukti
# =============================================================================

import mlflow
from typing import Any  # Function signatures mein flexible types ke liye
from monitoring.logger import get_logger
from exceptions.exceptions import SkyLexConfigError  # Config issues ke liye (future use)
from config.settings import settings  # MLflow URI aur experiment name yahan se aate hain

# Is module ka named logger — logs mein "monitoring.mlflow_tracker" dikhega
logger = get_logger(__name__)


def initialize_mlflow() -> bool:
    """
    Initialize MLflow tracking for the SkyLex project.
    Sets tracking URI and experiment name from settings.
    Call this function ONCE at application startup.

    Returns:
        bool: True if initialized successfully, False otherwise.
    """
    try:
        # 📍 TRACKING URI SET KARO —
        # Yahan decide hota hai ki MLflow data KAHAAN store karega.
        # Local: "http://localhost:5000" (mlflow ui command se server chalao)
        # Remote: koi hosted MLflow server ka URL
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        # 🧪 EXPERIMENT SET YA CREATE KARO —
        # Experiment ek folder jaisa hota hai MLflow dashboard mein.
        # Agar is naam ka experiment already exist karta hai toh wahi use hoga,
        # nahi hai toh MLflow automatically naya bana dega — koi error nahi aayega
        mlflow.set_experiment(settings.mlflow_experiment_name)

        # Sab setup ho gaya — confirmation log karo
        logger.info(
            f"✅ MLflow initialized | "
            f"Tracking URI: {settings.mlflow_tracking_uri} | "
            f"Experiment: {settings.mlflow_experiment_name}"
        )
        return True  # Caller ko batao — initialization successful rahi

    except Exception as e:
        # ⚠️ NON-FATAL ERROR —
        # MLflow fail ho (jaise local server nahi chal raha) toh bhi
        # SkyLex normally kaam karta rahega — sirf tracking band rahegi
        logger.warning(f"⚠️ MLflow initialization failed: {e}")
        return False


class SkyLexMLflowTracker:
    """
    Wrapper around MLflow for tracking SkyLex experiments.
    Use as a context manager for clean run management.

    Example:
        with SkyLexMLflowTracker(run_name="chunking_experiment") as tracker:
            tracker.log_param("chunk_size", 500)
            tracker.log_metric("retrieval_score", 0.82)
    """
    # 💡 CONTEXT MANAGER KYA HOTA HAI?
    # `with SkyLexMLflowTracker(...) as tracker:` likhne par:
    #   - __enter__ call hota hai (run start hoti hai)
    #   - with block ka code chalta hai
    #   - __exit__ call hota hai (run end hoti hai) — chahe error aaye ya naa aaye
    # Isse manually mlflow.start_run() / mlflow.end_run() likhne ki zaroorat nahi

    def __init__(self, run_name: str) -> None:
        """
        Args:
            run_name: Descriptive name for this experiment run.
        """
        # Run ka naam store karo — MLflow dashboard mein yahi naam dikhega
        self.run_name = run_name
        # run object baad mein __enter__ mein set hoga — abhi None hai
        self.run = None

    def __enter__(self) -> "SkyLexMLflowTracker":
        """Start MLflow run when entering context."""
        try:
            # 🏁 RUN START KARO —
            # MLflow ek unique run_id generate karta hai aur tracking shuru karta hai.
            # self.run mein store karo taaki __exit__ mein check kar sakein ki run exist karta hai
            self.run = mlflow.start_run(run_name=self.run_name)
            logger.info(f"📊 MLflow run started | Run: {self.run_name}")
        except Exception as e:
            # Run start fail hua — warning log karo, self.run None rahega
            # __exit__ mein `if self.run` check hai isliye None pe crash nahi hoga
            logger.warning(f"⚠️ MLflow run start failed: {e}")
        return self  # `as tracker` ke liye self return karna zaroori hai

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End MLflow run when exiting context — even if error occurred."""
        # exc_type, exc_val, exc_tb — agar with block mein exception aayi toh
        # ye populated honge, warna teeno None honge. Hum inhe use nahi karte
        # kyunki chahe error aaye ya na aaye, run end karni hai.
        try:
            # Sirf tab end karo jab run successfully start hua tha
            if self.run:
                mlflow.end_run()
                logger.info(f"📊 MLflow run ended | Run: {self.run_name}")
        except Exception as e:
            logger.warning(f"⚠️ MLflow run end failed: {e}")

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a parameter for this run.
        Parameters are config values — chunk_size, model_name, etc.

        Args:
            key: Parameter name.
            value: Parameter value.
        """
        # 📝 PARAM VS METRIC FARK:
        # Param = configuration value jo experiment setup karta hai (chunk_size=500)
        # Metric = numeric result jo experiment produce karta hai (retrieval_score=0.82)
        # Params ek baar log hote hain, metrics multiple times (steps ke saath) log ho sakte hain
        try:
            mlflow.log_param(key, value)
            logger.debug(f"📊 MLflow param logged | {key}={value}")  # debug level — verbose info
        except Exception as e:
            # Non-fatal — ek param log fail hua toh baki experiment continue karega
            logger.warning(f"⚠️ MLflow log_param failed | {key}: {e}")

    def log_metric(self, key: str, value: float) -> None:
        """
        Log a metric for this run.
        Metrics are numeric results — retrieval_score, latency, cost, etc.

        Args:
            key: Metric name.
            value: Numeric metric value.
        """
        # 📊 METRIC FLOAT HONA CHAHIYE —
        # MLflow metrics sirf numeric values accept karta hai (int ya float).
        # String metrics ke liye log_param use karo.
        try:
            mlflow.log_metric(key, value)
            logger.debug(f"📊 MLflow metric logged | {key}={value}")
        except Exception as e:
            logger.warning(f"⚠️ MLflow log_metric failed | {key}: {e}")

    def log_params(self, params: dict) -> None:
        """
        Log multiple parameters at once.

        Args:
            params: Dictionary of parameter key-value pairs.
        """
        # 🚀 BULK VERSION OF log_param —
        # Ek saath poora config dict pass kar do instead of ek ek key likhne ke.
        # Example: tracker.log_params({"chunk_size": 500, "model": "gpt-4o-mini"})
        try:
            mlflow.log_params(params)
            logger.debug(f"📊 MLflow params logged | {params}")
        except Exception as e:
            logger.warning(f"⚠️ MLflow log_params failed: {e}")

    def log_metrics(self, metrics: dict) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric key-value pairs.
        """
        # 🚀 BULK VERSION OF log_metric —
        # Ek saath poore results dict pass kar do.
        # Example: tracker.log_metrics({"retrieval_score": 0.82, "latency_ms": 340.5})
        try:
            mlflow.log_metrics(metrics)
            logger.debug(f"📊 MLflow metrics logged | {metrics}")
        except Exception as e:
            logger.warning(f"⚠️ MLflow log_metrics failed: {e}")