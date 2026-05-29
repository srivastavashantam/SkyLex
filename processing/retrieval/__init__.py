"""
processing/retrieval/__init__.py

SkyLex Retrieval System — Public API.

Saare 7 RAG variants yahan se import karo:
  from processing.retrieval import RetrievalFactory, ALL_VARIANTS
"""

from processing.retrieval.retrieval_factory import ALL_VARIANTS, RetrievalFactory
from processing.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from processing.retrieval.sparse_retriever import SparseRetriever
from processing.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from processing.retrieval.reranker import CrossEncoderReranker
from processing.retrieval.contextual_retriever import ContextualRetriever
from processing.retrieval.parent_doc_retriever import ParentDocumentRetriever, PARENT_CHILD_MAPPING
from processing.retrieval.metadata_retriever import MetadataFilteredRetriever, FILTER_PRESETS
from processing.retrieval.bm25_indexer import build_all_indexes, load_bm25_index, print_index_stats

__all__ = [
    "RetrievalFactory",
    "ALL_VARIANTS",
    "RetrievalResult",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "ContextualRetriever",
    "ParentDocumentRetriever",
    "PARENT_CHILD_MAPPING",
    "MetadataFilteredRetriever",
    "FILTER_PRESETS",
    "build_all_indexes",
    "load_bm25_index",
    "print_index_stats",
]