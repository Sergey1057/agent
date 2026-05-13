"""Пайплайн индексации документов: чанкинг, эмбеддинги, JSON-индекс."""

from document_index.pipeline import build_index_for_file, compare_chunking_strategies
from document_index.rag import (
    RagRetrievalConfig,
    augment_user_message_with_rag,
    compare_rag_modes,
    merge_question_with_chunks,
    rewrite_query_for_rag,
    search_top_chunks,
)

__all__ = [
    "RagRetrievalConfig",
    "augment_user_message_with_rag",
    "build_index_for_file",
    "compare_chunking_strategies",
    "compare_rag_modes",
    "merge_question_with_chunks",
    "rewrite_query_for_rag",
    "search_top_chunks",
]
