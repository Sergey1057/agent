"""Пайплайн индексации документов: чанкинг, эмбеддинги, JSON-индекс."""

from document_index.pipeline import build_index_for_file, compare_chunking_strategies
from document_index.rag import (
    RagAugmentResult,
    RagGroundingCheck,
    RagRetrievalConfig,
    RagRetrievalOutcome,
    augment_user_message_with_rag,
    build_rag_grounded_user_message,
    compare_rag_modes,
    merge_question_with_chunks,
    parse_rag_grounding_sections,
    retrieve_for_rag,
    default_rag_index_path,
    resolve_rag_index_path,
    rewrite_query_for_rag,
    search_top_chunks,
    validate_rag_grounding_reply,
)

__all__ = [
    "RagAugmentResult",
    "RagGroundingCheck",
    "RagRetrievalConfig",
    "RagRetrievalOutcome",
    "augment_user_message_with_rag",
    "build_index_for_file",
    "build_rag_grounded_user_message",
    "compare_chunking_strategies",
    "compare_rag_modes",
    "default_rag_index_path",
    "merge_question_with_chunks",
    "parse_rag_grounding_sections",
    "retrieve_for_rag",
    "resolve_rag_index_path",
    "rewrite_query_for_rag",
    "search_top_chunks",
    "validate_rag_grounding_reply",
]
