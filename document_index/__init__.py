"""Пайплайн индексации документов: чанкинг, эмбеддинги, JSON-индекс."""

from document_index.pipeline import build_index_for_file, compare_chunking_strategies
from document_index.rag import augment_user_message_with_rag, merge_question_with_chunks, search_top_chunks

__all__ = [
    "augment_user_message_with_rag",
    "build_index_for_file",
    "compare_chunking_strategies",
    "merge_question_with_chunks",
    "search_top_chunks",
]
