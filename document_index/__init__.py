"""Пайплайн индексации документов: чанкинг, эмбеддинги, JSON-индекс."""

from document_index.pipeline import build_index_for_file, compare_chunking_strategies

__all__ = ["build_index_for_file", "compare_chunking_strategies"]
