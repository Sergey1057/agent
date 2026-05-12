from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from document_index.chunkers import Chunk, StrategyName, chunks_for_path
from document_index.embeddings import embed_texts_auto


def _chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lens = [len(c.text) for c in chunks]
    if not lens:
        return {
            "chunk_count": 0,
            "total_chars": 0,
            "avg_chars": 0.0,
            "max_chars": 0,
            "min_chars": 0,
            "sections_nonempty": 0,
        }
    sections = sum(1 for c in chunks if (c.section or "").strip())
    return {
        "chunk_count": len(chunks),
        "total_chars": sum(lens),
        "avg_chars": round(sum(lens) / len(lens), 2),
        "max_chars": max(lens),
        "min_chars": min(lens),
        "sections_nonempty": sections,
    }


def compare_chunking_strategies(
    path: Path,
    *,
    fixed_chunk_size: int = 1500,
    fixed_overlap: int = 200,
    max_section_chars: int = 4000,
    section_sub_overlap: int = 200,
) -> dict[str, Any]:
    """Сравнение двух стратегий без эмбеддингов (быстро, для отчёта)."""
    path = path.resolve()
    fixed = chunks_for_path(
        path,
        strategy="fixed_size",
        fixed_chunk_size=fixed_chunk_size,
        fixed_overlap=fixed_overlap,
    )
    struct = chunks_for_path(
        path,
        strategy="structure",
        max_section_chars=max_section_chars,
        section_sub_overlap=section_sub_overlap,
    )
    return {
        "document": str(path),
        "parameters": {
            "fixed_size": {"chunk_size": fixed_chunk_size, "overlap": fixed_overlap},
            "structure": {
                "max_section_chars": max_section_chars,
                "section_sub_overlap": section_sub_overlap,
            },
        },
        "fixed_size": _chunk_stats(fixed),
        "structure": _chunk_stats(struct),
        "comparison_notes": (
            "fixed_size: равномерные окна по символам; границы могут резать смысловые абзацы. "
            "Поле total_chars для fixed_size суммирует длины чанков и завышено из‑за перекрытия окон. "
            "structure: чанки привязаны к заголовкам разделов; длинные разделы режутся с тем же "
            "полем section в метаданных; строки‑заголовки попадают в metadata.section, а не дублируются в text."
        ),
    }


def build_index_for_file(
    path: Path,
    *,
    strategy: StrategyName,
    out_path: Path | None = None,
    fixed_chunk_size: int = 1500,
    fixed_overlap: int = 200,
    max_section_chars: int = 4000,
    section_sub_overlap: int = 200,
    dummy_embeddings: bool = False,
) -> dict[str, Any]:
    """
    Полный пайплайн: чанки → эмбеддинги → JSON-индекс.
    """
    path = path.resolve()
    chunks = chunks_for_path(
        path,
        strategy=strategy,
        fixed_chunk_size=fixed_chunk_size,
        fixed_overlap=fixed_overlap,
        max_section_chars=max_section_chars,
        section_sub_overlap=section_sub_overlap,
    )
    texts = [c.text for c in chunks]
    t0 = time.perf_counter()
    vectors, emb_meta = embed_texts_auto(texts, dummy=dummy_embeddings)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Число эмбеддингов {len(vectors)} != числу чанков {len(chunks)}")

    records = []
    for ch, vec in zip(chunks, vectors, strict=True):
        records.append(
            {
                "text": ch.text,
                "embedding": vec,
                "metadata": ch.metadata(),
            }
        )

    index: dict[str, Any] = {
        "version": 1,
        "document": str(path),
        "strategy": strategy,
        "embedding": emb_meta,
        "indexing_time_ms": elapsed_ms,
        "chunk_stats": _chunk_stats(chunks),
        "chunks": records,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    return index


def save_comparison_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def chunk_dicts_only(path: Path, strategy: StrategyName) -> list[dict[str, Any]]:
    """Для тестов: только чанки как словари."""
    chunks = chunks_for_path(path, strategy=strategy)
    return [{"text": c.text, "metadata": c.metadata()} for c in chunks]
