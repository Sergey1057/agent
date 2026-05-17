"""
Сбор документации проекта (README, docs/, схемы) и построение единого RAG-индекса.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from document_index.chunkers import Chunk, StrategyName, chunks_for_path
from document_index.embeddings import embed_texts_auto

README_CANDIDATES: tuple[str, ...] = (
    "README.md",
    "README.MD",
    "readme.md",
    "README.rst",
    "README",
    "REDME.md",
)

DOC_DIR_NAMES: tuple[str, ...] = ("docs", "project/docs", "doc", "documentation")

SCHEMA_GLOBS: tuple[str, ...] = (
    "**/*.openapi.yaml",
    "**/*.openapi.yml",
    "**/*.openapi.json",
    "**/openapi.yaml",
    "**/openapi.yml",
    "**/openapi.json",
    "**/*schema*.json",
    "**/api/**/*.md",
    "**/api/**/*.yaml",
    "**/api/**/*.yml",
)


def collect_corpus_paths(
    project_root: Path,
    *,
    extra_paths: Iterable[Path | str] | None = None,
) -> list[Path]:
    """
    Собирает пути к текстовой документации в корне проекта:
    README*, каталоги docs/, OpenAPI/схемы.
    """
    root = project_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Корень проекта не найден: {root}")

    found: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen or not p.is_file():
            return
        seen.add(key)
        found.append(p.resolve())

    for name in README_CANDIDATES:
        add(root / name)

    for rel_dir in DOC_DIR_NAMES:
        d = root / rel_dir
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in (
                ".md",
                ".txt",
                ".rst",
                ".adoc",
                ".json",
                ".yaml",
                ".yml",
            ):
                add(p)

    for pattern in SCHEMA_GLOBS:
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                add(p)

    if extra_paths:
        for raw in extra_paths:
            p = Path(raw).expanduser()
            if p.is_file():
                add(p)
            elif p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and f.suffix.lower() in (
                        ".md",
                        ".txt",
                        ".json",
                        ".yaml",
                        ".yml",
                    ):
                        add(f)

    return sorted(found, key=lambda x: str(x).lower())


def build_index_for_corpus(
    paths: list[Path],
    *,
    strategy: StrategyName,
    out_path: Path | None = None,
    project_root: Path | None = None,
    fixed_chunk_size: int = 1500,
    fixed_overlap: int = 200,
    max_section_chars: int = 4000,
    section_sub_overlap: int = 200,
    dummy_embeddings: bool = False,
) -> dict[str, Any]:
    """Чанки из нескольких файлов → один JSON-индекс."""
    if not paths:
        raise ValueError("Список документов для индексации пуст.")

    all_chunks: list[Chunk] = []
    sources: list[str] = []
    for path in paths:
        path = path.resolve()
        chunks = chunks_for_path(
            path,
            strategy=strategy,
            fixed_chunk_size=fixed_chunk_size,
            fixed_overlap=fixed_overlap,
            max_section_chars=max_section_chars,
            section_sub_overlap=section_sub_overlap,
        )
        all_chunks.extend(chunks)
        sources.append(str(path))

    texts = [c.text for c in all_chunks]
    t0 = time.perf_counter()
    vectors, emb_meta = embed_texts_auto(texts, dummy=dummy_embeddings)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if len(vectors) != len(all_chunks):
        raise RuntimeError(
            f"Число эмбеддингов {len(vectors)} != числу чанков {len(all_chunks)}"
        )

    records = []
    for ch, vec in zip(all_chunks, vectors, strict=True):
        records.append(
            {
                "text": ch.text,
                "embedding": vec,
                "metadata": ch.metadata(),
            }
        )

    from document_index.pipeline import _chunk_stats

    index: dict[str, Any] = {
        "version": 1,
        "document": str(project_root.resolve()) if project_root else sources[0],
        "corpus_sources": sources,
        "strategy": strategy,
        "embedding": emb_meta,
        "indexing_time_ms": elapsed_ms,
        "chunk_stats": _chunk_stats(all_chunks),
        "chunks": records,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    return index
