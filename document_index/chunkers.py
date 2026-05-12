from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

StrategyName = Literal["fixed_size", "structure"]


@dataclass
class Chunk:
    """Фрагмент текста с метаданными для индекса."""

    text: str
    chunk_id: str
    source: str
    title: str
    section: str
    strategy: StrategyName
    char_start: int = 0
    char_end: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def metadata(self) -> dict[str, str]:
        m: dict[str, str] = {
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "strategy": self.strategy,
        }
        m.update(self.extra)
        return m


def _document_title(first_lines: list[str], path: Path) -> str:
    for ln in first_lines[:5]:
        t = ln.strip()
        if t:
            if len(t) <= 200:
                return t
            return t[:197] + "..."
    return path.name


def _stable_source_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def chunk_fixed_size(
    text: str,
    *,
    source: str,
    title: str,
    chunk_size: int = 1500,
    overlap: int = 200,
    strategy: StrategyName = "fixed_size",
) -> list[Chunk]:
    """Разбиение по фиксированному числу символов с перекрытием."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    sid = _stable_source_id(source)
    chunks: list[Chunk] = []
    n = len(text)
    start = 0
    idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            cid = f"{strategy}:{sid}:{idx:05d}"
            chunks.append(
                Chunk(
                    text=piece,
                    chunk_id=cid,
                    source=source,
                    title=title,
                    section="",
                    strategy=strategy,
                    char_start=start,
                    char_end=end,
                    extra={"subsection": ""},
                )
            )
            idx += 1
        if end >= n:
            break
        start = end - overlap
    return chunks


_HEADING_ATTENTION = re.compile(r"^(ВНИМАНИЕ!|Внимание!)", re.IGNORECASE)
_ALL_CAPS_CYR = re.compile(r"^[А-ЯЁA-Z0-9\s,\.\-–—«»]+[А-ЯЁA-Z0-9]\s*$")
_NUMBERED = re.compile(r"^\d+[\).]\s")


def _is_likely_heading(line: str, prev_blank: bool, next_line: str | None) -> bool:
    s = line.strip()
    if not s or len(s) > 220:
        return False
    if _NUMBERED.match(s):
        return False
    if _HEADING_ATTENTION.match(s):
        return True
    if len(s) >= 8 and len(s) <= 200 and _ALL_CAPS_CYR.match(s) and s.count(".") <= 1:
        return True
    # Короткая «тема» в начале абзаца (после пустой строки), 1–3 коротких предложения
    if s.endswith(".") and 10 <= len(s) <= 200 and prev_blank:
        parts = re.split(r"(?<=[.!?])\s+", s)
        if len(parts) <= 3 and s.count(",") < 6:
            if next_line:
                nxt = next_line.strip()
                if nxt and nxt[0].islower():
                    return False
            return True
    return False


def _split_oversized_section(
    body: str,
    *,
    max_chars: int,
    overlap: int,
    section: str,
    source: str,
    title: str,
    sid: str,
    base_idx: int,
) -> tuple[list[Chunk], int]:
    """Внутри одного раздела — при необходимости фиксированные под-чанки."""
    body = body.strip()
    if not body:
        return [], base_idx
    if len(body) <= max_chars:
        cid = f"structure:{sid}:{base_idx:05d}"
        return (
            [
                Chunk(
                    text=body,
                    chunk_id=cid,
                    source=source,
                    title=title,
                    section=section,
                    strategy="structure",
                    extra={"subsection": ""},
                )
            ],
            base_idx + 1,
        )
    out: list[Chunk] = []
    idx = base_idx
    start = 0
    n = len(body)
    while start < n:
        end = min(start + max_chars, n)
        piece = body[start:end].strip()
        if piece:
            cid = f"structure:{sid}:{idx:05d}"
            out.append(
                Chunk(
                    text=piece,
                    chunk_id=cid,
                    source=source,
                    title=title,
                    section=section,
                    strategy="structure",
                    char_start=start,
                    char_end=end,
                    extra={"subsection": f"part_{idx - base_idx + 1}"},
                )
            )
            idx += 1
        if end >= n:
            break
        start = end - overlap
    return out, idx


def chunk_by_structure(
    text: str,
    *,
    source: str,
    title: str,
    max_section_chars: int = 4000,
    section_sub_overlap: int = 200,
) -> list[Chunk]:
    """
    Разбиение по структуре: эвристика заголовков (отдельные строки перед абзацами),
    длинные разделы дополнительно режутся по размеру с сохранением section в метаданных.
    """
    lines = text.splitlines()
    doc_title = title or _document_title(lines, Path(source))
    sid = _stable_source_id(source)
    chunks: list[Chunk] = []
    if not lines:
        return chunks

    current_heading = doc_title
    buf: list[str] = []
    idx = 0

    def flush_buffer() -> None:
        nonlocal idx, chunks, buf, current_heading
        body = "\n".join(buf).strip()
        buf = []
        if not body:
            return
        new_chunks, idx = _split_oversized_section(
            body,
            max_chars=max_section_chars,
            overlap=section_sub_overlap,
            section=current_heading,
            source=source,
            title=doc_title,
            sid=sid,
            base_idx=idx,
        )
        chunks.extend(new_chunks)

    prev_blank = True
    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        blank = not line.strip()
        if not blank and _is_likely_heading(line, prev_blank, next_line):
            flush_buffer()
            current_heading = line.strip()
            i += 1
            prev_blank = False
            continue
        buf.append(line)
        prev_blank = blank
        i += 1
    flush_buffer()
    return chunks


def chunks_for_path(
    path: Path,
    *,
    strategy: StrategyName,
    fixed_chunk_size: int = 1500,
    fixed_overlap: int = 200,
    max_section_chars: int = 4000,
    section_sub_overlap: int = 200,
) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    source = str(path.resolve())
    title = _document_title(text.splitlines(), path)
    if strategy == "fixed_size":
        return chunk_fixed_size(
            text,
            source=source,
            title=title,
            chunk_size=fixed_chunk_size,
            overlap=fixed_overlap,
        )
    return chunk_by_structure(
        text,
        source=source,
        title=title,
        max_section_chars=max_section_chars,
        section_sub_overlap=section_sub_overlap,
    )
