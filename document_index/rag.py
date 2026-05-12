from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from document_index.embeddings import embed_texts_auto

# Для «фиктивных» эмбеддингов (deterministic_hash) косинус не отражает смысл — только случайное
# сходство векторов. Тогда ранжируем чанки по пересечению слов/префиксов с запросом.
_LEX_STOPWORDS: frozenset[str] = frozenset(
    """
    что как для при это или без над под от из по со же ли бы том тех пор раз два
    ваш вас наш нас вам им его ему них ней ним кто чем ком кого кому чей чья
    где когда куда откуда почему лищь ведь уже ещё еще все всё так вот там тут
    нет да не ни даже оно она они они мне меня мной тебе тебя вам вас нас вас
    есть был была были быть буду будет нужно можно надо либо иначе разве
    сколько какой какая какие каким каком какому зачем почему чей чьё чьем
    """.split()
)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _dummy_from_index_meta(emb_meta: dict[str, Any]) -> bool:
    prov = str(emb_meta.get("provider") or "").lower()
    return prov == "deterministic_hash"


def _lexical_match_terms(question: str) -> list[str]:
    """Нормализованные фрагменты запроса (длина ≥4), без слишком общих слов."""
    qlow = (question or "").lower()
    raw = re.findall(r"[\w]{4,}", qlow, flags=re.UNICODE)
    # «тура/туру» в длинных чанках почти всегда входит в «туристский» и ломает ранжирование;
    # если в вопросе есть «установить», опираемся на другие термины.
    if "установ" in qlow:
        raw = [t for t in raw if t not in ("тура", "туру", "туре", "туры")]
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str, *, min_len: int = 4) -> None:
        if len(s) < min_len or s in seen or s in _LEX_STOPWORDS:
            return
        seen.add(s)
        out.append(s)

    for t in raw:
        if t in _LEX_STOPWORDS:
            continue
        if len(t) <= 10:
            add(t)
        else:
            add(t[:9])
        if len(t) >= 7 and t.endswith(
            ("ту", "ты", "те", "ти", "та", "то", "ю", "ем", "ом", "ам", "ах", "ях", "ую")
        ):
            add(t[:-1], min_len=5)
        if len(t) >= 5 and t.endswith(("ов", "ом", "ам", "ах", "ям", "ой", "ый", "ая", "ое", "ие", "ых")):
            stem = t[:-2]
            if len(stem) >= 3:
                add(stem, min_len=3)
    return out


def _lexical_chunk_score_final(
    terms: list[str], chunk_lower: str, idf: dict[str, float], chunk_len: int
) -> float:
    """Сумма взвешенных совпадений, штраф за очень длинный чанк."""
    if not terms:
        return 0.0
    s = 0.0
    for t in terms:
        c = chunk_lower.count(t)
        if c:
            s += min(4, c) * idf.get(t, 1.0)
    return float(s / math.log(max(chunk_len, 80)))


def _search_top_chunks_lexical(
    question: str,
    chunks_raw: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    terms = _lexical_match_terms(question)
    texts_lower = [str(ch.get("text") or "").lower() for ch in chunks_raw]
    n_docs = len(texts_lower) or 1
    idf: dict[str, float] = {}
    for t in terms:
        df = sum(1 for txt in texts_lower if t in txt)
        idf[t] = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, (ch, low) in enumerate(zip(chunks_raw, texts_lower, strict=True)):
        txt = str(ch.get("text") or "")
        s = _lexical_chunk_score_final(terms, low, idf, len(txt))
        meta = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
        scored.append(
            (
                s,
                idx,
                {
                    "text": txt,
                    "metadata": dict(meta),
                    "score": s,
                    "score_kind": "lexical",
                },
            )
        )
    scored.sort(key=lambda x: (-x[0], x[1]))
    if not scored or scored[0][0] <= 0.0:
        # Нет пересечения с запросом — отдаём начало документа (лучше, чем случайный косинус)
        k = max(1, min(int(top_k), len(chunks_raw)))
        out: list[dict[str, Any]] = []
        for i in range(k):
            ch = chunks_raw[i]
            meta = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
            out.append(
                {
                    "text": str(ch.get("text") or ""),
                    "metadata": dict(meta),
                    "score": 0.0,
                    "score_kind": "lexical_fallback",
                }
            )
        return out
    k = max(1, min(int(top_k), len(scored)))
    return [t[2] for t in scored[:k]]


def load_index_chunks(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    chunks = raw.get("chunks") or []
    emb_meta = raw.get("embedding") or {}
    if not isinstance(chunks, list):
        raise ValueError("Поле chunks в индексе должно быть списком.")
    return chunks, emb_meta


def search_top_chunks(
    question: str,
    index_path: Path,
    *,
    top_k: int = 5,
    dummy_embeddings: bool | None = None,
) -> list[dict[str, Any]]:
    """
    Поиск top-k чанков: при семантических эмбеддингах — косинус к вектору вопроса;
    при индексе с deterministic_hash — лексический ранг (подстроки запроса + IDF по чанкам),
    т.к. такие векторы не пригодны для смыслового поиска.
    Каждый элемент: text, metadata, score, опционально score_kind.
    """
    qtext = (question or "").strip()
    chunks_raw, emb_meta = load_index_chunks(index_path)
    if not chunks_raw:
        return []
    dummy = (
        dummy_embeddings if dummy_embeddings is not None else _dummy_from_index_meta(emb_meta)
    )
    if dummy:
        return _search_top_chunks_lexical(qtext, chunks_raw, top_k=top_k)

    vecs, _ = embed_texts_auto([qtext], dummy=False)
    q = [float(x) for x in vecs[0]]
    dim_q = len(q)

    first_emb: list[float] | None = None
    for ch in chunks_raw:
        emb = ch.get("embedding")
        if isinstance(emb, list) and emb:
            first_emb = [float(x) for x in emb]
            break
    if first_emb is None:
        return []
    if len(first_emb) != dim_q:
        raise ValueError(
            f"Размерность эмбеддинга запроса ({dim_q}) не совпадает с индексом ({len(first_emb)}). "
            "Соберите индекс тем же провайдером эмбеддингов (см. document_index.embeddings)."
        )

    scored: list[tuple[float, dict[str, Any]]] = []
    for ch in chunks_raw:
        emb = ch.get("embedding")
        if not isinstance(emb, list) or len(emb) != dim_q:
            continue
        vec = [float(x) for x in emb]
        s = _cosine_sim(q, vec)
        meta = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
        text = str(ch.get("text") or "")
        scored.append((s, {"text": text, "metadata": dict(meta), "score": s, "score_kind": "cosine"}))
    scored.sort(key=lambda x: x[0], reverse=True)
    k = max(1, min(int(top_k), len(scored)))
    return [t[1] for t in scored[:k]]


def merge_question_with_chunks(question: str, hits: list[dict[str, Any]]) -> str:
    """Текст пользовательского сообщения для API: выдержки + вопрос."""
    q = (question or "").strip()
    lines: list[str] = [
        "Ответь строго по выдержкам ниже. Перескажи формулировки из выдержек; "
        "не добавляй фактов из общих знаний, если их нет в тексте выдержек. "
        "Если ответа в выдержках нет — напиши, что в документе это не указано.",
        "",
    ]
    if not hits:
        lines.append("(Релевантных фрагментов не найдено — ответь по общим знаниям, если уместно.)")
        lines.append("")
    else:
        for i, h in enumerate(hits, start=1):
            meta = h.get("metadata") or {}
            section = str(meta.get("section") or "").strip()
            score = float(h.get("score") or 0.0)
            kind = str(h.get("score_kind") or "cosine")
            if kind == "lexical":
                head = f"--- Выдержка {i} (совпадения с запросом: {score:.0f})"
            elif kind == "lexical_fallback":
                head = f"--- Выдержка {i} (начало документа; по запросу совпадений не найдено)"
            else:
                head = f"--- Выдержка {i} (сходство {score:.4f})"
            if section:
                head += f" | раздел: {section}"
            lines.append(head)
            lines.append(str(h.get("text") or "").strip())
            lines.append("")
    lines.append("--- Вопрос пользователя")
    lines.append(q)
    return "\n".join(lines).strip()


def augment_user_message_with_rag(
    question: str,
    index_path: Path | str,
    *,
    top_k: int = 5,
    dummy_embeddings: bool | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Вопрос → поиск чанков → объединение с вопросом.
    Возвращает (текст для поля user в chat completion, список попаданий).
    """
    path = Path(index_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Файл индекса не найден: {path}")
    hits = search_top_chunks(
        question,
        path,
        top_k=top_k,
        dummy_embeddings=dummy_embeddings,
    )
    return merge_question_with_chunks(question, hits), hits
