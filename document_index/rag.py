from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_index.embeddings import embed_texts_auto

_REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_rag_index_path(raw: Path | str) -> Path:
    """
    Находит существующий JSON-индекс по строке пути.

    Частые ошибки:
    - Ведущий `/` (`/memory/...`) — в Unix это путь от **корня диска**, а не от проекта.
    - Файл лежит в `memory/index_out/`, а указывают только `memory/index_structure.json`.
    """
    s = str(raw).strip()
    if not s:
        raise FileNotFoundError("Путь к индексу пустой.")
    p = Path(s).expanduser()
    try:
        direct = p.resolve()
    except OSError:
        direct = p
    if direct.is_file():
        return direct

    candidates: list[Path] = []
    if p.is_absolute() and len(direct.parts) > 1:
        rel = Path(*direct.parts[1:])
        for base in (Path.cwd(), _REPO_ROOT):
            candidates.append((base / rel).resolve())
            if rel.parts and rel.parts[0] == "memory" and "index_out" not in rel.parts:
                candidates.append((base / "memory" / "index_out" / rel.name).resolve())
    for cand in candidates:
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue

    raise FileNotFoundError(
        f"Файл индекса не найден: {raw}\n"
        f"  Указанный путь ведёт к `{direct}` (не существует).\n"
        f"  Если имели в виду каталог проекта, **не ставьте** `/` в начале: например, "
        f"`memory/index_out/index_structure.json`.\n"
        f"  Индекс по умолчанию в этом репозитории: `memory/index_out/index_structure.json`."
    )

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


def _lexical_idf_for_corpus(terms: list[str], texts_lower: list[str]) -> dict[str, float]:
    n_docs = len(texts_lower) or 1
    idf: dict[str, float] = {}
    for t in terms:
        df = sum(1 for txt in texts_lower if t in txt)
        idf[t] = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    return idf


def _normalize_lex_scores(raw_scores: list[float]) -> list[float]:
    """Сжатие в [0, 1] для смешивания с косинусом (Монотонное преобразование)."""
    if not raw_scores:
        return []
    mx = max(raw_scores) or 1.0
    return [min(1.0, s / mx) for s in raw_scores]


def rewrite_query_for_rag(question: str) -> str:
    """
    Лёгкое переписывание запроса для поиска: пробелы, хвостовая вежливость.
    Не меняет смысл агрессивно — только убирает шум для эмбеддинга/лексики.
    """
    q = (question or "").strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(
        r"(?i)\s*,?\s*(пожалуйста|спасибо|заранее благодарен|заранее спасибо)\s*[!.]?\s*$",
        "",
        q,
    ).strip()
    return q


@dataclass(frozen=True)
class RagRetrievalConfig:
    """
    Двухэтапный RAG: широкий пул кандидатов → (опц.) гибридный реранк → порог → top-K.

    - retrieve_k: сколько кандидатов после первого ранжирования (до отсечения).
    - final_top_k: сколько чанков отдаётся в промпт после фильтра.
    - min_similarity: нижний порог для семантического режима (косинус или гибридный score).
    - min_lexical_ratio: для lexical/dummy — отсечь ниже max_score * ratio (если min_lexical_absolute None).
    - min_lexical_absolute: абсолютный порог лексического score (перекрывает ratio, если задан).
    - hybrid_rerank: для настоящих эмбеддингов — пересортировать top retrieve_k смесью косинус + лексика.
    - query_rewrite: применить rewrite_query_for_rag к строке поиска.
    """

    retrieve_k: int
    final_top_k: int
    min_similarity: float | None
    min_lexical_ratio: float
    min_lexical_absolute: float | None
    hybrid_rerank: bool
    query_rewrite: bool
    hybrid_cosine_weight: float = 0.62

    @staticmethod
    def from_env(final_top_k: int) -> RagRetrievalConfig:
        fk = max(1, int(final_top_k))
        rk_env = (os.environ.get("LLM_AGENT_RAG_RETRIEVE_K") or "").strip()
        retrieve_k = int(rk_env) if rk_env.isdigit() else max(fk * 4, 16)
        retrieve_k = max(retrieve_k, fk)

        ms = (os.environ.get("LLM_AGENT_RAG_MIN_SIM") or "").strip()
        min_sim: float | None = 0.22
        if ms:
            if ms.lower() in ("none", "off", "0", "-1"):
                min_sim = None
            else:
                try:
                    min_sim = float(ms)
                except ValueError:
                    min_sim = 0.22

        legacy = (os.environ.get("LLM_AGENT_RAG_LEGACY") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if legacy:
            return RagRetrievalConfig.legacy(fk)

        qr = (os.environ.get("LLM_AGENT_RAG_QUERY_REWRITE") or "").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

        hyb = (os.environ.get("LLM_AGENT_RAG_HYBRID") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

        ratio_s = (os.environ.get("LLM_AGENT_RAG_MIN_LEX_RATIO") or "").strip()
        ratio = 0.12
        if ratio_s:
            try:
                ratio = float(ratio_s)
            except ValueError:
                pass

        abs_s = (os.environ.get("LLM_AGENT_RAG_MIN_LEX_ABS") or "").strip()
        abs_lex: float | None = None
        if abs_s:
            try:
                abs_lex = float(abs_s)
            except ValueError:
                abs_lex = None

        return RagRetrievalConfig(
            retrieve_k=retrieve_k,
            final_top_k=fk,
            min_similarity=min_sim,
            min_lexical_ratio=ratio,
            min_lexical_absolute=abs_lex,
            hybrid_rerank=hyb,
            query_rewrite=qr,
        )

    @staticmethod
    def legacy(final_top_k: int) -> RagRetrievalConfig:
        fk = max(1, int(final_top_k))
        return RagRetrievalConfig(
            retrieve_k=fk,
            final_top_k=fk,
            min_similarity=None,
            min_lexical_ratio=0.0,
            min_lexical_absolute=None,
            hybrid_rerank=False,
            query_rewrite=False,
        )


@dataclass(frozen=True)
class RagRetrievalOutcome:
    """Результат поиска чанков: попадания + достаточность контекста для ответа «по документам»."""

    hits: list[dict[str, Any]]
    context_sufficient: bool
    best_score: float
    weak_reason: str


@dataclass(frozen=True)
class RagAugmentResult:
    """Готовый user-текст для API и метаданные retrieval (для валидации ответа ассистента)."""

    prompt: str
    hits: list[dict[str, Any]]
    context_sufficient: bool
    best_score: float
    weak_reason: str


@dataclass(frozen=True)
class RagGroundingCheck:
    """Проверка структуры ответа (источники, цитаты) и привязки цитат к текстам чанков."""

    has_answer_section: bool
    has_sources_section: bool
    has_quotes_section: bool
    quotes_nonempty: bool
    quotes_verbatim_in_chunks: bool
    notes: str


def _hit(
    text: str,
    meta: dict[str, Any],
    *,
    score: float,
    score_kind: str,
    retrieval_score: float | None = None,
) -> dict[str, Any]:
    h: dict[str, Any] = {
        "text": text,
        "metadata": dict(meta),
        "score": float(score),
        "score_kind": score_kind,
    }
    if retrieval_score is not None:
        h["retrieval_score"] = float(retrieval_score)
    return h


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
                _hit(str(txt), dict(meta), score=s, score_kind="lexical"),
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
                _hit(
                    str(ch.get("text") or ""),
                    dict(meta),
                    score=0.0,
                    score_kind="lexical_fallback",
                )
            )
        return out
    k = max(1, min(int(top_k), len(scored)))
    return [t[2] for t in scored[:k]]


def _lexical_hits_sorted(
    question: str, chunks_raw: list[dict[str, Any]]
) -> tuple[list[str], dict[str, float], list[tuple[float, int, dict[str, Any]]]]:
    """Все чанки с лексическим score, отсортированы по убыванию."""
    terms = _lexical_match_terms(question)
    texts_lower = [str(ch.get("text") or "").lower() for ch in chunks_raw]
    idf: dict[str, float] = {}
    for t in terms:
        df = sum(1 for txt in texts_lower if t in txt)
        idf[t] = math.log((len(texts_lower) + 1.0) / (df + 1.0)) + 1.0

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, (ch, low) in enumerate(zip(chunks_raw, texts_lower, strict=True)):
        txt = str(ch.get("text") or "")
        s = _lexical_chunk_score_final(terms, low, idf, len(txt))
        meta = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
        scored.append((s, idx, _hit(str(txt), dict(meta), score=s, score_kind="lexical")))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return terms, idf, scored


def _apply_lexical_second_stage(
    question: str,
    chunks_raw: list[dict[str, Any]],
    cfg: RagRetrievalConfig,
) -> list[dict[str, Any]]:
    _terms, _idf_all, scored = _lexical_hits_sorted(question, chunks_raw)
    if not scored or scored[0][0] <= 0.0:
        return _search_top_chunks_lexical(question, chunks_raw, top_k=cfg.final_top_k)

    r = max(1, min(cfg.retrieve_k, len(scored)))
    pool = scored[:r]
    max_s = pool[0][0] if pool else 0.0
    floor_abs = cfg.min_lexical_absolute
    if floor_abs is None:
        floor_abs = max_s * cfg.min_lexical_ratio
    filtered = [(s, i, h) for s, i, h in pool if s >= floor_abs]
    if not filtered:
        filtered = pool[: max(1, min(cfg.final_top_k, len(pool)))]
    filtered.sort(key=lambda x: (-x[0], x[1]))
    k = max(1, min(cfg.final_top_k, len(filtered)))
    return [t[2] for t in filtered[:k]]


def _cosine_candidates(
    qvec: list[float],
    chunks_raw: list[dict[str, Any]],
    dim_q: int,
    retrieve_k: int,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for ch in chunks_raw:
        emb = ch.get("embedding")
        if not isinstance(emb, list) or len(emb) != dim_q:
            continue
        vec = [float(x) for x in emb]
        s = _cosine_sim(qvec, vec)
        meta = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
        text = str(ch.get("text") or "")
        scored.append(
            (
                s,
                _hit(text, dict(meta), score=s, score_kind="cosine", retrieval_score=s),
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    r = max(1, min(retrieve_k, len(scored)))
    return [t[1] for t in scored[:r]]


def _hybrid_rerank_pool(
    question: str,
    pool: list[dict[str, Any]],
    *,
    w_cos: float,
) -> list[dict[str, Any]]:
    terms = _lexical_match_terms(question)
    texts_lower = [str(h.get("text") or "").lower() for h in pool]
    idf = _lexical_idf_for_corpus(terms, texts_lower)
    raw_lex = [
        _lexical_chunk_score_final(terms, low, idf, len(str(h.get("text") or "")))
        for h, low in zip(pool, texts_lower, strict=True)
    ]
    lex_n = _normalize_lex_scores(raw_lex)
    w_l = 1.0 - w_cos
    out: list[dict[str, Any]] = []
    for h, cos0, ln in zip(pool, [float(h.get("retrieval_score", h.get("score", 0.0))) for h in pool], lex_n, strict=True):
        hybrid = w_cos * cos0 + w_l * ln
        meta = h.get("metadata") if isinstance(h.get("metadata"), dict) else {}
        out.append(
            _hit(
                str(h.get("text") or ""),
                dict(meta),
                score=hybrid,
                score_kind="cosine_lexical_hybrid",
                retrieval_score=cos0,
            )
        )
    out.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return out


def _filter_by_floor(hits: list[dict[str, Any]], floor: float | None) -> list[dict[str, Any]]:
    if floor is None:
        return list(hits)
    return [h for h in hits if float(h.get("score") or 0.0) >= floor]


def load_index_chunks(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    chunks = raw.get("chunks") or []
    emb_meta = raw.get("embedding") or {}
    if not isinstance(chunks, list):
        raise ValueError("Поле chunks в индексе должно быть списком.")
    return chunks, emb_meta


def _rag_allow_weak_context_env() -> bool:
    v = (os.environ.get("LLM_AGENT_RAG_ALLOW_WEAK_CONTEXT") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


# Совпадения только по этим токенам часто дают ложную «релевантность» к длинной памятке.
_RAG_GENERIC_LEXEMES: frozenset[str] = frozenset(
    """
    турист туриста туристов туристск туристы
    памятке памятки памятк
    документ документа документы
    ключ ключи
    этом этим этой этих этого
    нужен нужна нужно нужны
    запрос запросу запросы
    информаци информации информаций
    взять взят
    """.split()
)


def _lexical_context_sufficient(question: str, hits: list[dict[str, Any]]) -> tuple[bool, str]:
    """
    Дополнительная проверка для lexical/dummy: пересечение с объединением текста топ-3 чанков;
    отсев случаев, когда совпадают только общеупотребительные лексемы из длинного гайда.
    """
    if not hits:
        return False, "no_hits"
    if any(str(h.get("score_kind")) == "lexical_fallback" for h in hits):
        return False, "lexical_fallback"
    terms = _lexical_match_terms(question)
    blob = "\n".join(
        str((hits[i] or {}).get("text") or "").lower() for i in range(min(3, len(hits)))
    )
    if not terms:
        return False, "no_query_terms"
    matched = [t for t in terms if t in blob]
    if len(terms) <= 2:
        ok = len(matched) >= len(terms)
        return (ok, "" if ok else "partial_term_match")
    if len(matched) < 2:
        return False, "few_term_hits_in_top_chunks"
    ratio = len(matched) / len(terms)
    if ratio < 0.32:
        return False, "low_term_coverage"
    if len(terms) >= 4 and all(t in _RAG_GENERIC_LEXEMES for t in matched):
        return False, "only_generic_term_overlap"
    return True, ""


def _hit_meta_dict(h: dict[str, Any]) -> dict[str, Any]:
    m = h.get("metadata")
    return dict(m) if isinstance(m, dict) else {}


def _chunk_source_label(meta: dict[str, Any]) -> str:
    raw = str(meta.get("source") or meta.get("path") or "").strip()
    if not raw:
        return "(источник не указан)"
    try:
        return Path(raw).name
    except (OSError, TypeError, ValueError):
        return raw


def _norm_ws_substring(needle: str, haystack: str) -> bool:
    n = " ".join((needle or "").split())
    if len(n) < 8:
        return n in (haystack or "")
    h = " ".join((haystack or "").split())
    return n in h


def retrieve_for_rag(
    question: str,
    index_path: Path | str,
    *,
    top_k: int = 5,
    dummy_embeddings: bool | None = None,
    config: RagRetrievalConfig | None = None,
) -> RagRetrievalOutcome:
    """
    Поиск чанков + флаг context_sufficient.

    Если релевантность ниже порога (семантика: всё отсечено min_similarity; лексика: lexical_fallback),
    context_sufficient=False — в промпте не подставляются тексты документов, модель обязана ответить
    «не знаю» (см. build_rag_grounded_user_message). Переменная LLM_AGENT_RAG_ALLOW_WEAK_CONTEXT=1
    возвращает старое поведение: считать контекст достаточным и показывать чанки даже при fallback.
    """
    path = resolve_rag_index_path(index_path)
    cfg = config if config is not None else RagRetrievalConfig.from_env(top_k)
    q_raw = (question or "").strip()
    qtext = rewrite_query_for_rag(q_raw) if cfg.query_rewrite else q_raw

    chunks_raw, emb_meta = load_index_chunks(path)
    if not chunks_raw:
        return RagRetrievalOutcome(
            hits=[], context_sufficient=False, best_score=0.0, weak_reason="empty_index"
        )
    dummy = (
        dummy_embeddings if dummy_embeddings is not None else _dummy_from_index_meta(emb_meta)
    )
    relax = _rag_allow_weak_context_env()

    if dummy:
        hits = _apply_lexical_second_stage(qtext, chunks_raw, cfg)
        lex_ok, lex_reason = _lexical_context_sufficient(qtext, hits)
        weak = not lex_ok
        best = max((float(h.get("score") or 0.0) for h in hits), default=0.0)
        reason = lex_reason if weak else ""
        ok = (not weak) or (relax and bool(hits))
        return RagRetrievalOutcome(
            hits=hits,
            context_sufficient=ok,
            best_score=best,
            weak_reason="" if ok else (reason or "lexical_weak"),
        )

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
        return RagRetrievalOutcome(
            hits=[], context_sufficient=False, best_score=0.0, weak_reason="no_embeddings_in_index"
        )
    if len(first_emb) != dim_q:
        raise ValueError(
            f"Размерность эмбеддинга запроса ({dim_q}) не совпадает с индексом ({len(first_emb)}). "
            "Соберите индекс тем же провайдером эмбеддингов (см. document_index.embeddings)."
        )

    pool = _cosine_candidates(q, chunks_raw, dim_q, cfg.retrieve_k)
    if cfg.hybrid_rerank and len(pool) > 1:
        pool = _hybrid_rerank_pool(qtext, pool, w_cos=cfg.hybrid_cosine_weight)
    before_floor = list(pool)
    pool_filtered = _filter_by_floor(pool, cfg.min_similarity)
    weak = not pool_filtered
    if weak:
        pool = before_floor[: max(1, min(cfg.final_top_k, len(before_floor)))]
    else:
        pool = pool_filtered
    k = max(1, min(cfg.final_top_k, len(pool)))
    hits = pool[:k]
    best = float(hits[0].get("score") or 0.0) if hits else 0.0
    reason = "similarity_below_threshold" if weak else ""
    ok = (not weak) or (relax and bool(hits))
    return RagRetrievalOutcome(
        hits=hits,
        context_sufficient=ok,
        best_score=best,
        weak_reason="" if ok else reason,
    )


def search_top_chunks(
    question: str,
    index_path: Path,
    *,
    top_k: int = 5,
    dummy_embeddings: bool | None = None,
    config: RagRetrievalConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Поиск top-k чанков: при семантических эмбеддингах — косинус к вектору вопроса;
    при индексе с deterministic_hash — лексический ранг (подстроки запроса + IDF по чанкам),
    т.к. такие векторы не пригодны для смыслового поиска.

    Двухэтапный режим (по умолчанию, см. RagRetrievalConfig.from_env):
    шире retrieve_k → гибридный реранк → порог min_similarity → final_top_k.

    Передать RagRetrievalConfig.legacy(top_k) для старого поведения (один этап, без порога).

    Каждый элемент: text, metadata, score, опционально score_kind, retrieval_score.
    """
    return retrieve_for_rag(
        question,
        index_path,
        top_k=top_k,
        dummy_embeddings=dummy_embeddings,
        config=config,
    ).hits


def build_rag_grounded_user_message(question: str, outcome: RagRetrievalOutcome) -> str:
    """
    User-сообщение для chat completion: правила формата (ответ / источники / цитаты) + выдержки
    или режим «недостаточный контекст» без текста документов.
    """
    q = (question or "").strip()
    lines: list[str] = [
        "Ты ассистент с доступом только к приведённым ниже выдержкам из документов (RAG).",
        "",
        "ФОРМАТ ОТВЕТА (обязательно, три раздела с заголовками ровно в таком виде):",
        "### Ответ",
        "<содержательный ответ>",
        "### Источники",
        "<список строк: источник (имя файла) | раздел: … | chunk_id: … — только для выдержек, на которые опирался ответ>",
        "### Цитаты",
        "Каждая цитата — отдельная строка, начинается с «>», затем опционально [chunk_id] и дословный фрагмент из выдержки с этим chunk_id (до ~500 символов).",
        "",
        "Анти-галлюцинации:",
        "- Любой факт из раздела «### Ответ» должен иметь опору в «### Цитаты»; цитата — копия текста из выдержки с тем же chunk_id.",
        "- Нельзя выдумывать chunk_id, источники и цитаты: только из списка выдержек ниже.",
        "- Нельзя дополнять ответ знаниями вне выдержек, если контекст помечен как достаточный.",
        "",
    ]

    if not outcome.context_sufficient:
        lines.extend(
            [
                "СТАТУС КОНТЕКСТА: релевантность найденных фрагментов запросу ниже внутреннего порога "
                "(или совпадений по запросу не найдено). Тексты документов намеренно НЕ приводятся.",
                "",
                "Обязательное поведение:",
                "- В «### Ответ» прямо напиши, что по имеющимся документам ты **не знаешь** надёжного ответа; "
                "попроси пользователя уточнить вопрос, указать тему или фрагмент документа.",
                "- В «### Источники» укажи одну строку вида: нет надёжных источников — релевантность ниже порога.",
                "- В «### Цитаты» укажи: нет — цитировать нечего (контекст недостаточен).",
                "- Не выдумывай содержание документов и не заполняй ответ общеизвестными фактами вместо документов.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Ниже — выдержки. Ответ строится **только** на них.",
                "",
            ]
        )

    if outcome.context_sufficient and outcome.hits:
        for i, h in enumerate(outcome.hits, start=1):
            meta = _hit_meta_dict(h)
            section, cid = (
                str(meta.get("section") or "").strip() or "(раздел не указан)",
                str(meta.get("chunk_id") or "").strip() or "(chunk_id не указан)",
            )
            src = _chunk_source_label(meta)
            score = float(h.get("score") or 0.0)
            kind = str(h.get("score_kind") or "cosine")
            rscore = h.get("retrieval_score")
            if kind == "lexical":
                head = f"--- Выдержка {i} | chunk_id: {cid} | источник: {src} | раздел: {section} | лексика: {score:.2f}"
            elif kind == "lexical_fallback":
                head = f"--- Выдержка {i} | chunk_id: {cid} | источник: {src} | раздел: {section} | лексический fallback"
            elif kind == "cosine_lexical_hybrid":
                head = (
                    f"--- Выдержка {i} | chunk_id: {cid} | источник: {src} | раздел: {section} "
                    f"| гибрид: {score:.4f}"
                )
                if rscore is not None:
                    head += f" | косинус: {float(rscore):.4f}"
            else:
                head = (
                    f"--- Выдержка {i} | chunk_id: {cid} | источник: {src} | раздел: {section} "
                    f"| сходство: {score:.4f}"
                )
            lines.append(head)
            lines.append(str(h.get("text") or "").strip())
            lines.append("")
    elif outcome.context_sufficient and not outcome.hits:
        lines.append("(Индекс не содержит чанков.)")
        lines.append("")

    lines.append("--- Вопрос пользователя")
    lines.append(q)
    return "\n".join(lines).strip()


def merge_question_with_chunks(question: str, hits: list[dict[str, Any]]) -> str:
    """Текст пользовательского сообщения для API: выдержки + вопрос (формат с источниками и цитатами)."""
    qrw = rewrite_query_for_rag(question.strip())
    lex_ok, lex_reason = _lexical_context_sufficient(qrw, hits)
    weak = not lex_ok
    best = max((float(h.get("score") or 0.0) for h in hits), default=0.0)
    reason = lex_reason if weak else ""
    if _rag_allow_weak_context_env() and hits:
        weak = False
    outcome = RagRetrievalOutcome(
        hits=hits,
        context_sufficient=not weak,
        best_score=best,
        weak_reason="" if not weak else reason,
    )
    return build_rag_grounded_user_message(question, outcome)


def _slice_after_heading(text: str, heading: str) -> str | None:
    needle = f"### {heading}"
    i = text.find(needle)
    if i < 0:
        return None
    start = i + len(needle)
    rest = text[start:]
    m = re.search(r"^###\s+\S", rest, flags=re.MULTILINE)
    if m:
        return rest[: m.start()].strip()
    return rest.strip()


def parse_rag_grounding_sections(reply: str) -> dict[str, str]:
    """Выделяет три раздела Markdown по заголовкам ### Ответ / ### Источники / ### Цитаты."""
    t = reply or ""
    return {
        "answer": (_slice_after_heading(t, "Ответ") or "").strip(),
        "sources": (_slice_after_heading(t, "Источники") or "").strip(),
        "quotes": (_slice_after_heading(t, "Цитаты") or "").strip(),
    }


def validate_rag_grounding_reply(
    reply: str,
    hits: list[dict[str, Any]],
    *,
    context_sufficient: bool,
) -> RagGroundingCheck:
    """Проверяет наличие разделов и что цитаты — подстроки текстов чанков (по chunk_id, если указан)."""
    sec = parse_rag_grounding_sections(reply)
    has_a = bool(sec["answer"])
    has_s = bool(sec["sources"])
    has_q = bool(sec["quotes"])
    quote_lines = [ln for ln in sec["quotes"].splitlines() if ln.strip().startswith(">")]
    quotes_nonempty = bool(quote_lines) if context_sufficient else bool(sec["quotes"].strip())
    if not context_sufficient:
        ok_weak = (
            has_a
            and has_s
            and has_q
            and (
                "не знаю" in sec["answer"].lower()
                or "недостаточн" in sec["answer"].lower()
            )
        )
        return RagGroundingCheck(
            has_answer_section=has_a,
            has_sources_section=has_s,
            has_quotes_section=has_q,
            quotes_nonempty=quotes_nonempty,
            quotes_verbatim_in_chunks=True,
            notes="weak_context_mode" + ("" if ok_weak else ";ожидалось_явное_не_знаю_или_недостаточно"),
        )

    by_id: dict[str, str] = {}
    all_texts: list[str] = []
    for h in hits:
        meta = _hit_meta_dict(h)
        cid = str(meta.get("chunk_id") or "").strip()
        txt = str(h.get("text") or "")
        all_texts.append(txt)
        if cid:
            by_id[cid] = txt

    quote_lines = [ln for ln in sec["quotes"].splitlines() if ln.strip().startswith(">")]
    if not quote_lines:
        return RagGroundingCheck(
            has_answer_section=has_a,
            has_sources_section=has_s,
            has_quotes_section=has_q,
            quotes_nonempty=False,
            quotes_verbatim_in_chunks=False,
            notes="нет_строк_цитат_с_>",
        )

    verbatim_ok = True
    for ln in quote_lines:
        m = re.match(r"^\s*>\s*(?:\[([^\]]+)\])?\s*(.+)$", ln.strip())
        if not m:
            verbatim_ok = False
            continue
        cid_opt, qtext = m.group(1) or "", (m.group(2) or "").strip()
        if len(qtext) < 12:
            continue
        hay = by_id.get(cid_opt.strip(), "") if cid_opt.strip() else "\n".join(all_texts)
        if not _norm_ws_substring(qtext, hay):
            verbatim_ok = False

    return RagGroundingCheck(
        has_answer_section=has_a,
        has_sources_section=has_s,
        has_quotes_section=has_q,
        quotes_nonempty=quotes_nonempty,
        quotes_verbatim_in_chunks=verbatim_ok,
        notes="strong_context",
    )


def augment_user_message_with_rag(
    question: str,
    index_path: Path | str,
    *,
    top_k: int = 5,
    dummy_embeddings: bool | None = None,
    config: RagRetrievalConfig | None = None,
) -> RagAugmentResult:
    """
    Вопрос → поиск чанков → объединение с вопросом.
    Возвращает готовый user-текст, попадания и флаг достаточности контекста.
    """
    path = resolve_rag_index_path(index_path)
    out = retrieve_for_rag(
        question,
        path,
        top_k=top_k,
        dummy_embeddings=dummy_embeddings,
        config=config,
    )
    prompt = build_rag_grounded_user_message(question, out)
    return RagAugmentResult(
        prompt=prompt,
        hits=out.hits,
        context_sufficient=out.context_sufficient,
        best_score=out.best_score,
        weak_reason=out.weak_reason,
    )


def compare_rag_modes(
    question: str,
    index_path: Path,
    *,
    top_k: int = 5,
    dummy_embeddings: bool | None = None,
) -> dict[str, Any]:
    """
    Сводка для A/B: legacy (без порога/реранка/rewrite) vs улучшенный пайплайн.
    Удобно для отчёта и отладки retrieval.
    """
    path = resolve_rag_index_path(index_path)
    legacy_hits = search_top_chunks(
        question,
        path,
        top_k=top_k,
        dummy_embeddings=dummy_embeddings,
        config=RagRetrievalConfig.legacy(top_k),
    )
    enhanced_hits = search_top_chunks(
        question,
        path,
        top_k=top_k,
        dummy_embeddings=dummy_embeddings,
        config=RagRetrievalConfig.from_env(top_k),
    )

    def _compact(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for h in hits:
            meta = h.get("metadata") if isinstance(h.get("metadata"), dict) else {}
            sec = str(meta.get("section") or "")[:80]
            txt = str(h.get("text") or "")
            out.append(
                {
                    "score": float(h.get("score") or 0.0),
                    "score_kind": str(h.get("score_kind") or ""),
                    "retrieval_score": h.get("retrieval_score"),
                    "section": sec,
                    "text_preview": txt[:160].replace("\n", " "),
                }
            )
        return out

    cfg_e = RagRetrievalConfig.from_env(top_k)
    q_enhanced_search = (
        rewrite_query_for_rag(question) if cfg_e.query_rewrite else question.strip()
    )

    return {
        "question": question.strip(),
        "query_after_rewrite": q_enhanced_search,
        "index": str(path),
        "top_k": top_k,
        "retrieve_k_default": cfg_e.retrieve_k,
        "legacy": {"hits": _compact(legacy_hits), "n": len(legacy_hits)},
        "enhanced": {"hits": _compact(enhanced_hits), "n": len(enhanced_hits)},
        "overlap_sections": _overlap_metric(legacy_hits, enhanced_hits),
    }


def _overlap_metric(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    def _sig(h: dict[str, Any]) -> str:
        meta = h.get("metadata") if isinstance(h.get("metadata"), dict) else {}
        sec = str(meta.get("section") or "")
        t = str(h.get("text") or "")[:120]
        return f"{sec}|{t}"

    sa = {_sig(h) for h in a}
    sb = {_sig(h) for h in b}
    inter = sa & sb
    union = sa | sb
    jacc = len(inter) / len(union) if union else 1.0
    return {
        "jaccard_text_signature": round(jacc, 4),
        "intersection_count": len(inter),
        "legacy_only": len(sa - sb),
        "enhanced_only": len(sb - sa),
    }
