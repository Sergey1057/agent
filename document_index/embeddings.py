from __future__ import annotations

import hashlib
import json
import math
import os
import random
import ssl
import urllib.error
import urllib.request
from typing import Any

import certifi


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def embed_texts_openai(
    texts: list[str],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[list[list[float]], dict[str, Any]]:
    """
    Эмбеддинги через OpenAI-compatible POST /v1/embeddings.
    Переменные окружения: OPENAI_API_KEY, опционально OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Задайте OPENAI_API_KEY для генерации эмбеддингов (или установите sentence-transformers "
            "и вызовите embed_texts_local из кода)."
        )
    url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
    embed_url = f"{url}/embeddings"
    m = model or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    payload = json.dumps({"model": m, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        embed_url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, context=_ssl_context(), timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Embeddings HTTP {e.code}: {body}") from e

    data = raw.get("data") or []
    # OpenAI returns sorted by index
    rows = sorted(data, key=lambda x: int(x.get("index", 0)))
    vectors = [list(map(float, r["embedding"])) for r in rows]
    meta = {
        "provider": "openai_compatible",
        "model": m,
        "dims": len(vectors[0]) if vectors else 0,
    }
    return vectors, meta


def embed_texts_deterministic(
    texts: list[str],
    *,
    dim: int = 256,
) -> tuple[list[list[float]], dict[str, Any]]:
    """
    Детерминированные нормализованные векторы из хеша текста (без сети; не для семантического поиска).
    """
    out: list[list[float]] = []
    for t in texts:
        seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out, {"provider": "deterministic_hash", "model": f"sha256_seed_gaussian_{dim}", "dims": dim}


def embed_texts_auto(
    texts: list[str],
    *,
    batch_size: int = 64,
    dummy: bool = False,
) -> tuple[list[list[float]], dict[str, Any]]:
    """
    Пакетная генерация: при dummy — детерминированные векторы; иначе при OPENAI_API_KEY — OpenAI;
    иначе sentence-transformers (локально).
    """
    if dummy:
        return embed_texts_deterministic(texts)
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return embed_texts_openai(texts)
    try:
        return embed_texts_local_batches(texts, batch_size=batch_size)
    except ImportError as e:
        raise RuntimeError(
            "Нет OPENAI_API_KEY и не установлен sentence-transformers. "
            "Установите ключ API, выполните: pip install sentence-transformers, "
            "или передайте --dummy-embeddings для офлайн-проверки пайплайна."
        ) from e
    except Exception as e:
        raise RuntimeError(
            "Локальные эмбеддинги недоступны (загрузка модели или сеть). "
            "Задайте OPENAI_API_KEY, настройте доступ к Hugging Face, "
            "или передайте --dummy-embeddings для офлайн-проверки пайплайна."
        ) from e


def embed_texts_local_batches(
    texts: list[str],
    *,
    batch_size: int = 32,
    model_name: str | None = None,
) -> tuple[list[list[float]], dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    name = model_name or os.environ.get("SENTENCE_TRANSFORMERS_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    model = SentenceTransformer(name)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        out.extend([row.astype(float).tolist() for row in emb])
    dims = len(out[0]) if out else 0
    return out, {"provider": "sentence_transformers", "model": name, "dims": dims}
