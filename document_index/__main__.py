"""
CLI: индексация документа и сравнение стратегий чанкинга.

Примеры:
  python -m document_index --input memory/note.txt --out-dir memory/index_out
    (нужны OPENAI_API_KEY или пакет sentence-transformers из requirements.txt)
  python -m document_index --input memory/note.txt --dummy-embeddings --out-dir memory/index_out
  python -m document_index --input memory/note.txt --compare-only --comparison-out memory/chunking_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def main() -> int:
    if load_dotenv:
        load_dotenv()

    p = argparse.ArgumentParser(description="Индексация документа: чанки, эмбеддинги, JSON.")
    p.add_argument("--input", type=Path, default=Path("memory/note.txt"), help="Путь к текстовому файлу")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("memory/index_out"),
        help="Каталог для index_fixed_size.json, index_structure.json, chunking_comparison.json",
    )
    p.add_argument("--compare-only", action="store_true", help="Только отчёт сравнения чанкинга, без эмбеддингов")
    p.add_argument("--comparison-out", type=Path, default=None, help="Путь для JSON сравнения (переопределяет out-dir)")
    p.add_argument("--fixed-size", type=int, default=1500)
    p.add_argument("--fixed-overlap", type=int, default=200)
    p.add_argument("--max-section-chars", type=int, default=4000)
    p.add_argument("--section-overlap", type=int, default=200)
    p.add_argument(
        "--dummy-embeddings",
        action="store_true",
        help="Детерминированные векторы без API/HF (только проверка формата индекса)",
    )
    args = p.parse_args()
    inp = args.input.resolve()
    if not inp.is_file():
        print(f"Файл не найден: {inp}", file=sys.stderr)
        return 1

    from document_index.pipeline import (
        build_index_for_file,
        compare_chunking_strategies,
        save_comparison_report,
    )

    comparison = compare_chunking_strategies(
        inp,
        fixed_chunk_size=args.fixed_size,
        fixed_overlap=args.fixed_overlap,
        max_section_chars=args.max_section_chars,
        section_sub_overlap=args.section_overlap,
    )
    comp_path = args.comparison_out
    if comp_path is None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        comp_path = args.out_dir / "chunking_comparison.json"
    save_comparison_report(comparison, comp_path)
    print(f"Сравнение стратегий записано: {comp_path}")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))

    if args.compare_only:
        return 0

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for strat in ("fixed_size", "structure"):
        out_json = out_dir / f"index_{strat}.json"
        print(f"Индексация ({strat}) → {out_json} ...")
        try:
            build_index_for_file(
                inp,
                strategy=strat,  # type: ignore[arg-type]
                out_path=out_json,
                fixed_chunk_size=args.fixed_size,
                fixed_overlap=args.fixed_overlap,
                max_section_chars=args.max_section_chars,
                section_sub_overlap=args.section_overlap,
                dummy_embeddings=args.dummy_embeddings,
            )
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            print(
                "\nВарианты:\n"
                "  1) Установить локальные эмбеддинги: pip install sentence-transformers\n"
                "     (или из корня проекта: pip install -r requirements.txt)\n"
                "  2) Задать OPENAI_API_KEY для эмбеддингов через совместимый API\n"
                "  3) Офлайн без смыслового поиска: добавьте флаг --dummy-embeddings\n",
                file=sys.stderr,
            )
            return 1
        print(f"Готово: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
