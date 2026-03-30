#!/usr/bin/env python3
"""
Тонкий интерфейс: только ввод/вывод. Вся работа с API — в LLMAgent.
"""

import argparse
import sys

from agent import LLMAgent


def main() -> None:
    p = argparse.ArgumentParser(description="Простой CLI-чат с LLM через агента")
    p.add_argument(
        "-q",
        "--query",
        help="Один запрос без интерактивного режима",
    )
    args = p.parse_args()
    agent = LLMAgent()

    if args.query is not None:
        out = agent.run(args.query)
        print(out)
        return

    print("Интерактивный режим. Пустая строка — выход. Ctrl+D — выход.\n")
    for line in sys.stdin:
        user = line.rstrip("\n\r")
        if user == "":
            break
        print(agent.run(user))
        print()


if __name__ == "__main__":
    main()
