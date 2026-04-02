#!/usr/bin/env python3
"""
Тонкий интерфейс: только ввод/вывод. Вся работа с API — в LLMAgent.
"""

import argparse
import sys

from agent import LLMAgent, clear_history_file


def main() -> None:
    p = argparse.ArgumentParser(description="Простой CLI-чат с LLM через агента")
    p.add_argument(
        "-q",
        "--query",
        help="Один запрос без интерактивного режима",
    )
    p.add_argument(
        "--reset-history",
        action="store_true",
        help="Очистить сохранённую историю диалога и начать новый чат",
    )
    args = p.parse_args()
    if args.reset_history:
        clear_history_file()
    agent = LLMAgent()

    if args.query is not None:
        out = agent.run(args.query)
        print(out)
        return

    print(
        "Пустая строка — выход. Ctrl+D — выход.\n"
    )
    while True:
        try:
            user = input("Сообщение: ").rstrip("\n\r")
        except EOFError:
            break
        if user == "":
            break
        print(agent.run(user))
        print()


if __name__ == "__main__":
    main()



