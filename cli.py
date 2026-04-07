#!/usr/bin/env python3
"""
Тонкий интерфейс: только ввод/вывод. Вся работа с API — в LLMAgent.
Стратегия контекста: --context-strategy или LLM_AGENT_CONTEXT_STRATEGY.
"""

from __future__ import annotations

import argparse

from agent import LLMAgent, RunResult, clear_history_file
from context_strategies import ContextStrategyKind


def _print_run_result(result: RunResult) -> None:
    print(result.text)
    if result.stats is not None:
        line = result.stats.format_line()
        if line:
            print(line)


def _strategy_from_arg(raw: str | None) -> ContextStrategyKind | None:
    if raw is None or raw.strip() == "":
        return None
    return ContextStrategyKind(raw.strip())


def _format_memory_proposals(items: list[dict[str, str]]) -> str:
    if not items:
        return ""
    lines = ["Найдены кандидаты для памяти:"]
    for i, it in enumerate(items, start=1):
        mem_type = it.get("type", "")
        section = it.get("section", "")
        key = it.get("key", "")
        value = it.get("value", "")
        target = mem_type if not section else f"{mem_type}.{section}"
        lines.append(f"  {i}) {target} -> {key}: {value}")
    lines.append("Сохранить? [y/N]")
    return "\n".join(lines)


def _handle_slash_command(agent: LLMAgent, line: str) -> str | None:
    """
    Обрабатывает команды, начинающиеся с /.
    Возвращает текст для печати; None — передать строку в agent.run как обычное сообщение.
    """
    s = line.strip()
    if not s.startswith("/"):
        return None
    parts = s.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    # Алиас: /show memory -> /memory show
    if cmd == "/show" and arg.lower() == "memory":
        cmd = "/memory"
        arg = "show"

    if cmd in ("/help", "/?"):
        return (
            "Команды:\n"
            "  /strategy [sliding_window|sticky_facts|branching] — текущая или смена стратегии\n"
            "  /memory show — показать все типы памяти\n"
            "  /memory put short_term <key> <value> — явное сохранение в краткосрочную память\n"
            "  /memory put working <key> <value> — явное сохранение в рабочую память\n"
            "  /memory put long_term <profile|decisions|knowledge> <key> <value> — явное сохранение в долговременную\n"
            "  /facts — показать блок facts (sticky_facts)\n"
            "  /fact ключ остальное_текстом_значение — добавить/обновить факт вручную\n"
            "  /split или /checkpoint — checkpoint и две ветки (только branching)\n"
            "  /switch или /checkout <branch_a|branch_b> — смена активной ветки\n"
            "  /status — состояние ветвления\n"
            "  /help — эта справка"
        )

    if cmd == "/strategy":
        if not arg:
            return f"Текущая стратегия: {agent.context_strategy.value}"
        try:
            agent.set_context_strategy(arg)
        except ValueError as e:
            return f"Неизвестная стратегия: {e}"
        return f"Стратегия переключена на: {agent.context_strategy.value}"

    if cmd == "/facts":
        return agent.format_facts_lines()

    if cmd == "/memory":
        if not arg:
            return (
                "Формат:\n"
                "  /memory show\n"
                "  /memory put short_term <key> <value>\n"
                "  /memory put working <key> <value>\n"
                "  /memory put long_term <profile|decisions|knowledge> <key> <value>"
            )
        mem_parts = arg.split(maxsplit=4)
        sub = mem_parts[0].lower()
        if sub == "show":
            return agent.format_memory_lines()
        if sub != "put":
            return "Подкоманда /memory должна быть show или put."
        if len(mem_parts) < 4:
            return "Недостаточно аргументов: /memory put <type> ..."
        mem_type = mem_parts[1]
        if mem_type in ("long_term", "long"):
            if len(mem_parts) < 5:
                return (
                    "Для long_term нужно: /memory put long_term "
                    "<profile|decisions|knowledge> <key> <value>"
                )
            section = mem_parts[2]
            key = mem_parts[3]
            value = mem_parts[4]
            ok, err = agent.save_memory_entry(
                "long_term", key, value, long_term_section=section
            )
            if not ok:
                return f"Не удалось сохранить: {err}"
            return f"Сохранено в long_term.{section}: {key}"
        key = mem_parts[2]
        value = mem_parts[3] if len(mem_parts) == 4 else mem_parts[3] + " " + mem_parts[4]
        ok, err = agent.save_memory_entry(mem_type, key, value)
        if not ok:
            return f"Не удалось сохранить: {err}"
        return f"Сохранено в {mem_type}: {key}"

    if cmd == "/fact":
        if not arg:
            return (
                "Формат: /fact ключ текст значения (всё после первого слова — значение).\n"
                "Пример: /fact цель выполнить домашнее задание"
            )
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            return "Укажите и ключ, и значение: /fact ключ значение"
        ok, err = agent.merge_fact(parts[0], parts[1])
        if not ok:
            return err
        v = parts[1]
        tail = "…" if len(v) > 200 else ""
        return f"Записано: {parts[0]} = {v[:200]}{tail}"

    if cmd in ("/split", "/checkpoint"):
        ok, err = agent.split_dialog_branches()
        if ok:
            return "Checkpoint: созданы ветки branch_a и branch_b. Активна branch_a."
        return f"Не удалось: {err}"

    if cmd in ("/switch", "/checkout"):
        if not arg:
            return "Укажите ветку: /switch branch_b или /checkout branch_b"
        ok, err = agent.switch_dialog_branch(arg)
        if ok:
            return f"Активная ветка: {arg}"
        return f"Не удалось: {err}"

    if cmd == "/status":
        return agent.branching_status_line()

    # Не наша команда — пусть уйдёт в модель (например /path/to/file)
    return None


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
    p.add_argument(
        "--context-strategy",
        choices=[x.value for x in ContextStrategyKind],
        default=None,
        help=(
            "Стратегия контекста (иначе LLM_AGENT_CONTEXT_STRATEGY, по умолчанию из файла "
            "или sliding_window)"
        ),
    )
    args = p.parse_args()
    if args.reset_history:
        clear_history_file()

    strat = _strategy_from_arg(args.context_strategy)
    agent = LLMAgent(context_strategy=strat)

    if args.query is not None:
        _print_run_result(agent.run(args.query))
        return

    print(
        "Пустая строка — выход. Ctrl+D — выход.\n"
        f"{agent.branching_status_line()}\n"
        "Команды: /help\n"
    )
    while True:
        try:
            user = input("Сообщение: ").rstrip("\n\r")
        except EOFError:
            break
        if user == "":
            break
        if user.startswith("/"):
            out = _handle_slash_command(agent, user)
            if out is not None:
                print(out)
                print()
                continue
        proposals = agent.propose_memory_entries(user)
        if proposals:
            print(_format_memory_proposals(proposals))
            choice = input("> ").strip().lower()
            if choice in ("y", "yes", "д", "да"):
                saved, errors = agent.apply_memory_proposals(proposals)
                print(f"Сохранено в память: {saved}")
                if errors:
                    print("Ошибки сохранения:")
                    for e in errors:
                        print(f"  - {e}")
                print()
        _print_run_result(agent.run(user))
        print()


if __name__ == "__main__":
    main()
