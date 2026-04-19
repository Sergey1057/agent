#!/usr/bin/env python3
"""
Тонкий интерфейс: только ввод/вывод. Вся работа с API — в LLMAgent.
Стратегия контекста: --context-strategy или LLM_AGENT_CONTEXT_STRATEGY.
"""

from __future__ import annotations

import argparse

from agent import LLMAgent, RunResult, clear_history_file
from context_strategies import ContextStrategyKind
from user_profile import parse_profile_set_rest


def _single_profile_id(rest: str) -> tuple[str | None, str | None]:
    """
    Имя профиля для /profile new и /profile copy — одно слово без пробелов,
    иначе весь хвост ошибочно становится имени (как в «new name Сергей style …»).
    """
    s = (rest or "").strip()
    if not s:
        return None, None
    parts = s.split()
    if len(parts) > 1:
        return None, (
            "Имя профиля — одно слово без пробелов. Поля задаются отдельно, например:\n"
            "  /profile new Сергей\n"
            "  /profile set name Сергей style веселый format json constraints не более 100 символов"
        )
    return parts[0], None


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
            "  /profile show | list — профили; set — поля активного; use/new/copy/delete — сценарии\n"
            "  /facts — показать блок facts (sticky_facts)\n"
            "  /fact ключ остальное_текстом_значение — добавить/обновить факт вручную\n"
            "  /split или /checkpoint — checkpoint и две ветки (только branching)\n"
            "  /switch или /checkout <branch_a|branch_b> — смена активной ветки\n"
            "  /status — состояние ветвления\n"
            "  /task ... — состояние задачи как FSM (этап/шаг/ожидаемое действие, pause/resume)\n"
            "  /invariants — инварианты проекта (архитектура, стек, бизнес-правила; отдельно от диалога)\n"
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

    if cmd == "/profile":
        if not arg:
            return (
                "Формат:\n"
                "  /profile show — активный профиль\n"
                "  /profile list — все сохранённые профили\n"
                "  /profile set <поле> <значение> — правки только активного профиля\n"
                "  /profile use <имя>  или  /profile switch <имя> — сменить активный\n"
                "  /profile new <имя> — пустой профиль (имя — одно слово); поля — через /profile set\n"
                "  /profile copy <имя> — копия активного под новым именем (одно слово)\n"
                "  /profile delete <имя> — удалить (не единственный)\n"
                "Поля: display_name, style, format, constraints, extra.<ключ>\n"
                "Несколько полей в строке: … style … format … constraints …"
            )
        tw = arg.split(maxsplit=1)
        sub = tw[0].strip().lower()
        rest = tw[1].strip() if len(tw) > 1 else ""

        if sub == "show":
            return agent.format_user_profile_lines()
        if sub == "list":
            return agent.format_user_profile_list_lines()
        if sub in ("use", "switch"):
            if not rest:
                return "Формат: /profile use <имя>"
            ok, err = agent.activate_user_profile(rest)
            if not ok:
                return err
            return f"Активный профиль: «{rest}»"
        if sub in ("new", "add"):
            if not rest:
                return "Формат: /profile new <имя> — имя одним словом, затем /profile set …"
            pid, err = _single_profile_id(rest)
            if err:
                return err
            assert pid is not None
            ok, err = agent.create_user_profile(pid, copy_from_active=False)
            if not ok:
                return err
            return f"Создан и активирован профиль «{pid}»"
        if sub in ("copy", "duplicate"):
            if not rest:
                return "Формат: /profile copy <имя>"
            pid, err = _single_profile_id(rest)
            if err:
                return err
            assert pid is not None
            ok, err = agent.duplicate_user_profile(pid)
            if not ok:
                return err
            return f"Скопирован активный профиль в «{pid}», он активен"
        if sub in ("delete", "rm"):
            if not rest:
                return "Формат: /profile delete <имя>"
            ok, err = agent.delete_user_profile(rest)
            if not ok:
                return err
            return f"Профиль «{rest}» удалён"
        if sub == "set":
            triple = arg.split(maxsplit=2)
            if len(triple) < 3:
                return "Формат: /profile set <поле> <значение>"
            field_name, value = triple[1].strip(), triple[2]
            assignments, parse_err = parse_profile_set_rest(field_name, value)
            if parse_err:
                return parse_err
            ok, err = agent.set_user_profile_fields(assignments)
            if not ok:
                return err
            keys = ", ".join(sorted(assignments.keys()))
            return f"Профиль обновлён: {keys}"
        return "Неизвестная подкоманда /profile. См. /profile без аргументов."

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

    if cmd == "/task":
        if not arg:
            return (
                "Формат:\n"
                "  /task status\n"
                "  /task start [шаг] [| ожидаемое действие]\n"
                "  /task stage <planning|execution|validation|done>\n"
                "  /task step <текущий шаг>\n"
                "  /task expect <ожидаемое действие>\n"
                "  /task next\n"
                "  /task pause\n"
                "  /task resume"
            )
        tw = arg.split(maxsplit=1)
        sub = tw[0].strip().lower()
        rest = tw[1].strip() if len(tw) > 1 else ""
        if sub == "status":
            return agent.format_task_state_lines()
        if sub == "start":
            step = ""
            expected = ""
            if rest:
                if "|" in rest:
                    left, right = rest.split("|", 1)
                    step = left.strip()
                    expected = right.strip()
                else:
                    step = rest
            agent.task_reset(step=step, expected_action=expected)
            return agent.format_task_state_lines()
        if sub == "stage":
            if not rest:
                return "Формат: /task stage <planning|execution|validation|done>"
            ok, err = agent.task_set_stage(rest)
            if not ok:
                return err
            return agent.format_task_state_lines()
        if sub == "step":
            if not rest:
                return "Формат: /task step <текущий шаг>"
            _, _ = agent.task_set_step(rest)
            return agent.format_task_state_lines()
        if sub == "expect":
            if not rest:
                return "Формат: /task expect <ожидаемое действие>"
            _, _ = agent.task_set_expected_action(rest)
            return agent.format_task_state_lines()
        if sub in ("next", "advance"):
            ok, err = agent.task_advance()
            if not ok:
                return err
            return agent.format_task_state_lines()
        if sub == "pause":
            ok, err = agent.task_pause()
            if not ok:
                return err
            return agent.format_task_state_lines()
        if sub == "resume":
            ok, err = agent.task_resume()
            if not ok:
                return err
            return agent.format_task_state_lines()
        return "Неизвестная подкоманда /task. Используйте: status/start/stage/step/expect/next/pause/resume."

    if cmd == "/invariants":
        if not arg:
            return agent.format_invariants_lines()
        tw = arg.split(maxsplit=1)
        sub = tw[0].strip().lower()
        rest = tw[1].strip() if len(tw) > 1 else ""

        if sub in ("show", "list"):
            return agent.format_invariants_lines()
        if sub == "set":
            triple = arg.split(maxsplit=2)
            if len(triple) < 3:
                return (
                    "Формат: /invariants set <раздел> <текст>\n"
                    "Разделы: architecture (arch), technical_decisions (tech), stack, "
                    "business_rules (business), extra.<ключ>"
                )
            section, value = triple[1].strip(), triple[2]
            ok, err = agent.set_invariant_section(section, value)
            if not ok:
                return err
            return f"Инвариант обновлён: {section}"
        if sub == "clear":
            if not rest:
                return "Формат: /invariants clear <раздел|all> — all очищает все разделы и extra"
            ok, err = agent.clear_invariant_section(rest)
            if not ok:
                return err
            return f"Очищено: {rest}"
        return (
            "Формат:\n"
            "  /invariants | /invariants show — показать\n"
            "  /invariants set <раздел> <текст>\n"
            "  /invariants clear <раздел|all>"
        )

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
