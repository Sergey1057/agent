#!/usr/bin/env python3
"""
Тонкий интерфейс: только ввод/вывод. Вся работа с API — в LLMAgent.
Стратегия контекста: --context-strategy или LLM_AGENT_CONTEXT_STRATEGY.
"""

from __future__ import annotations

import argparse
import json
import time

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
            "  /task ... — состояние задачи как FSM (этап/шаг/ожидаемое действие, pause/resume + явные переходы)\n"
            "  /invariants — инварианты проекта (архитектура, стек, бизнес-правила; отдельно от диалога)\n"
            "  /github <owner>/<repo> [вопрос] — вызвать MCP GitHub tool; при вопросе ответить с учётом результата\n"
            "  /schedule ... — MCP-планировщик: reminder, периодический сбор и summary\n"
            "  /mcp flow <owner>/<repo> <query> [| file_path] — длинный orchestration flow через несколько MCP-серверов\n"
            "  /mcp auto <текст запроса> — авто-выбор MCP-инструмента и сервера по policy\n"
            "  /pipeline <query> [| file_path] — MCP-цепочка search -> summorize -> saveToFile\n"
            "  /rag [on|off|top N] [путь_к_index.json] — RAG: выдержки из индекса в запрос к LLM\n"
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
                "  /task stage <planning|plan_approved|execution|validation|done>\n"
                "  /task step <текущий шаг>\n"
                "  /task expect <ожидаемое действие>\n"
                "  /task next\n"
                "  /task approve-plan | start-execution | start-validation | finalize\n"
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
                return (
                    "Формат: /task stage "
                    "<planning|plan_approved|execution|validation|done>"
                )
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
        if sub in ("approve-plan", "approve_plan", "approve"):
            ok, err = agent.task_set_stage("plan_approved")
            if not ok:
                return err
            return agent.format_task_state_lines()
        if sub in ("start-execution", "start_execution", "execute"):
            ok, err = agent.task_set_stage("execution")
            if not ok:
                return err
            return agent.format_task_state_lines()
        if sub in ("start-validation", "start_validation", "validate"):
            ok, err = agent.task_set_stage("validation")
            if not ok:
                return err
            return agent.format_task_state_lines()
        if sub in ("finalize", "finish"):
            ok, err = agent.task_set_stage("done")
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
        return (
            "Неизвестная подкоманда /task. Используйте: "
            "status/start/stage/step/expect/next/approve-plan/start-execution/"
            "start-validation/finalize/pause/resume."
        )

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

    if cmd == "/github":
        if not arg:
            return (
                "Формат: /github <owner>/<repo> [вопрос]\n"
                "Пример: /github python/cpython какие ключевые метрики?"
            )
        parts = arg.split(maxsplit=1)
        repo_ref = parts[0].strip()
        question = parts[1].strip() if len(parts) > 1 else ""
        if "/" not in repo_ref:
            return "Ожидается owner/repo, например: python/cpython"
        owner, repo = repo_ref.split("/", 1)
        payload = agent.fetch_github_repo_via_mcp(owner, repo, include_readme=False)
        if payload.get("status") != "ok":
            return f"MCP error: {payload.get('error', 'unknown')}"
        if not question:
            return f"MCP result:\n{payload}"
        enriched = (
            "Используй данные из MCP GitHub инструмента при ответе.\n"
            f"Данные: {payload}\n"
            f"Вопрос пользователя: {question}"
        )
        result = agent.run(enriched)
        return result.text

    if cmd == "/schedule":
        if not arg:
            return (
                "Формат:\n"
                "  /schedule list\n"
                "  /schedule run [limit]\n"
                "  /schedule summary [hours]\n"
                "  /schedule report [hours]\n"
                "  /schedule pipeline <query> [| file_path]\n"
                "  /schedule add reminder <name> <delay_sec> <message>\n"
                "  /schedule add collector <name> <interval_sec> <source>\n"
                "  /schedule add summary <name> <interval_sec> <message>"
            )
        tw = arg.split(maxsplit=1)
        sub = tw[0].strip().lower()
        rest = tw[1].strip() if len(tw) > 1 else ""
        if sub == "list":
            payload = agent.scheduler_list_tasks_via_mcp(include_inactive=True)
            return json.dumps(payload, ensure_ascii=False, indent=2)
        if sub == "run":
            limit = 20
            if rest:
                try:
                    limit = max(1, int(rest))
                except ValueError:
                    return "run: limit должен быть числом."
            payload = agent.scheduler_run_due_via_mcp(limit=limit)
            return json.dumps(payload, ensure_ascii=False, indent=2)
        if sub == "summary":
            hours = 24
            if rest:
                try:
                    hours = max(1, int(rest))
                except ValueError:
                    return "summary: hours должен быть числом."
            payload = agent.scheduler_summary_via_mcp(hours=hours)
            return json.dumps(payload, ensure_ascii=False, indent=2)
        if sub in ("report", "human-summary", "human_summary"):
            hours = 24
            if rest:
                try:
                    hours = max(1, int(rest))
                except ValueError:
                    return "report: hours должен быть числом."
            payload = agent.scheduler_human_summary_via_mcp(hours=hours)
            if payload.get("status") != "ok":
                return json.dumps(payload, ensure_ascii=False, indent=2)
            text = str(payload.get("summary_text") or "").strip()
            if text:
                return text
            return json.dumps(payload, ensure_ascii=False, indent=2)
        if sub == "pipeline":
            if not rest:
                return (
                    "Формат: /schedule pipeline <query> [| file_path]\n"
                    "Пример: /schedule pipeline MCP pipeline данные | memory/pipeline_result.txt"
                )
            query = rest
            file_path = ""
            if "|" in rest:
                left, right = rest.split("|", 1)
                query = left.strip()
                file_path = right.strip()
            query = query.strip()
            if not query:
                return "pipeline: нужен query."
            payload = agent.scheduler_run_tools_pipeline_via_mcp(
                query=query,
                file_path=file_path,
                limit=5,
            )
            return json.dumps(payload, ensure_ascii=False, indent=2)
        if sub == "add":
            parts = rest.split(maxsplit=4)
            if len(parts) < 4:
                return (
                    "Формат:\n"
                    "  /schedule add reminder <name> <delay_sec> <message>\n"
                    "  /schedule add collector <name> <interval_sec> <source>\n"
                    "  /schedule add summary <name> <interval_sec> <message>"
                )
            kind_alias = parts[0].strip().lower()
            name = parts[1].strip()
            try:
                seconds = max(0, int(parts[2]))
            except ValueError:
                return "Параметр delay/interval должен быть числом секунд."
            text = parts[3] if len(parts) == 4 else parts[3] + " " + parts[4]
            if kind_alias in ("reminder", "remind"):
                payload = agent.scheduler_upsert_task_via_mcp(
                    name=name,
                    kind="reminder",
                    delay_seconds=seconds,
                    payload={"message": text},
                )
                return json.dumps(payload, ensure_ascii=False, indent=2)
            if kind_alias in ("collector", "collect", "data_collection"):
                payload = agent.scheduler_upsert_task_via_mcp(
                    name=name,
                    kind="data_collection",
                    delay_seconds=0,
                    interval_seconds=seconds,
                    payload={"source": text},
                )
                return json.dumps(payload, ensure_ascii=False, indent=2)
            if kind_alias == "summary":
                payload = agent.scheduler_upsert_task_via_mcp(
                    name=name,
                    kind="summary",
                    delay_seconds=0,
                    interval_seconds=seconds,
                    payload={"message": text},
                )
                return json.dumps(payload, ensure_ascii=False, indent=2)
            return "Неизвестный тип: используйте reminder | collector | summary."
        return "Неизвестная подкоманда /schedule."

    if cmd == "/pipeline":
        if not arg:
            return (
                "Формат: /pipeline <query> [| file_path]\n"
                "Пример: /pipeline MCP pipeline данные | memory/pipeline_result.txt"
            )
        query = arg
        file_path = ""
        if "|" in arg:
            left, right = arg.split("|", 1)
            query = left.strip()
            file_path = right.strip()
        query = query.strip()
        if not query:
            return "Нужен query: /pipeline <query> [| file_path]"
        payload = agent.scheduler_run_tools_pipeline_via_mcp(
            query=query,
            file_path=file_path,
            limit=5,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    if cmd == "/rag":
        parts = arg.split()
        if not parts:
            return (
                f"{agent.rag_status_line()}\n\n"
                "Формат:\n"
                "  /rag on [путь/index.json] — включить (путь опционален, если уже задан)\n"
                "  /rag off — выключить\n"
                "  /rag top <N> — число чанков (N ≥ 1)\n"
                "Переменные окружения: LLM_AGENT_RAG, LLM_AGENT_RAG_INDEX, LLM_AGENT_RAG_TOP_K"
            )
        sub = parts[0].lower()
        if sub == "off":
            agent.set_rag(False)
            return agent.rag_status_line()
        if sub == "on":
            path_arg = parts[1] if len(parts) > 1 else None
            if path_arg:
                agent.set_rag(True, index_path=path_arg)
            else:
                agent.set_rag(True)
            return agent.rag_status_line()
        if sub == "top":
            if len(parts) < 2:
                return "Формат: /rag top <N>"
            try:
                agent.set_rag_top_k(int(parts[1]))
            except ValueError:
                return "N должно быть целым числом."
            return agent.rag_status_line()
        return "Неизвестная подкоманда /rag. См. /rag без аргументов."

    if cmd == "/mcp":
        if not arg:
            return (
                "Формат:\n"
                "  /mcp flow <owner>/<repo> <query> [| file_path]\n"
                "  /mcp auto <текст запроса>\n"
                "Пример: /mcp flow python/cpython mcp orchestration тест | memory/pipeline_from_cli.txt"
            )
        tw = arg.split(maxsplit=2)
        if tw[0].strip().lower() == "auto":
            request = tw[1].strip() if len(tw) > 1 else ""
            if len(tw) > 2:
                request = request + " " + tw[2].strip()
            if not request.strip():
                return "Формат: /mcp auto <текст запроса>"
            payload = agent.route_mcp_request(request.strip())
            return json.dumps(payload, ensure_ascii=False, indent=2)
        if len(tw) < 3 or tw[0].strip().lower() != "flow":
            return (
                "Поддерживается: /mcp flow ... и /mcp auto ..."
            )
        repo_ref = tw[1].strip()
        query_part = tw[2].strip()
        if "/" not in repo_ref:
            return "Ожидается owner/repo, например: python/cpython"
        owner, repo = repo_ref.split("/", 1)
        query = query_part
        file_path = ""
        if "|" in query_part:
            left, right = query_part.split("|", 1)
            query = left.strip()
            file_path = right.strip()
        if not query:
            return "Нужен query: /mcp flow <owner>/<repo> <query> [| file_path]"
        payload = agent.run_multi_server_mcp_flow(
            owner=owner,
            repo=repo,
            query=query,
            file_path=file_path,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

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
    p.add_argument(
        "--github-repo",
        default="",
        help="Вызвать MCP GitHub tool для owner/repo (пример: python/cpython)",
    )
    p.add_argument(
        "--github-ask",
        default="",
        help="Если задано с --github-repo: вопрос к агенту с использованием MCP результата",
    )
    p.add_argument(
        "--schedule-daemon",
        action="store_true",
        help="Запустить 24/7 цикл scheduler MCP: run_due + периодический summary",
    )
    p.add_argument(
        "--schedule-interval-sec",
        type=int,
        default=30,
        help="Интервал проверки due-задач в daemon-режиме",
    )
    p.add_argument(
        "--schedule-summary-every-sec",
        type=int,
        default=300,
        help="Как часто печатать агрегированную сводку в daemon-режиме",
    )
    p.add_argument(
        "--mcp-flow-repo",
        default="",
        help="Запустить длинный MCP flow для owner/repo (пример: python/cpython)",
    )
    p.add_argument(
        "--mcp-flow-query",
        default="",
        help="Запрос для длинного MCP flow",
    )
    p.add_argument(
        "--mcp-flow-file-path",
        default="memory/pipeline_from_cli.txt",
        help="Файл для saveToFile внутри MCP flow",
    )
    p.add_argument(
        "--mcp-auto",
        default="",
        help="Запустить policy-роутер MCP по текстовому запросу",
    )
    p.add_argument(
        "--rag",
        action="store_true",
        help="Включить RAG: к запросу подмешиваются релевантные чанки из JSON-индекса",
    )
    p.add_argument(
        "--no-rag",
        action="store_true",
        help="Выключить RAG даже при LLM_AGENT_RAG=1",
    )
    p.add_argument(
        "--rag-index",
        default="",
        help="Путь к index_*.json (иначе LLM_AGENT_RAG_INDEX)",
    )
    p.add_argument(
        "--rag-top-k",
        type=int,
        default=None,
        help="Сколько чанков подмешивать (иначе LLM_AGENT_RAG_TOP_K или 5)",
    )
    args = p.parse_args()
    if args.reset_history:
        clear_history_file()

    strat = _strategy_from_arg(args.context_strategy)
    rag_kw: bool | None = None
    if args.no_rag:
        rag_kw = False
    elif args.rag:
        rag_kw = True
    rag_index_kw = args.rag_index.strip() or None
    agent = LLMAgent(
        context_strategy=strat,
        rag_enabled=rag_kw,
        rag_index_path=rag_index_kw,
        rag_top_k=args.rag_top_k,
    )

    if args.github_repo:
        if "/" not in args.github_repo:
            raise SystemExit("--github-repo должен быть в формате owner/repo")
        owner, repo = args.github_repo.split("/", 1)
        payload = agent.fetch_github_repo_via_mcp(owner, repo, include_readme=False)
        print(f"MCP result: {payload}")
        if args.github_ask.strip():
            enriched = (
                "Используй данные из MCP GitHub инструмента при ответе.\n"
                f"Данные: {payload}\n"
                f"Вопрос пользователя: {args.github_ask.strip()}"
            )
            _print_run_result(agent.run(enriched))
        return

    if args.query is not None:
        _print_run_result(agent.run(args.query))
        return

    if args.mcp_flow_repo:
        if "/" not in args.mcp_flow_repo:
            raise SystemExit("--mcp-flow-repo должен быть в формате owner/repo")
        if not args.mcp_flow_query.strip():
            raise SystemExit("--mcp-flow-query обязателен вместе с --mcp-flow-repo")
        owner, repo = args.mcp_flow_repo.split("/", 1)
        payload = agent.run_multi_server_mcp_flow(
            owner=owner,
            repo=repo,
            query=args.mcp_flow_query.strip(),
            file_path=args.mcp_flow_file_path.strip(),
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.mcp_auto.strip():
        payload = agent.route_mcp_request(args.mcp_auto.strip())
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.schedule_daemon:
        poll_sec = max(1, int(args.schedule_interval_sec))
        summary_sec = max(1, int(args.schedule_summary_every_sec))
        next_summary_at = time.time()
        print(
            "Scheduler daemon запущен (Ctrl+C для остановки).\n"
            f"Интервал run_due: {poll_sec} сек; summary: каждые {summary_sec} сек.\n"
        )
        try:
            while True:
                run_payload = agent.scheduler_run_due_via_mcp(limit=50)
                executed_count = int(run_payload.get("executed_count", 0) or 0)
                if executed_count > 0:
                    print(f"[scheduler] Выполнено задач: {executed_count}")
                    print(json.dumps(run_payload, ensure_ascii=False, indent=2))
                now_ts = time.time()
                if now_ts >= next_summary_at:
                    summary_payload = agent.scheduler_summary_via_mcp(hours=24)
                    text_payload = agent.scheduler_human_summary_via_mcp(hours=24)
                    text = str(text_payload.get("summary_text") or "").strip()
                    print("[scheduler] Summary (24h):")
                    if text:
                        print(text)
                    else:
                        print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
                    next_summary_at = now_ts + summary_sec
                time.sleep(poll_sec)
        except KeyboardInterrupt:
            print("\nScheduler daemon остановлен.")
        return

    print(
        "Пустая строка — выход. Ctrl+D — выход.\n"
        f"{agent.branching_status_line()}\n"
        f"{agent.rag_status_line()}\n"
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
