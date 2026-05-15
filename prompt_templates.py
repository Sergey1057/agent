"""
Шаблоны промптов: RAG user-message и system для local backend.
Файлы с плейсхолдерами, команда /prompt в чате, флаги CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from document_index.rag import RagRetrievalOutcome

PLACEHOLDER_CONTEXT_RULES = "{{CONTEXT_RULES}}"
PLACEHOLDER_EXCERPTS = "{{EXCERPTS}}"
PLACEHOLDER_QUESTION = "{{QUESTION}}"
PLACEHOLDER_MODEL = "{{MODEL}}"

DEFAULT_LOCAL_SYSTEM_TEMPLATE = (
    "Ты локальная языковая модель «{{MODEL}}», запущенная через LM Studio на этом компьютере. "
    "Это не GigaChat и не облачный ChatGPT. "
    "Отвечай по существу на вопросы пользователя на том языке, на котором он пишет. "
    "Не отказывайся от обычных информационных вопросов."
)

DEFAULT_RAG_USER_TEMPLATE = """Ты ассистент с доступом только к приведённым ниже выдержкам из документов (RAG).

ФОРМАТ ОТВЕТА (обязательно, три раздела с заголовками ровно в таком виде):
### Ответ
<содержательный ответ>
### Источники
<список строк: источник (имя файла) | раздел: … | chunk_id: … — только для выдержек, на которые опирался ответ>
### Цитаты
Каждая цитата — отдельная строка, начинается с «>», затем опционально [chunk_id] и дословный фрагмент из выдержки с этим chunk_id (до ~500 символов).

Анти-галлюцинации:
- Любой факт из раздела «### Ответ» должен иметь опору в «### Цитаты»; цитата — копия текста из выдержки с тем же chunk_id.
- Нельзя выдумывать chunk_id, источники и цитаты: только из списка выдержек ниже.
- Нельзя дополнять ответ знаниями вне выдержек, если контекст помечен как достаточный.

{{CONTEXT_RULES}}

{{EXCERPTS}}

--- Вопрос пользователя
{{QUESTION}}"""


@dataclass
class PromptOverrides:
    rag_template_text: str | None = None
    rag_template_path: Path | None = None
    local_system_text: str | None = None
    local_system_path: Path | None = None

    @classmethod
    def from_env(
        cls,
        *,
        rag_prompt_file: str | Path | None = None,
        local_prompt_file: str | Path | None = None,
    ) -> PromptOverrides:
        rag_path = _resolve_path(
            rag_prompt_file or os.environ.get("LLM_AGENT_RAG_PROMPT_FILE")
        )
        local_path = _resolve_path(
            local_prompt_file or os.environ.get("LLM_AGENT_LOCAL_PROMPT_FILE")
        )
        if local_path is None:
            default_local = example_prompt_path("local")
            if default_local.is_file():
                local_path = default_local
        rag_text = _read_optional_file(rag_path)
        local_text = _read_optional_file(local_path)
        return cls(
            rag_template_text=rag_text,
            rag_template_path=rag_path,
            local_system_text=local_text,
            local_system_path=local_path,
        )

    def format_status_lines(self) -> list[str]:
        rag_s = (
            f"файл: {self.rag_template_path}"
            if self.rag_template_path
            else ("свой текст" if self.rag_template_text else "встроенный по умолчанию")
        )
        loc_s = (
            f"файл: {self.local_system_path}"
            if self.local_system_path
            else ("свой текст" if self.local_system_text else "встроенный по умолчанию")
        )
        return [
            f"Промпт RAG: {rag_s}",
            f"Промпт local system: {loc_s}",
        ]

    def format_help_block(self) -> str:
        return (
            "  /prompt — статус шаблонов\n"
            "  /prompt rag load <путь.txt> — шаблон RAG (плейсхолдеры ниже)\n"
            "  /prompt rag reset | /prompt reset\n"
            "  /prompt local load <путь.txt> — system для backend=local ({{MODEL}})\n"
            "  /prompt local reload — перечитать файл (memory/prompts/local_system.txt или последний load)\n"
            "  /prompt local reset\n"
            "  /prompt example rag | local — путь к примеру в memory/prompts/\n"
            f"  Плейсхолдеры RAG: {PLACEHOLDER_CONTEXT_RULES}, "
            f"{PLACEHOLDER_EXCERPTS}, {PLACEHOLDER_QUESTION}\n"
            f"  Local: {PLACEHOLDER_MODEL}\n"
            "  Env: LLM_AGENT_RAG_PROMPT_FILE, LLM_AGENT_LOCAL_PROMPT_FILE\n"
            "  CLI: --rag-prompt-file, --local-prompt-file"
        )


def _resolve_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _read_optional_file(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Файл шаблона не найден: {path}")
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def read_prompt_file(path: str | Path) -> str:
    p = _resolve_path(path)
    if p is None:
        raise ValueError("Пустой путь к файлу шаблона")
    text = _read_optional_file(p)
    if not text:
        raise ValueError(f"Файл шаблона пуст: {p}")
    return text


def render_local_system_message(model_name: str, template: str | None) -> str:
    tpl = (template or DEFAULT_LOCAL_SYSTEM_TEMPLATE).strip()
    return tpl.replace(PLACEHOLDER_MODEL, model_name)


def render_rag_user_prompt(
    question: str,
    outcome: RagRetrievalOutcome,
    *,
    template: str | None = None,
) -> str:
    from document_index.rag import (
        format_rag_context_rules,
        format_rag_excerpts_block,
    )

    q = (question or "").strip()
    rules = format_rag_context_rules(outcome)
    excerpts = format_rag_excerpts_block(outcome)
    tpl = (template or DEFAULT_RAG_USER_TEMPLATE).strip()

    if PLACEHOLDER_CONTEXT_RULES in tpl:
        body = tpl.replace(PLACEHOLDER_CONTEXT_RULES, rules)
    else:
        body = rules + "\n\n" + tpl if rules else tpl

    if PLACEHOLDER_EXCERPTS in body:
        body = body.replace(PLACEHOLDER_EXCERPTS, excerpts)
    elif excerpts.strip():
        body = body.rstrip() + "\n\n" + excerpts

    if PLACEHOLDER_QUESTION in body:
        body = body.replace(PLACEHOLDER_QUESTION, q)
    else:
        body = body.rstrip() + "\n\n--- Вопрос пользователя\n" + q

    return body.strip()


def example_prompt_path(kind: str) -> Path:
    base = Path(__file__).resolve().parent / "memory" / "prompts"
    name = "rag_grounded.txt" if kind == "rag" else "local_system.txt"
    return base / name


def handle_prompt_slash_command(agent: Any, arg: str) -> str:
    a = (arg or "").strip()
    if not a or a.lower() in ("show", "status"):
        return "\n".join(agent.prompt_status_lines())

    parts = a.split(maxsplit=1)
    sub = parts[0].lower().replace("_", "-")
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub == "reset":
        agent.reset_prompt_templates()
        return "Шаблоны сброшены.\n" + "\n".join(agent.prompt_status_lines())

    if sub == "example":
        which = rest.lower() or "rag"
        if which not in ("rag", "local"):
            return "Формат: /prompt example rag | local"
        p = example_prompt_path(which)
        if not p.is_file():
            return f"Пример не найден: {p}"
        return f"Пример шаблона ({which}):\n{p}\n\n---\n{p.read_text(encoding='utf-8')[:2000]}"

    if sub == "rag":
        if not rest:
            return "Формат: /prompt rag load <путь> | /prompt rag reset"
        rparts = rest.split(maxsplit=1)
        rsub = rparts[0].lower()
        rpath = rparts[1].strip() if len(rparts) > 1 else ""
        if rsub == "reset":
            agent.reset_rag_prompt_template()
            return "\n".join(agent.prompt_status_lines())
        if rsub == "load":
            if not rpath:
                return "Формат: /prompt rag load <путь.txt>"
            try:
                agent.set_rag_prompt_file(rpath)
            except (OSError, ValueError) as e:
                return str(e)
            return "\n".join(agent.prompt_status_lines())
        return "Подкоманды rag: load <путь>, reset"

    if sub == "local":
        if not rest:
            return (
                "Формат: /prompt local load <путь> | reload | reset\n"
                f"Файл по умолчанию: {example_prompt_path('local')}"
            )
        rparts = rest.split(maxsplit=1)
        rsub = rparts[0].lower()
        rpath = rparts[1].strip() if len(rparts) > 1 else ""
        if rsub == "reload":
            try:
                agent.reload_local_prompt_template()
            except (OSError, ValueError) as e:
                return str(e)
            return "\n".join(agent.prompt_status_lines())
        if rsub == "reset":
            agent.reset_local_prompt_template()
            return "\n".join(agent.prompt_status_lines())
        if rsub == "load":
            if not rpath:
                return "Формат: /prompt local load <путь.txt>"
            try:
                agent.set_local_prompt_file(rpath)
            except (OSError, ValueError) as e:
                return str(e)
            return "\n".join(agent.prompt_status_lines())
        return "Подкоманды local: load <путь>, reset"

    return "Неизвестная подкоманда.\n" + PromptOverrides().format_help_block()
