"""Текстовый режим инструмента: без JSON-схемы в запросе.

Шаблон tool calling у Qwen добавляет к каждому вызову ~270 входных токенов (замер scripts/measure_prompt.py:
условие 94 токена, условие + схема sh 364). Поэтому модели схему не передаём: она отвечает блоком ```bash```,
а рейл превращает его в обычный вызов инструмента sh. Цикл ReActAgent, исполнение инструмента и рейлы
openJiuwen работают как раньше.

Историю модели тоже отдаём текстом: если в ней остаются tool_calls, модель отвечает нативным <tool_call>,
а прокси без списка инструментов вырезает его, и ответ приходит пустым (scripts/debug_textmode.py).
Поэтому перед отправкой вызовы sh становятся репликами assistant с ```bash```, а результаты — репликами user.
"""
from __future__ import annotations

import json
import re
import uuid

from openjiuwen.core.foundation.llm.schema.message import AssistantMessage, UserMessage
from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall
from openjiuwen.core.single_agent.rail.base import AgentRail

_FENCE = re.compile(r"```[ \t]*(?:bash|sh|shell|console)?[ \t]*\n(.*?)```", re.S)
# Команда без огораживания: модель иногда пишет её голой строкой.
_BARE = re.compile(r"^(?:python3?|grep|cat|head|tail|sed|awk|echo|printf|ls|find|wc|sort|cut|cd)\b")


def _strip_prompt(cmd: str) -> str:
    """Убирает приглашение шелла «$ » в начале строк: модель копирует его из вывода предзапуска."""
    return "\n".join(ln[2:] if ln.startswith("$ ") else ln for ln in cmd.splitlines()).strip()


def extract_command(text: str | None) -> str | None:
    if not text:
        return None
    m = _FENCE.search(text)
    if m:
        cmd = _strip_prompt(m.group(1).strip())
        return cmd or None
    t = _strip_prompt(text.strip())
    # Без огораживания модель иногда пишет сначала комментарий с планом, потом команду.
    code = [ln for ln in t.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if code and _BARE.match(code[0].strip()):
        return t
    return None


def is_plan_only(text: str | None) -> bool:
    """Ответ состоит только из строк-комментариев (план цепочки без команды)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return bool(lines) and all(ln.startswith("#") for ln in lines)


# Команды, которые без аргументов ничего не делают: одиночное слово «python» или «cat» — это ответ, не команда.
_NEEDS_ARGS = {"python", "python3", "grep", "cat", "head", "tail", "sed", "awk", "wc", "sort", "cut"}
_FUNCTION_WORDS = {"of", "the", "and", "for", "in", "at", "to", "a", "an", "on", "with"}


def looks_like_command(answer: str | None) -> bool:
    """Финальная реплика — команда или план, а не ответ. Ответ строчными, который лишь начинается с имени
    команды, командой не считаем: «head of sales» (должность), «python» (язык) раньше не записывались вовсе
    (проба 13.11)."""
    a = (answer or "").strip()
    if a.startswith(("#", "$ ")):
        return True
    m = _BARE.match(a)
    if not m:
        return False
    rest = a[m.end():].split()
    if not rest:
        return m.group(0) not in _NEEDS_ARGS
    plain_phrase = all(w.isalpha() for w in rest) and any(w.lower() in _FUNCTION_WORDS for w in rest)
    return not plain_phrase


def _field(m, name):
    return m.get(name) if isinstance(m, dict) else getattr(m, name, None)


def to_text_history(messages):
    """tool_calls -> assistant с ```bash```, ответы инструмента -> user «Output:»."""
    out = []
    for m in messages:
        role = _field(m, "role")
        calls = _field(m, "tool_calls")
        if role == "assistant" and not calls and not (_field(m, "content") or "").strip():
            continue  # пустой ответ, на котором мы переключились на схему инструмента
        if role == "assistant" and calls:
            cmds = []
            for tc in calls:
                args = _field(tc, "arguments") or (_field(_field(tc, "function") or {}, "arguments")) or "{}"
                try:
                    cmds.append(json.loads(args).get("cmd", ""))
                except (ValueError, AttributeError):
                    cmds.append(str(args))
            out.append(AssistantMessage(content="\n".join(f"```bash\n{c}\n```" for c in cmds)))
        elif role == "tool":
            out.append(UserMessage(content=f"Output:\n{_field(m, 'content') or ''}"))
        else:
            out.append(m)
    return out


class TextToolRail(AgentRail):
    """Убирает схему инструментов из запроса и превращает ```bash``` в вызов sh.

    Если модель всё же отвечает нативным <tool_call> (прокси без схемы его вырезает, приходит пустой ответ),
    задача переключается на обычный режим со схемой: native=True и ещё одна итерация ReActAgent.
    """

    priority = 90  # раньше MeterRail: в траекторию попадает уже вызов инструмента

    def __init__(self, tool_name: str = "sh", native: bool = False):
        super().__init__()
        self.tool_name = tool_name
        self.native = native
        self.fallbacks = 0

    async def before_model_call(self, ctx) -> None:
        if not self.native:
            ctx.inputs.tools = []
        # Клиент модели у каждого агента свой; оборачиваем его один раз, чтобы история уходила текстом.
        llm = ctx.agent._get_llm()
        if not getattr(llm, "_text_history_wrapped", False):
            orig = llm.invoke

            async def invoke(*args, **kwargs):
                if "messages" in kwargs:
                    kwargs["messages"] = to_text_history(kwargs["messages"])
                elif args:
                    args = (to_text_history(args[0]), *args[1:])
                return await orig(*args, **kwargs)

            llm.invoke = invoke
            llm._text_history_wrapped = True

    async def after_model_call(self, ctx) -> None:
        resp = ctx.inputs.response
        if resp is None or getattr(resp, "tool_calls", None):
            return
        content = getattr(resp, "content", None)
        usage = getattr(resp, "usage_metadata", None)
        if not (content or "").strip() and usage is not None and (usage.output_tokens or 0) > 0 and not self.native:
            self.native = True
            self.fallbacks += 1
            ctx.request_model_continue()
            return
        if is_plan_only(content):
            # Модель выписала план цепочки, но команду не дала: план остаётся в истории, нужна ещё итерация.
            ctx.request_model_continue()
            return
        cmd = extract_command(content)
        if cmd is None:
            return
        resp.tool_calls = [ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", type="function", name=self.tool_name,
                                    arguments=json.dumps({"cmd": cmd}, ensure_ascii=False))]
        resp.content = ""
