"""Решение одной задачи через ReActAgent из openJiuwen.

Агент минимальный: короткий системный промпт, один инструмент sh, без памяти и планировщика.
Рейл MeterRail считает расход по usage каждого вызова, повторяет вызов при 429, останавливает
агента по появлению answer.txt, по признаку «невыполнимо» и по потолку расхода на задачу.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import random
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from openjiuwen.core.foundation.tool import tool
from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig
from openjiuwen.core.single_agent.rail.base import AgentRail

from .answers import NO_ANSWER_MARK, done_word, normalize_answer, wants_lines
from .budget import Budget
from .coding import (SYSTEM_PROMPT_AUDIT, SYSTEM_PROMPT_CODING, apply_patch, build_audit_prompt, build_prompt,
                     defined_names, extract_blocks, find_layout, parses, run_tests, says_impossible, split_patch)
from .prefetch import prefetch_files, prefetch_search
from . import compute as direct_compute
from . import datasearch
from . import motion
from . import search as direct_search
from .shell import run_bash, run_python
from .tasks import Task
from .texttool import TextToolRail, looks_like_command

# Без фигурных скобок: ReActAgent подставляет {поля} из входа.
SYSTEM_PROMPT = (
    "You are a terse autonomous solver with one tool, sh: it runs a bash command in the current "
    "directory (python3 available). Use as few calls as possible: one command that computes the answer "
    "and writes it in the same command, e.g. python3 -c \"print(...)\" | tee answer.txt. "
    "Never ls or cat whole directories; search with grep -ri. Do not explain. "
    "If the files or the question lack what is needed (missing entity, field, column, value or number), "
    "do not guess: write UNANSWERABLE to answer.txt. "
    "The run ends as soon as answer.txt exists, so write it only with the final answer."
)

# Текстовый режим (TOOL_MODE=text, по умолчанию): схема инструмента модели не передаётся, см. texttool.py.
SYSTEM_PROMPT_TEXT = (
    "You solve one task in the current directory with bash (python3 available). Each reply is either one "
    "```bash block, which is run and its output returned to you, or the final answer alone. "
    "Be brief: prefer one command that computes the answer and writes it, e.g. "
    "python3 -c \"print(...)\" | tee answer.txt; the run ends once answer.txt exists. "
    "Write compact code: no comments, no blank lines, short names, one line if possible. "
    "Never ls or cat whole directories; search with grep -ri. If the data or question lacks what is needed, "
    "do not guess: write UNANSWERABLE to answer.txt. No explanations."
)

# Шаблонная концовка условия; её смысл уже есть в системном промпте, а это ~45 токенов на каждый вызов.
_BOILERPLATE = re.compile(
    r"\s*Work only inside the current directory\.\s*When you are done, write ONLY the final answer to the file "
    r"answer\.txt in the current directory \(for example with bash: echo 42 > answer\.txt\)\.\s*$")

# Подсказки под тип задачи: добавляются к условию только этой задачи, а не к общему промпту.
TYPE_HINTS = {
    "coding": (
        "Hint: hidden tests also check general correctness, so restore the behaviour each function's name "
        "and docstring intend (e.g. true division for an average), not just what the visible tests need. "
        "Read mylib.py and the tests in one command, fix with one python3 edit, rerun the tests."
    ),
    # Без шаблона вида '^# Name': модель копирует его дословно (docs/experiments.md).
    "search": (
        "Hint: grep -ri -A4 with the exact full name prints that record in one call. Look up a referenced "
        "employer or mentor only if the question asks about them (e.g. their city or birth year); if it asks "
        "who they are, the name in the record is the answer. Answer as soon as a record gives the asked value. "
        "Chained questions: start your first command with one comment line spelling out the chain and follow "
        "it exactly, e.g. '# X -> mentor -> employer -> headquarters' for the company where X's mentor works, "
        "'# X -> mentor -> mentor -> born' for the mentor of X's mentor. "
        "Write only the final value to answer.txt (echo Lisbon > answer.txt), never pipe grep output into it."
    ),
    "data_analysis": (
        "Hint: compute with a one-line python3 -c using csv.DictReader (mind the delimiter and header shown "
        "below)."
    ),
}

SH_SCHEMA = {
    "type": "object",
    "properties": {"cmd": {"type": "string", "description": "bash command"}},
    "required": ["cmd"],
}


@dataclass
class Attempt:
    answer: str | None
    result_type: str
    cost: float
    calls: int
    retries: int = 0
    # Сигналы сомнения для второго прохода (budget_agent/second.py), считаются без токенов.
    flags: tuple[str, ...] = ()


_TRANSIENT = ("RateLimitError", "429", "APITimeoutError", "APIConnectionError", "InternalServerError",
              "502", "503", "504", "timed out", "Timeout")
MAX_RETRIES = 6
# Таймаут вызова. Оборванный по таймауту запрос прокси досчитывает и оплачивает, а харнесс его не видит: 13.11
# вызов qwen3.8-27b на 1 406 токенов шёл 67 с при таймауте 60 — по прокси вышло 1 698 + 1 445, по счётчику
# харнесса только 1 445; на заглушке с ответом за 70 с каждая задача давала 4–7 оплаченных запросов и ни одного
# ответа. Обычный вызов 13.11 доходил до 57,5 с, запаса у 60 с не было, а медленнее нашего прокси финальный
# быть может. Поэтому: 120 с на обычный вызов; для моделей с рассуждениями (~21 токен в секунду) таймаут растёт с
# потолком ответа (2 500 токенов — 197 с); после таймаута один повтор, а не шесть.
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", 120))
REASON_TOKENS_PER_S = float(os.environ.get("REASON_TOKENS_PER_S", 15))
TIMEOUT_RETRIES = 1
_TIMEOUT_MARKS = ("APITimeoutError", "timed out", "Timeout")


def is_reasoning(model: str) -> bool:
    return not model.endswith("-noreason")


def call_timeout(model: str, max_tokens: int) -> float:
    if not is_reasoning(model):
        return LLM_TIMEOUT_S
    return max(LLM_TIMEOUT_S, max_tokens / REASON_TOKENS_PER_S + 30.0)


def retry_limit(error_text: str) -> int:
    """Сколько раз повторять вызов после этой ошибки: отклонённый запрос (429, 5xx) токенов не тратит, оборванный
    по таймауту — тратит."""
    return TIMEOUT_RETRIES if any(s in error_text for s in _TIMEOUT_MARKS) else MAX_RETRIES
# Потолок запросов в минуту на весь прогон. В памятке тарифа «Мероприятие» — до 60 запросов в минуту;
# у нас пик доходил до 72 за 60 с (gen-s1, 2 задачи параллельно), и на таком прокси пошли бы 429.
# 50 в минуту — это минимум 3 минуты на 100 задач (~150 вызовов) при лимите в час. 0 — без ограничения.
LLM_RPM = float(os.environ.get("LLM_RPM", 50))


class RateLimiter:
    """Не больше `rpm` запросов к модели за любые `window` секунд, общий на все задачи."""

    def __init__(self, rpm: float, window: float = 60.0):
        self.rpm = int(rpm)
        self.window = window
        self.times: deque[float] = deque()
        self.lock = asyncio.Lock()
        self.waited = 0.0

    async def acquire(self) -> None:
        if self.rpm <= 0:
            return
        async with self.lock:  # ждущие встают в очередь: порядок сохраняется
            while True:
                now = time.monotonic()
                while self.times and now - self.times[0] >= self.window:
                    self.times.popleft()
                if len(self.times) < self.rpm:
                    self.times.append(now)
                    return
                pause = self.window - (now - self.times[0]) + 0.05
                self.waited += pause
                await asyncio.sleep(pause)


RATE = RateLimiter(LLM_RPM)


class MeterRail(AgentRail):
    """Учёт расхода, повтор при 429 и временных сбоях, остановки агента."""

    def __init__(self, task: Task, budget: Budget, model: str, stop_below: float, task_cap: float,
                 traj_path: Path | None = None):
        super().__init__()
        self.task = task
        self.task_id = task.task_id
        self.budget = budget
        self.model = model
        self.stop_below = stop_below
        self.task_cap = task_cap
        self.traj_path = traj_path
        self.calls = 0
        self.retries = 0
        self.rejected = 0
        self._seen: set[int] = set()

    def _traj(self, **row) -> None:
        if self.traj_path is None:
            return
        with self.traj_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    async def after_tool_call(self, ctx) -> None:
        self._traj(kind="tool", name=ctx.inputs.tool_name, args=ctx.inputs.tool_args,
                   result=str(ctx.inputs.tool_result)[:2000])
        # Как только появился answer.txt, заканчиваем без лишнего вызова модели.
        ans_file = self.task.workdir / "answer.txt"
        if not ans_file.exists():
            return
        answer = normalize_answer(ans_file.read_text(encoding="utf-8", errors="replace"),
                                  multiline=wants_lines(self.task.instruction))
        if answer is None:
            return
        if answer == NO_ANSWER_MARK:
            ans_file.unlink()
            self._traj(kind="harness", note="model says unanswerable")
            ctx.request_force_finish({"output": "", "result_type": "unanswerable"})
            return
        if self.task.task_type == "coding" and not await tests_pass(self.task):
            # Модель объявила DONE, но тесты падают: убираем ответ и даём ей продолжить.
            ans_file.unlink()
            self.rejected += 1
            self._traj(kind="harness", note="answer.txt removed: tests still fail")
            return
        self._traj(kind="harness", note=f"finish on answer.txt: {answer!r}")
        ctx.request_force_finish({"output": answer, "result_type": "answer"})

    async def before_model_call(self, ctx) -> None:
        if self.budget.remaining < self.stop_below:
            ctx.request_force_finish({"output": "", "result_type": "budget_stop"})
        elif self.budget.by_task.get(self.task_id, 0.0) >= self.task_cap:
            self._traj(kind="harness", note=f"task cap {self.task_cap:.0f} reached")
            ctx.request_force_finish({"output": "", "result_type": "task_cap"})
        else:
            # Хук срабатывает перед каждой попыткой, включая повторы после 429.
            await RATE.acquire()

    async def on_model_exception(self, ctx) -> None:
        # Отклонённый запрос токенов не тратит, поэтому повторяем с растущей паузой. Таймаут — другое дело: запрос
        # у прокси досчитывается и оплачивается, повтор один (retry_limit).
        text = f"{type(ctx.exception).__name__}: {ctx.exception}"
        if ctx.retry_attempt < retry_limit(text) and any(s in text for s in _TRANSIENT):
            self.retries += 1
            ctx.request_retry(delay_seconds=min(2 ** ctx.retry_attempt, 20) + random.random())

    async def after_model_call(self, ctx) -> None:
        # Хук срабатывает и после неудачной попытки: тогда usage нет, а ответ может быть прежним.
        resp = ctx.inputs.response
        usage = getattr(resp, "usage_metadata", None)
        if usage is None or id(resp) in self._seen:
            return
        self._seen.add(id(resp))
        self.calls += 1
        self.last_completion = int(usage.output_tokens or 0)
        self._traj(kind="model", model=self.model, prompt=usage.input_tokens, completion=usage.output_tokens,
                   content=getattr(resp, "content", None),
                   tool_calls=[{"name": tc.name, "args": tc.arguments} for tc in (resp.tool_calls or [])])
        self.budget.record(
            self.task_id, self.model,
            int(usage.input_tokens or 0), int(usage.output_tokens or 0),
            int(getattr(usage, "reasoning_tokens", 0) or 0),
        )


async def tests_pass(task: Task) -> bool:
    """Прогоняет тесты задачи (команда из условия или по раскладке). Это делает харнесс, токены не тратятся."""
    layout = find_layout(task.workdir, task.instruction)
    if layout is None:
        tests = sorted(task.workdir.glob("test_*.py"))
        if not tests:
            return True
        out = await run_bash(" && ".join(f"python3 {t.name} >/dev/null 2>&1" for t in tests) + " && echo PASS",
                             task.workdir, timeout=120)
        return out.strip().endswith("PASS")
    return (await run_tests(layout)).passed


def build_query(task: Task, text_mode: bool) -> str:
    instr = _BOILERPLATE.sub("", task.instruction) if text_mode else task.instruction
    hint = TYPE_HINTS.get(task.task_type)
    parts = [instr] + ([hint] if hint else [])
    if os.environ.get("PREFETCH", "1") == "1":
        pre = None
        if task.task_type == "search":
            pre = prefetch_search(task)
        elif task.task_type == "data_analysis":
            pre = prefetch_files(task)
        if pre:
            parts.append(pre)
    return "\n".join(parts)


def _new_agent(model: str, system_prompt: str, max_iterations: int, max_tokens: int) -> ReActAgent:
    cfg = (
        ReActAgentConfig()
        .configure_model_client("OpenAI", os.environ["LLM_API_KEY"].strip(),
                                os.environ["LLM_BASE_URL"].strip().rstrip("/"), model)
        .configure_prompt_template([{"role": "system", "content": system_prompt}])
        .configure_max_iterations(max_iterations)
    )
    cfg.model_config_obj.max_tokens = max_tokens
    # По умолчанию у клиента таймаут 360 с, а на задачу у нас 240: повисший запрос убивал всю задачу (дважды
    # 09.11 на медленном прокси). Свой таймаут — call_timeout, повтор делает MeterRail.
    timeout = call_timeout(model, max_tokens)
    cfg.model_client_config.timeout = timeout
    if timeout != LLM_TIMEOUT_S:
        # openJiuwen держит один HTTP-клиент на процесс (ключ и адрес), его таймаут — от первого вызова, то есть
        # 60 с. Без отдельного клиента удлинённый таймаут не действует (проверено заглушкой модели 13.11).
        cfg.model_client_config.use_shared_llm_http_client = False
    cfg.model_client_config.max_retries = 0
    cfg.model_config_obj.temperature = 0
    agent = ReActAgent(AgentCard(name="solver"))
    agent.configure(cfg)
    return agent


FAST_MAX_ROUNDS = 3
FAST_MAX_TOKENS = 1024
# Второй проход для застрявшего coding: модель с рассуждениями. Рассуждения оплачиваются как выход,
# поэтому только из остатка бюджета (run_agent.py) и с потолком на задачу.
REASON_MODEL = "qwen3.8-27b"
REASON_ROUNDS = 1
# Потолок рассуждений: на coding-x1 удачные спасения стоили 500–1 200 (одно 4 200), а на невыполнимой задаче
# модель упиралась в 5 000 и ничего не отвечала.
REASON_MAX_TOKENS = 3000
TEXT_MAX_TOKENS = 400
# Лимит шагов по типу. search с предварительным grep решается за 1–3 вызова (gen-s1), а невыполнимый
# search всегда доходил до лимита: 6 вызовов по ~60. Лимит 4 оставляет запас в один шаг.
TYPE_MAX_ITERATIONS = {"search": 4}


# Исходный код coding-задач (до первой правки): сверка во втором проходе не должна возвращать к нему функции.
ORIGINAL_SOURCES: dict[str, dict[str, str]] = {}


def _function_dumps(text: str) -> dict[str, str]:
    """Функции и методы модуля -> ast.dump тела (без docstring и комментариев)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    out = {}
    for node in tree.body:
        items = [(node.name, node)] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else []
        if isinstance(node, ast.ClassDef):
            items = [(f"{node.name}.{m.name}", m) for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for name, fn in items:
            body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                                   and isinstance(getattr(fn.body[0], "value", None), ast.Constant)) else fn.body
            out[name] = ast.dump(ast.Module(body=body, type_ignores=[]))
    return out


def reverts_to_original(task_id: str, before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Функции, которые правка вернула к исходному коду задачи. Исходный код заведомо с ошибками, а первый проход
    менял эти функции не зря: на hold-c7 сверка вернула `int(...)` вместо `round(...)`, хотя банковское округление
    требовал русский комментарий, и испортила решённую задачу."""
    orig = ORIGINAL_SOURCES.get(task_id)
    if not orig:
        return []
    out = []
    for rel, text in after.items():
        if rel not in orig or text == before.get(rel):
            continue
        o, b, a = _function_dumps(orig[rel]), _function_dumps(before.get(rel, "")), _function_dumps(text)
        out += [f"{rel}:{n}" for n in a if n in o and n in b and a[n] == o[n] and b[n] != o[n]]
    return out


async def solve_coding_fast(task: Task, budget: Budget, model: str, stop_below: float, task_cap: float,
                            traj_path: Path | None, rounds: int = FAST_MAX_ROUNDS,
                            max_tokens: int = FAST_MAX_TOKENS, allow_impossible: bool = False
                            ) -> tuple[str, int, int, bool]:
    """Быстрый путь coding (budget_agent/coding.py). Возвращает (статус, вызовов, повторов, был ли прогресс:
    хоть раз упавших тестов стало меньше — признак того, что задача решаемая и стоит второго прохода).

    Статусы:
    - "pass": тесты прошли;
    - "stuck": модель давала пригодный код, но два раунда подряд упавших тестов не стало меньше.
      Часто это невыполнимая задача (противоречивые или невозможные тесты), обычному агенту не отдаём:
      на отложенном наборе такие задачи съедали ~600 токенов впустую. Если бюджет останется, задачу
      во втором проходе получит модель с рассуждениями (run_agent.py);
    - "n/a": раскладка не та или модель ни разу не дала пригодного кода. Дело в формате, а не в задаче,
      поэтому пробует обычный агент.
    """
    layout = find_layout(task.workdir, task.instruction)
    if layout is None:
        return "n/a", 0, 0, False
    primary = layout.rel(layout.sources[0])
    ORIGINAL_SOURCES.setdefault(task.task_id, layout.read_sources())
    run = await run_tests(layout)
    calls = retries = 0
    usable = improved = False
    stalls = 0
    note = ""  # что не так с прошлым ответом: без подсказки при temperature 0 тот же запрос даст тот же ответ
    doc_rounds = 1  # видимые тесты прошли, а примеры из docstring нет: ещё не больше одного раунда
    for rnd in range(rounds):
        if run.passed and (not run.doc_fail or doc_rounds <= 0):
            break
        if run.passed:
            doc_rounds -= 1
        sources = layout.read_sources()
        agent = _new_agent(model, SYSTEM_PROMPT_CODING, max_iterations=1, max_tokens=max_tokens)
        meter = MeterRail(task, budget, model, stop_below, task_cap, traj_path)
        await agent.register_rail(meter)
        prompt = build_prompt(layout, sources, run, task.instruction, allow_impossible, note)
        res = await agent.invoke({"query": prompt,
                                  "conversation_id": f"{task.task_id}-fast{rnd}-{uuid.uuid4().hex[:6]}"})
        calls, retries = calls + meter.calls, retries + meter.retries
        if not isinstance(res, dict) or res.get("result_type") != "answer":
            break  # бюджет, потолок задачи или ошибка: дальше не тратим
        if allow_impossible and says_impossible(res.get("output")):
            meter._traj(kind="harness", note=f"fast round {rnd} ({model}): model says IMPOSSIBLE")
            return "impossible", calls, retries, improved
        blocks = extract_blocks(res.get("output"))
        new = apply_patch(sources, blocks, primary)
        if new is None:
            if blocks and all(parses(b) for b in blocks):
                # Код разобрался, но ничего не меняет. Один раз переспрашиваем с подсказкой, потом стоп.
                stalls += 1
                meter._traj(kind="harness", note=f"fast round {rnd}: reply changes nothing (stall {stalls})")
                if stalls >= 2:
                    break
                note = NOTE_UNCHANGED
            else:
                # Без подсказки при temperature 0 тот же запрос даёт тот же непригодный ответ (hold-c6: трижды).
                meter._traj(kind="harness", note=f"fast round {rnd}: no usable code in reply")
                note = NOTE_UNUSABLE
            continue
        usable = True
        changed = {rel: text for rel, text in new.items() if text != sources[rel]}
        layout.write_sources(changed)
        new_run = await run_tests(layout)
        how = "whole"
        if not new_run.passed and new_run.key() >= run.key():
            # Целиком не помогло: пробуем принять полезную часть, по одному определению (без токенов).
            layout.write_sources({rel: sources[rel] for rel in changed})
            part = await _apply_greedy(layout, sources, blocks, primary, run)
            if part is not None:
                new, new_run = part
                changed = {rel: text for rel, text in new.items() if text != sources[rel]}
                how = "partial"
            else:
                layout.write_sources(changed)
        meter._traj(kind="harness", note=f"fast round {rnd} ({model}): failures {run.key()} -> {new_run.key()}, "
                                         f"{how} change of {sorted(changed)}")
        names = ", ".join(defined_names(blocks)) or "the code"
        if new_run.key() > run.key() or (run.passed and not new_run.passed):
            layout.write_sources({rel: sources[rel] for rel in changed})  # стало хуже: откатываем правку
            stalls += 1
            note = NOTE_WORSE.format(names=names)
        elif new_run.key() == run.key():
            run = new_run  # правку оставляем, но сводка должна соответствовать коду
            stalls += 1
            note = NOTE_NO_PROGRESS.format(names=names)
        else:
            stalls = 0
            run = new_run
            note = ""
            improved = True
        if stalls >= 2:
            break
    if run.passed:
        return "pass", calls, retries, improved
    return ("stuck" if usable else "n/a"), calls, retries, improved


async def solve_coding_audit(task: Task, budget: Budget, model: str, stop_below: float, task_cap: float,
                             traj_path: Path | None, snapshot_dir: Path | None = None,
                             max_tokens: int = FAST_MAX_TOKENS) -> tuple[str, int]:
    """Второй проход coding: видимые тесты прошли, модель сверяет код с docstring (coding.AUDIT_RULE). Правка
    остаётся, только если видимые тесты и примеры из docstring по-прежнему проходят. Код до правки сохраняется в
    snapshot_dir: по нему bench/score.py считает баллы «без второго прохода». Возвращает (исход, вызовов)."""
    layout = find_layout(task.workdir, task.instruction)
    if layout is None:
        return "n/a", 0
    run = await run_tests(layout)
    if not run.passed:
        return "n/a", 0
    sources = layout.read_sources()
    prompt = build_audit_prompt(sources, task.instruction)
    if prompt is None:
        return "too_big", 0
    agent = _new_agent(model, SYSTEM_PROMPT_AUDIT, max_iterations=1, max_tokens=max_tokens)
    meter = MeterRail(task, budget, model, stop_below, task_cap, traj_path)
    await agent.register_rail(meter)
    res = await agent.invoke({"query": prompt, "conversation_id": f"{task.task_id}-audit-{uuid.uuid4().hex[:6]}"})
    if not isinstance(res, dict) or res.get("result_type") != "answer":
        return "error", meter.calls
    blocks = extract_blocks(res.get("output"))
    new = apply_patch(sources, blocks, layout.rel(layout.sources[0])) if blocks else None
    changed = {rel: text for rel, text in (new or {}).items() if text != sources[rel]}
    if not changed:
        meter._traj(kind="harness", note="audit: no change")
        return "confirmed", meter.calls
    back = reverts_to_original(task.task_id, sources, changed)
    if back:
        meter._traj(kind="harness", note=f"audit: change reverts {back} to the original task code, rejected")
        return "kept", meter.calls
    accepted = False
    try:
        layout.write_sources(changed)
        new_run = await run_tests(layout)
        if not new_run.passed or (new_run.doc_fail and not run.doc_fail):
            meter._traj(kind="harness", note=f"audit: change of {sorted(changed)} breaks tests, undone")
            return "kept", meter.calls
        if snapshot_dir is not None:
            for rel in changed:
                p = snapshot_dir / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(sources[rel], encoding="utf-8")
        meter._traj(kind="harness", note=f"audit: kept change of {sorted(changed)} ({', '.join(defined_names(blocks))})")
        accepted = True
        return "changed", meter.calls
    finally:
        # Отмена по дедлайну тоже должна вернуть уже решённую задачу в рабочее состояние.
        if not accepted:
            layout.write_sources({rel: sources[rel] for rel in changed})


NOTE_UNCHANGED = ("Your previous reply repeated the current code, yet these tests still fail. The cause is elsewhere: "
                  "check other functions and methods, module and class constants, __init__/__eq__, and whether "
                  "inputs get mutated.")
NOTE_WORSE = "Your previous change of {names} made more tests fail and was undone. Find a different fix."
NOTE_UNUSABLE = ("Your previous reply could not be applied: it was not valid Python. Write each changed method inside "
                 "its class (class Name: then the method), functions at top level, no stray indentation.")
NOTE_NO_PROGRESS = "Your previous change of {names} did not reduce the failures. Look for another cause."


async def _apply_greedy(layout, sources: dict[str, str], blocks: list[str], primary: str, base):
    """Определения из ответа по одному: оставляем те, после которых тестов падает не больше. Результат
    принимаем, только если в итоге стало лучше, чем до раунда. Файлы на выходе — в принятом состоянии
    или исходные (None)."""
    units = split_patch(blocks)
    if len(units) < 2 or not all(parses(t) for t in sources.values()):
        return None
    cur, cur_run = dict(sources), base
    for unit in units:
        cand = apply_patch(cur, unit, primary)
        if cand is None:
            continue
        diff = {rel: text for rel, text in cand.items() if text != cur[rel]}
        layout.write_sources(diff)
        r = await run_tests(layout)
        if r.key() <= cur_run.key() and not (cur_run.passed and not r.passed):
            cur, cur_run = cand, r
        else:
            layout.write_sources({rel: cur[rel] for rel in diff})
    if cur_run.key() < base.key():
        return cur, cur_run
    layout.write_sources({rel: sources[rel] for rel in cur if cur[rel] != sources[rel]})
    return None


SEARCH_MAX_TOKENS = 120
SEARCH_REVERSE_MAX_TOKENS = 400
# Потолок ответа с кодом. 300 обрезали ~1 % ответов (разбор лога с часовыми поясами на hold-c7: оба раза ровно
# 300, задача потеряна); медиана 44–68, p99 167–300. Выше потолок ничего не стоит, пока не нужен.
COMPUTE_MAX_TOKENS = 300
COMPUTE_MAX_TOKENS_BY_TYPE = {"math": 500, "data_analysis": 600, "search": 600}
COMPUTE_TRIES = 2
MATH_MOTION = os.environ.get("MATH_MOTION", "1") == "1"
MOTION_MAX_TOKENS = 250
# search по таблицам и конфигам базы — кодом (budget_agent/datasearch.py). SEARCH_DATA=0 выключает.
SEARCH_DATA = os.environ.get("SEARCH_DATA", "1") == "1"
_EMPTY_SELECTION = re.compile(r"empty sequence|ZeroDivisionError|division by zero|printed nothing|"
                              r"index out of range", re.I)


def _search_data_files(task: Task) -> list:
    r = datasearch.route(direct_search.question_text(task.instruction), task.workdir)
    return r.files if r else datasearch.data_files(task.workdir)[:datasearch.MAX_DESCRIBED]


def _answer_snapshot(task: Task) -> dict[Path, tuple[bytes, int] | None]:
    """answer.txt до запуска кода модели: код может записать его сам, а во втором проходе там уже лежит ответ
    первого прохода, который перепроверка не должна подменить в обход голосования."""
    snap = {}
    for p in dict.fromkeys([task.workdir / "answer.txt", task.dir / "answer.txt"]):
        try:
            snap[p] = (p.read_bytes(), p.stat().st_mtime_ns) if p.exists() else None
        except OSError:
            snap[p] = None
    return snap


def _answer_written_and_restore(snap: dict[Path, tuple[bytes, int] | None]) -> str | None:
    """Что код модели записал в answer.txt (None — не писал); файлы возвращаются в прежнее состояние."""
    written = None
    for p, old in snap.items():
        try:
            new = (p.read_bytes(), p.stat().st_mtime_ns) if p.exists() else None
        except OSError:
            continue
        if new == old:
            continue
        if new is not None and written is None:
            written = new[0].decode("utf-8", "replace")
        try:
            if old is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(old[0])
        except OSError:
            pass
    return written


async def solve_compute_direct(task: Task, budget: Budget, model: str, stop_below: float, task_cap: float,
                               traj_path: Path | None, spent_before: float, max_tokens: int | None = None,
                               system_extra: str = "", data_files: list | None = None) -> Attempt | None:
    """math и data_analysis в один вызов (budget_agent/compute.py). None — код не заработал за две попытки:
    тогда задачу берёт обычный агент. max_tokens и system_extra — для второго мнения (second.py).
    search: вопрос о таблице или конфиге базы (budget_agent/datasearch.py), data_files — какие файлы показать."""
    max_tokens = max_tokens or COMPUTE_MAX_TOKENS_BY_TYPE.get(task.task_type, COMPUTE_MAX_TOKENS)
    question = direct_compute.question_text(task.instruction)
    system = direct_compute.system_prompt(task.task_type) + system_extra
    extract_motion = (MATH_MOTION and task.task_type == "math" and not system_extra
                      and model == "qwen3.8-27b-noreason" and motion.applies(question))
    if task.task_type == "data_analysis":
        pre = prefetch_files(task)
        question = f"{question}\n{direct_compute.DATA_HINT}" + (f"\n{pre}" if pre else "")
    elif task.task_type == "search":
        files = data_files if data_files is not None else _search_data_files(task)
        ask = direct_search.question_text(task.instruction)
        question = ask + "\n" + datasearch.prefetch(task.workdir, files, ask)
        system = direct_compute.SYSTEM_PROMPT_DATA + datasearch.SEARCH_DATA_RULE + system_extra
    ask = direct_search.question_text(task.instruction) if task.task_type == "search" else question
    feedback = ""
    calls = retries = 0
    flags: list[str] = []
    for attempt in range(COMPUTE_TRIES):
        if attempt:
            flags.append("retry")
        agent = _new_agent(model, motion.SYSTEM_PROMPT if extract_motion else system, max_iterations=1,
                           max_tokens=MOTION_MAX_TOKENS if extract_motion else max_tokens)
        meter = MeterRail(task, budget, model, stop_below, task_cap, traj_path)
        await agent.register_rail(meter)
        if task.task_type == "search" and not attempt:
            meter._traj(kind="harness", note=f"search via code: {[d.rel for d in files]}")
        res = await agent.invoke({"query": question + feedback,
                                  "conversation_id": f"{task.task_id}-compute{attempt}-{uuid.uuid4().hex[:6]}"})
        calls, retries = calls + meter.calls, retries + meter.retries
        cost = budget.by_task.get(task.task_id, 0.0) - spent_before
        if not isinstance(res, dict) or res.get("result_type") != "answer":
            return Attempt(answer=None, result_type=str(res.get("result_type", "?")) if isinstance(res, dict) else "?",
                           cost=cost, calls=calls, retries=retries)
        reply = res.get("output") or ""
        if extract_motion:
            answer = motion.solve(reply, question) if getattr(meter, "last_completion", 0) < MOTION_MAX_TOKENS else None
            if answer is not None:
                meter._traj(kind="harness", note=f"motion: extracted initial conditions, exact meeting={answer}")
                return Attempt(answer=answer, result_type="answer", cost=cost, calls=calls, retries=retries)
            meter._traj(kind="harness", note="motion: invalid data or no meeting; fallback to Python")
            extract_motion = False
            feedback = "\nSolve the original question with Python code."
            continue
        if normalize_answer(reply) == NO_ANSWER_MARK:
            meter._traj(kind="harness", note="compute: model says unanswerable")
            return Attempt(answer=None, result_type="unanswerable", cost=cost, calls=calls, retries=retries)
        code = direct_compute.strip_code(reply)
        if getattr(meter, "last_completion", 0) >= max_tokens:
            # Ответ обрезан по потолку: недописанный код мог и запуститься, и напечатать неверное (hold-c4).
            feedback = "\nYour previous code was cut off at the length limit. Write a much shorter program."
            meter._traj(kind="harness", note="compute: reply cut at max_tokens")
            continue
        program = direct_compute.to_program(code)
        if program is None:
            feedback = "\nYour previous reply was not valid Python. Reply with Python code only."
            meter._traj(kind="harness", note="compute: reply is not python")
            continue
        if task.task_type == "math" and attempt + 1 < COMPUTE_TRIES and direct_compute.is_literal(code):
            # Голое число вместо вычисления: посчитано в уме, на hold-c5 неверно (112 вместо 56).
            feedback = (f"\nYour previous reply only stated a number ({code.strip()[:40]}). Compute it in code from the "
                        "numbers in the question; if the unknown is defined by conditions, loop over candidates.")
            meter._traj(kind="harness", note="compute: reply is a bare number, retry")
            continue
        snap = _answer_snapshot(task)
        try:
            rc, out, err = await run_python(program, task.workdir)
        finally:
            # Код второго мнения может записать answer.txt ещё до отмены/таймаута.
            filed = _answer_written_and_restore(snap)
        multi = wants_lines(task.instruction)  # «one per line»: ответ — весь вывод, а не последняя строка
        line = direct_compute.last_output_line(out)
        src = out if multi else line
        if filed is not None and direct_compute.last_output_line(filed):
            # Программа сама записала ответ в answer.txt: это и есть её ответ (вывод мог быть чем угодно).
            line = direct_compute.last_output_line(filed)
            src = filed if multi else line
            meter._traj(kind="harness", note=f"compute: program wrote answer.txt {line!r}, file restored")
        answer = normalize_answer(src, multiline=multi) if rc == 0 and line else None
        meter._traj(kind="harness", note=f"compute run rc={rc} answer={answer!r} err={err.strip()[-200:]!r}")
        if task.task_type == "search" and answer is not None:
            # Пустая выборка или UNANSWERABLE из программы: в таблицах ответа нет, дальше — search по карточкам.
            if answer == NO_ANSWER_MARK or datasearch.EMPTY_RESULT.fullmatch(answer):
                meter._traj(kind="harness", note=f"compute: empty result {answer!r}, unanswerable")
                return Attempt(answer=None, result_type="unanswerable", cost=cost, calls=calls, retries=retries)
            if attempt + 1 < COMPUTE_TRIES and datasearch.not_a_name(ask, answer):
                feedback = (f"\nYour previous program printed {answer!r}, but the question asks who: follow the link "
                            "to the record or file with the person and print the name.")
                meter._traj(kind="harness", note="compute: 'who' answered with a non-name, retry")
                continue
        if answer is not None and answer != NO_ANSWER_MARK:
            flags += [f for f in direct_compute.doubt_flags(ask, out, answer) if not (multi and f == "multi")]
            if task.task_type == "search":
                flags.append("code")
            return Attempt(answer=answer, result_type="answer", cost=cost, calls=calls, retries=retries,
                           flags=tuple(flags))
        missing = direct_compute.missing_data_reason(err, task.workdir, question) if rc != 0 else None
        if missing:
            meter._traj(kind="harness", note=f"compute: {missing}, unanswerable")
            return Attempt(answer=None, result_type="unanswerable", cost=cost, calls=calls, retries=retries)
        last_err = (err.strip().splitlines() or ["the code printed nothing"])[-1][:200]
        feedback = f"\nYour previous code:\n{code}\nfailed: {last_err}"
        if task.task_type in ("search", "data_analysis") and _EMPTY_SELECTION.search(last_err):
            # Фильтр не нашёл ни одной строки («m[5] in '123456'» для «2026-01»), и на повторе модель лишь обернула
            # max() в «if d:» и не напечатала ничего (стресс-набор search).
            feedback += (". Your filter probably matched no rows: compare it with the sample rows shown above "
                         "(exact field names and value formats).")
    return None


async def solve_search_direct(task: Task, budget: Budget, model: str, stop_below: float, task_cap: float,
                              traj_path: Path | None, spent_before: float, max_tokens: int | None = None,
                              system_extra: str = "") -> Attempt | None:
    """search в один вызов (budget_agent/search.py). None — карточки не собрать или ответ не разобрать:
    тогда задачу берёт обычный агент с grep. max_tokens и system_extra — для второго мнения (second.py)."""
    question = direct_search.question_text(task.instruction)
    context = direct_search.build_context(question, task.workdir)
    if context is None:
        return None
    reverse = direct_search.is_reverse(question)
    system = direct_search.SYSTEM_PROMPT_SEARCH + (direct_search.REVERSE_RULE if reverse else "") + system_extra
    limit = max_tokens or (SEARCH_REVERSE_MAX_TOKENS if reverse else SEARCH_MAX_TOKENS)
    query = direct_search.build_prompt(question, context)
    calls = retries = 0
    for attempt in range(2):
        agent = _new_agent(model, system, max_iterations=1, max_tokens=limit)
        meter = MeterRail(task, budget, model, stop_below, task_cap, traj_path)
        await agent.register_rail(meter)
        res = await agent.invoke({"query": query, "conversation_id": f"{task.task_id}-direct-{uuid.uuid4().hex[:6]}"})
        calls, retries = calls + meter.calls, retries + meter.retries
        cut = getattr(meter, "last_completion", 0) >= limit
        if not cut or attempt:
            break
        # Ответ обрезан по потолку: последней строкой оказался бы кусок рассуждения. Один повтор, короче. При
        # подсчёте по всей базе модель перечисляла всех подряд с «No» и не успевала и в 800 (регресс 12.11).
        meter._traj(kind="harness", note="direct search: reply cut at max_tokens, retry")
        limit = max(limit * 2, 1200) if reverse else limit * 2
        query += ("\n\nYour previous answer was cut off. Be brief: do not list candidates that do not match; one "
                  "short line with the matching names only, then the answer.")
    cost = budget.by_task.get(task.task_id, 0.0) - spent_before
    if cut:
        # Обрезан и после повтора: последняя строка — обрывок рассуждения («Look up Mira Ivan» записалось ответом
        # на «сколько», регресс 12.11). Такой ответ не пишем.
        meter._traj(kind="harness", note="direct search: reply cut again, no answer")
        return Attempt(answer=None, result_type="cut", cost=cost, calls=calls, retries=retries)
    if not isinstance(res, dict) or res.get("result_type") != "answer":
        return Attempt(answer=None, result_type=str(res.get("result_type", "?")) if isinstance(res, dict) else "?",
                       cost=cost, calls=calls, retries=retries)
    answer = normalize_answer(direct_search.count_from_names(question, direct_search.final_line(res.get("output"))))
    meter._traj(kind="harness", note=f"direct search answer {answer!r}")
    if answer == NO_ANSWER_MARK:
        return Attempt(answer=None, result_type="unanswerable", cost=cost, calls=calls, retries=retries,
                       flags=tuple(direct_search.doubt_flags(question, context, None)))
    if answer is None or looks_like_command(answer) or len(answer) > 80:
        return None  # не разобрать: пусть пробует агент
    return Attempt(answer=answer, result_type="answer", cost=cost, calls=calls, retries=retries,
                   flags=tuple(direct_search.doubt_flags(question, context, answer)))


async def solve(task: Task, budget: Budget, model: str, *, max_iterations: int = 6,
                max_tokens: int = 768, stop_below: float = 200.0, task_cap: float = 2000.0,
                traj_dir: Path | None = None, escalate: bool = False, force_agent: bool = False) -> Attempt:
    """force_agent: сразу обычный агент с sh, без решения в один вызов (второй проход search, second.py)."""
    workdir = task.workdir
    stale = workdir / "answer.txt"
    if stale.exists():
        stale.unlink()
    traj_path = None
    if traj_dir is not None:
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / f"{task.task_id}.jsonl"
    spent_before = budget.by_task.get(task.task_id, 0.0)

    if task.task_type in ("math", "data_analysis") and os.environ.get("COMPUTE_DIRECT", "1") == "1" \
            and not force_agent:
        att = await solve_compute_direct(task, budget, model, stop_below, task_cap, traj_path, spent_before)
        if att is not None:
            return att

    if task.task_type == "search" and os.environ.get("SEARCH_DIRECT", "1") == "1" and not force_agent:
        # Вопрос о таблице или конфиге базы — сначала код; не вышло — обычный search по карточкам. Если карточек
        # по вопросу нет вовсе, код пробуем и при простом совпадении полей, до агента с grep (datasearch.py).
        route = datasearch.route(direct_search.question_text(task.instruction), workdir) if SEARCH_DATA else None
        if route is not None and route.kind != "fields":
            att = await solve_compute_direct(task, budget, model, stop_below, task_cap, traj_path, spent_before,
                                             data_files=route.files)
            if att is not None and att.result_type == "answer":
                return att
        att = await solve_search_direct(task, budget, model, stop_below, task_cap, traj_path, spent_before)
        if att is not None:
            return att
        if route is not None and route.kind == "fields":
            att = await solve_compute_direct(task, budget, model, stop_below, task_cap, traj_path, spent_before,
                                             data_files=route.files)
            if att is not None and att.result_type == "answer":
                return att

    fast_calls = fast_retries = 0
    if task.task_type == "coding" and escalate:
        status, fast_calls, fast_retries, _ = await solve_coding_fast(
            task, budget, REASON_MODEL, stop_below, task_cap, traj_path, rounds=REASON_ROUNDS,
            max_tokens=REASON_MAX_TOKENS, allow_impossible=True)
        return Attempt(answer=done_word(task.instruction) if status == "pass" else None,
                       result_type="answer" if status == "pass" else f"escalated_{status}",
                       cost=budget.by_task.get(task.task_id, 0.0) - spent_before,
                       calls=fast_calls, retries=fast_retries)
    if task.task_type == "coding" and os.environ.get("CODING_FAST", "1") == "1":
        status, fast_calls, fast_retries, improved = await solve_coding_fast(task, budget, model, stop_below,
                                                                             task_cap, traj_path)
        if status in ("pass", "stuck"):
            return Attempt(answer=done_word(task.instruction) if status == "pass" else None,
                           result_type="answer" if status == "pass" else "coding_stuck",
                           cost=budget.by_task.get(task.task_id, 0.0) - spent_before,
                           calls=fast_calls, retries=fast_retries, flags=("progress",) if improved else ())
        # "n/a": дальше обычный агент, он видит уже частично исправленный модуль

    last_cmd: list[str] = []

    @tool(name="sh", description="Run a bash command in the task directory; returns its output.",
          input_params=SH_SCHEMA)
    async def sh(cmd: str) -> str:
        # Защита от зацикливания: при temperature 0 модель бывает гоняет одни и те же команды до лимита шагов.
        if cmd.strip() in last_cmd:
            return ("Same command as before, its output is above. Stop searching: reply with the final answer "
                    "only, or UNANSWERABLE if the data does not contain it.")
        last_cmd.append(cmd.strip())
        return await run_bash(cmd, workdir)

    # coding в текстовом режиме стабильно отвечает нативным <tool_call>, который прокси вырезает:
    # для него сразу обычный режим со схемой инструмента (docs/experiments.md).
    text_mode = os.environ.get("TOOL_MODE", "text") == "text" and task.task_type != "coding"
    # Выход в 10 раз дороже входа: в текстовом режиме команда короткая, а длинный ответ — это рассуждения
    # в комментариях (на невыполнимой math было 430 токенов). Потолок их обрезает.
    call_max_tokens = min(max_tokens, TEXT_MAX_TOKENS) if text_mode else max_tokens
    iterations = TYPE_MAX_ITERATIONS.get(task.task_type, max_iterations)
    agent = _new_agent(model, SYSTEM_PROMPT_TEXT if text_mode else SYSTEM_PROMPT, iterations, call_max_tokens)
    agent.ability_manager.add_ability(sh.card, sh)
    meter = MeterRail(task, budget, model, stop_below, task_cap, traj_path)
    await agent.register_rail(meter)
    if text_mode:
        await agent.register_rail(TextToolRail("sh"))

    res = await agent.invoke({"query": build_query(task, text_mode),
                              "conversation_id": f"{task.task_id}-{uuid.uuid4().hex[:8]}"})
    result_type = str(res.get("result_type", "?")) if isinstance(res, dict) else "?"

    # Если модель записала answer.txt сама, верим файлу, иначе берём финальную реплику.
    answer = None
    multi = wants_lines(task.instruction)
    if stale.exists():
        answer = normalize_answer(stale.read_text(encoding="utf-8", errors="replace"), multiline=multi)
    if answer is None and result_type == "answer":
        answer = normalize_answer(res.get("output"), multiline=multi)
    if answer == NO_ANSWER_MARK:
        answer, result_type = None, "unanswerable"
    if answer is not None and looks_like_command(answer):
        answer, result_type = None, "command_as_answer"  # команда или план вместо ответа — не записываем
    if answer is None and stale.exists():
        stale.unlink()  # в answer.txt остался мусор (например, сырой вывод grep): лучше без ответа
    if answer is not None and task.task_type == "coding" and not await tests_pass(task):
        answer, result_type = None, "tests_fail"

    return Attempt(answer=answer, result_type=result_type,
                   cost=budget.by_task.get(task.task_id, 0.0) - spent_before, calls=meter.calls + fast_calls,
                   retries=meter.retries + fast_retries)
