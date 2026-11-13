"""Второй проход: перепроверка ответов с сигналом сомнения, только из излишка бюджета (docs/ideas.md).

Первый проход решает все задачи дёшево. Здесь берём задачи, где ответ под сомнением (сигналы считаются без
токенов, `Attempt.flags`), и тратим на них часть оставшегося бюджета, самые выгодные первыми.

Правила:
- ответ меняем, только если новое свидетельство сильнее старого: тесты прошли (coding), два независимых
  ответа совпали против исходного (search, math, data), агент нашёл ответ там, где один вызов его не нашёл;
- записанный ответ никогда не удаляем: штрафа за ответ на невыполнимую регламент не называет;
- перед каждой задачей проверяем излишек (остаток минус резерв) и дедлайн.

Почему так (все прогоны до 10.11, без hold-c2): у search сигналы «ответа нет в карточках» и «тип ответа не тот»
срабатывали только на неверных ответах (17 из 17), обратный поиск и сравнения ошибались в 43 % случаев, а
каждый третий отказ «невозможно ответить» при найденных карточках был выполнимой задачей. math и data
перепроверяются при сигналах сомнения или отказе. Перепроверку готового ответа без сигнала можно включить
через COMPUTE_PLAIN_VERIFY=1; сравнение режимов — docs/final-sprint.md.

Задачи без ответа (UNANSWERED_RETRY, ниже): search, math и data с обрывом по длине или потолком на задачу
получают одну попытку моделью с рассуждениями; задача любого типа после исключения или таймаута первого
прохода решается заново обычным путём. Резерв для них не держится.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass

from .budget import Budget
from .solver import (Attempt, REASON_MODEL, solve, solve_coding_audit, solve_compute_direct,
                     solve_search_direct)
from .answers import done_word
from .tasks import Task

# Второе мнение — другой моделью: та же модель при temperature 0 повторяет ту же ошибку. Спор решает
# модель с рассуждениями.
ALT_MODEL = os.environ.get("ALT_MODEL", "qwen3.6-35b-a3b-noreason")
TIE_MODEL = os.environ.get("TIE_MODEL", REASON_MODEL)
TIE_MAX_TOKENS = 2500
ALT_SEARCH_MAX_TOKENS = 400
ALT_COMPUTE_MAX_TOKENS = 600
# Для перепроверки search модель выписывает кандидатов со значением, по которому решает: ошибки сравнения
# («родился в 1975» засчитан как «до 1975», hold-c4) видны ей самой.
EXPLICIT_RULE = (" Before the answer, list every candidate you checked with the value that decides it, e.g. "
                 "'Ana Lee (born 1975), Bo Kim (born 1981)', on one line.")
COMPUTE_RULE = " Read the question again: print exactly the quantity it asks for."

# Доля неверных ответов среди тех, где сработал сигнал (оценка по старым прогонам, см. выше).
PRIOR = {"type": 0.8, "not_in_context": 0.8, "reverse": 0.4, "unanswerable": 0.3}
P_CODING = 0.3
# coding, прошедший видимые тесты: на регрессе 10.11 скрытые тесты упали у 3 из ~30 таких (поведение из docstring).
P_AUDIT = 0.1
P_COMPUTE_FLAG, P_COMPUTE_PLAIN, P_COMPUTE_NONE = 0.1, 0.04, 0.1
# Задачи search, math и data, отсечённые проверкой «невыполнимо» без LLM (precheck.py): по всем прогонам до 11.11
# (без hold-c2) 2 из 13 таких отсечений были выполнимыми задачами (hold-c5 search_0032, hold-c6 data_0039) — каждое
# правило ломалось на новой формулировке, а финальные формулировки будут другими (ответ организаторов 10.11).
# coding не перепроверяем: там 1 ошибка из 17 и перепроверка дорогая.
P_PRECHECK = 0.15
PRECHECK_TYPES = ("search", "math", "data_analysis")
# Задача search, math или data осталась после первого прохода без ответа, но не как «невыполнимо»: ответ дважды
# оборван по потолку длины или упёрся потолок на задачу. Раньше такая задача во второй проход не попадала: на регрессе 13.11 c7-search_0005 (награда 4) осталась пустой при 78 тысячах неистраченного
# бюджета — модель считала верно, но не уложилась ни в 400, ни в 1 200 токенов. Штрафа за неверный ответ регламент
# не называет, поэтому любой ответ здесь не хуже пустого: одна попытка моделью с рассуждениями и большим
# потолком ответа. Доля удач неизвестна (один случай); выключается UNANSWERED_RETRY=0.
P_UNANSWERED = 0.3
UNANSWERED_TYPES = ("search", "math", "data_analysis")
# «error» — агент с sh исчерпал шаги (ReActAgent): на невыполнимых search так кончается почти всегда, и повтор тем
# же путём 13.11 дал 0 из 8 (jul-os, KozlovKV, стресс-набор) за ~250 токенов каждый.
UNANSWERED_SKIP = ("unanswerable", "precheck", "escalated_impossible", "error")
UNANSWERED_RETRY = os.environ.get("UNANSWERED_RETRY", "1") == "1"
UNANSWERED_MAX_TOKENS = 2500
# Исключение или таймаут первого прохода — сбой, а не трудная задача: модель просто не ответила. Такую задачу
# любого типа решаем заново обычным путём и основной моделью (сотни токенов, а не до 2 800 у модели с
# рассуждениями). Исключение run_agent помечает «exception», а не «error», как исчерпанные шаги агента.
RERUN_RESULTS = ("exception", "timeout")
P_RERUN = 0.5
# Для задач без ответа резерв не держим (run_second_pass): он бережёт бюджет от необязательных перепроверок, а
# здесь без попытки балла точно нет. При малом переданном бюджете (отборочный прогон на двух задачах) резерв
# в 10 000 иначе запрещал бы любую вторую попытку.
NO_RESERVE_KINDS = ("unanswered", "rerun")
# Оценка стоимости действия (для очереди и проверки излишка) и потолок сверх уже потраченного на задачу.
EST = {"search_verify": 250, "search_agent": 450, "coding": 2500, "coding_audit": 300, "compute": 250,
       "precheck_verify": 120, "unanswered": 900, "rerun": 300}
# unanswered: потолок равен потолку ответа, поэтому вызов, упёршийся в 2 500 токенов, не повторяется с 5 000
# (так делает solve_search_direct после обрыва): живая попытка 13.11 уложилась в 1 153–1 406 токенов ответа.
# У math и data решатель после ошибки кода делает второй вызов, тогда расход до ~3 000.
# rerun: как наибольший потолок первого прохода (planner.UNKNOWN_CAP).
CAP = {"search_verify": 3500, "search_agent": 1500, "coding": 4000, "coding_audit": 2000, "compute": 3500,
       "precheck_verify": 600, "unanswered": UNANSWERED_MAX_TOKENS, "rerun": 1800}
PRECHECK_VERIFY = os.environ.get("PRECHECK_VERIFY", "1") == "1"
CODING_AUDIT = os.environ.get("CODING_AUDIT", "1") == "1"
# 12.11: на 44 задачах регресса и hold-a1 эти перепроверки дали +0 за 5 549 токенов.
# Подозрительные ответы и отказы остаются в очереди. Прежний режим — COMPUTE_PLAIN_VERIFY=1.
COMPUTE_PLAIN_VERIFY = os.environ.get("COMPUTE_PLAIN_VERIFY", "0") == "1"
# Порог награды для перепроверки math и data без сигнала — середина диапазона наград набора: на наших наборах
# (награды 1–5) это 3, как было; диапазон финальных наград организаторы не назвали («больше нуля», 10.11), и
# число 3 на шкале 0–1 выключило бы перепроверку, а на 10–100 включило бы её везде. COMPUTE_MIN_REWARD задаёт
# порог числом.
COMPUTE_REWARD_FRACTION = 0.5
_MIN_REWARD_ENV = os.environ.get("COMPUTE_MIN_REWARD")


def compute_reward_floor(rewards: list[float]) -> float:
    if _MIN_REWARD_ENV:
        return float(_MIN_REWARD_ENV)
    if not rewards:
        return 0.0
    lo, hi = min(rewards), max(rewards)
    return lo + COMPUTE_REWARD_FRACTION * (hi - lo)
STOP_BELOW = 200.0


@dataclass
class Candidate:
    task: Task
    att: Attempt
    kind: str
    p: float

    @property
    def est(self) -> float:
        return EST[self.kind]

    @property
    def priority(self) -> float:
        return self.task.reward * self.p / self.est


# Агент с grep на отказах search: на hold-c4 и hold-c5 6 из 6 отказов оказались правда невыполнимыми, ~2 000
# токенов без баллов (bench/RESULTS.md, v9). Выключено, включается SEARCH_AGENT_RECHECK=1.
SEARCH_AGENT_RECHECK = os.environ.get("SEARCH_AGENT_RECHECK", "0") == "1"


def candidates(first: dict[str, tuple[Task, Attempt]], compute: bool = True,
               prechecked: list[Task] | tuple[Task, ...] = ()) -> list[Candidate]:
    out = []
    floor = compute_reward_floor([t.reward for t, _ in first.values()] + [t.reward for t in prechecked])
    if PRECHECK_VERIFY:
        for task in prechecked:
            if task.task_type in PRECHECK_TYPES:
                out.append(Candidate(task, Attempt(answer=None, result_type="precheck", cost=0.0, calls=0),
                                     "precheck_verify", P_PRECHECK))
    for task, att in first.values():
        f = set(att.flags)
        typ = task.task_type
        if UNANSWERED_RETRY and att.answer is None and att.result_type in RERUN_RESULTS:
            out.append(Candidate(task, att, "rerun", P_RERUN))
        elif UNANSWERED_RETRY and att.answer is None and typ in UNANSWERED_TYPES \
                and att.result_type not in UNANSWERED_SKIP:
            out.append(Candidate(task, att, "unanswered", P_UNANSWERED))
        elif typ == "search" and "code" in f:
            # Ответ по таблице кодом (datasearch.py): перепроверка — тоже кодом, другой моделью, и только при
            # сигнале сомнения (несколько строк вывода, на «сколько» не число, повтор после ошибки).
            if compute and att.result_type == "answer" and f - {"code"}:
                out.append(Candidate(task, att, "compute", P_COMPUTE_FLAG))
        elif typ == "search":
            if att.result_type == "unanswerable" and "unanswerable" in f:
                if SEARCH_AGENT_RECHECK:
                    out.append(Candidate(task, att, "search_agent", PRIOR["unanswerable"]))
            elif att.answer is not None and f & {"type", "not_in_context", "reverse"}:
                out.append(Candidate(task, att, "search_verify",
                                     max(PRIOR[x] for x in f & {"type", "not_in_context", "reverse"})))
        elif typ == "coding":
            if att.result_type == "coding_stuck" and "progress" in f:
                out.append(Candidate(task, att, "coding", P_CODING))
            elif att.answer is not None and att.answer == done_word(task.instruction) and CODING_AUDIT:
                out.append(Candidate(task, att, "coding_audit", P_AUDIT))
        elif typ in ("math", "data_analysis") and compute:
            if att.result_type == "answer":
                # Без сигнала — только дорогие задачи: на честных hold-c5 и hold-c6 44 перепроверки дали одно
                # исправление (награда 3); порог убирает 16 из 44 и ~3 000 токенов, исправление остаётся.
                if f or (COMPUTE_PLAIN_VERIFY and task.reward >= floor):
                    out.append(Candidate(task, att, "compute", P_COMPUTE_FLAG if f else P_COMPUTE_PLAIN))
            elif att.result_type == "unanswerable":
                out.append(Candidate(task, att, "compute", P_COMPUTE_NONE))
    return sorted(out, key=lambda c: -c.priority)


def same(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    x, y = a.strip().lower(), b.strip().lower()
    if x == y:
        return True
    try:
        return float(x.replace(",", "")) == float(y.replace(",", ""))
    except ValueError:
        return False


def _degenerate(a: str | None) -> bool:
    return a is None or not a.strip() or same(a, "0")


async def _vote(first: str | None, alt_fn, tie_fn) -> tuple[str | None, str, int]:
    """Второе мнение, при расхождении — третье. Возвращает (ответ, исход, вызовов).

    Ненулевой ответ на 0 не меняем: ноль — типичный результат сломанного кода. На регрессе 10.11 второе мнение и
    модель с рассуждениями независимо написали `... if (seen:=set()) else 0` (пустое множество — ложь) и вдвоём
    заменили верные 578 на 0. Все исправления второго прохода по логам были на ненулевые значения."""
    alt = await alt_fn()
    calls = alt.calls if alt else 0
    a2 = alt.answer if alt else None
    if a2 is None or (_degenerate(a2) and not _degenerate(first)):
        return first, "kept", calls  # второе мнение ответа не дало: исходный не трогаем
    if same(a2, first):
        return first, "confirmed", calls
    tie = await tie_fn()
    calls += tie.calls if tie else 0
    a3 = tie.answer if tie else None
    if _degenerate(a3) and not _degenerate(first):
        return first, "kept", calls
    if a3 is not None and same(a3, a2):
        return a2, "changed", calls
    if a3 is not None and not same(a3, first):
        # Все три разные: верим модели с рассуждениями. По логам до 10.11 это ни разу не хуже исходного, а на
        # hold-c6 search_0026 (8 / 5 / 7, верно 7) — единственный способ попасть.
        return a3, "changed", calls
    return first, "kept", calls


async def _precheck_verify(task: Task, budget: Budget, model: str, cap: float, traj) -> tuple[str | None, str, int]:
    """Обычный первый проход для задачи, которую проверка без LLM сочла невыполнимой. Ответ берём, только если он
    опирается на данные задачи: модели на невыполнимых почти всегда что-то отвечают (docs/notes.md).
    - search: без карточек по вопросу вызова нет вовсе; ответ-слово должно быть в карточках, тип — подходить;
    - math и data: код запущен на файлах задачи, ответ не 0 и не пустой (фильтр по отсутствующему значению даёт 0),
      тип подходит."""
    before = budget.by_task.get(task.task_id, 0.0)
    if task.task_type == "search":
        att = await solve_search_direct(task, budget, model, STOP_BELOW, cap, traj, before)
        bad = {"not_in_context", "type"}
    else:
        att = await solve_compute_direct(task, budget, model, STOP_BELOW, cap, traj, before)
        bad = {"type"}
    calls = att.calls if att else 0
    if att is None or att.result_type != "answer":
        return None, (att.result_type if att else "none"), calls
    if _degenerate(att.answer):
        return None, "rejected_zero", calls
    if bad & set(att.flags):
        return None, "rejected_" + "_".join(sorted(bad & set(att.flags))), calls
    if task.task_type == "search" and _echoes_question(att.answer, task.instruction):
        # «Who is the mentor of Zara Quinn?» -> «mentor»: человека нет, модель повторила слово вопроса (chillcoders).
        return None, "rejected_echo", calls
    return att.answer, "changed", calls


def _echoes_question(answer: str | None, question: str) -> bool:
    a = (answer or "").strip()
    return bool(re.fullmatch(r"[a-z][a-z-]*", a)) and re.search(r"(?<![\w])" + re.escape(a) + r"(?![\w])",
                                                                  question) is not None


async def _retry_unanswered(task: Task, budget: Budget, cap: float, traj) -> tuple[str | None, str, int]:
    """Одна попытка для задачи без ответа: модель с рассуждениями, потолок ответа UNANSWERED_MAX_TOKENS. Ответ
    берём любой: писать было нечего, хуже пустого он не станет. «Невыполнимо» и отказ оставляют задачу пустой."""
    before = budget.by_task.get(task.task_id, 0.0)
    fn = solve_search_direct if task.task_type == "search" else solve_compute_direct
    att = await fn(task, budget, TIE_MODEL, STOP_BELOW, cap, traj, before, max_tokens=UNANSWERED_MAX_TOKENS)
    calls = att.calls if att else 0
    if att is None or att.result_type != "answer" or att.answer is None:
        return None, (att.result_type if att else "none"), calls
    return att.answer, "changed", calls


async def recheck(c: Candidate, budget: Budget, model: str, traj_dir) -> tuple[str | None, str, int]:
    task, att = c.task, c.att
    traj = None
    if traj_dir is not None:
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj = traj_dir / f"{task.task_id}.jsonl"
    before = budget.by_task.get(task.task_id, 0.0)
    cap = before + CAP[c.kind]

    if c.kind == "unanswered":
        return await _retry_unanswered(task, budget, cap, traj)

    if c.kind == "rerun":
        new = await solve(task, budget, model, task_cap=cap, traj_dir=traj_dir)
        return (new.answer, "changed" if new.answer is not None else new.result_type, new.calls)

    if c.kind == "coding":
        new = await solve(task, budget, model, task_cap=cap, traj_dir=traj_dir, escalate=True)
        return (new.answer, "changed" if new.answer else new.result_type, new.calls)

    if c.kind == "coding_audit":
        snap = traj_dir.parent / "pass1" / task.task_id if traj_dir is not None else None
        result, calls = await solve_coding_audit(task, budget, model, STOP_BELOW, cap, traj, snap)
        return att.answer, result, calls

    if c.kind == "precheck_verify":
        return await _precheck_verify(task, budget, model, cap, traj)

    if c.kind == "search_agent":
        new = await solve(task, budget, model, task_cap=cap, traj_dir=traj_dir, force_agent=True)
        return (new.answer, "changed" if new.answer else "kept", new.calls)

    if c.kind == "search_verify":
        def run(m, limit):
            return solve_search_direct(task, budget, m, STOP_BELOW, cap, traj, budget.by_task.get(task.task_id, 0.0),
                                       max_tokens=limit, system_extra=EXPLICIT_RULE)
        return await _vote(att.answer, lambda: run(ALT_MODEL, ALT_SEARCH_MAX_TOKENS),
                           lambda: run(TIE_MODEL, TIE_MAX_TOKENS))

    def run_c(m, limit):
        return solve_compute_direct(task, budget, m, STOP_BELOW, cap, traj, budget.by_task.get(task.task_id, 0.0),
                                    max_tokens=limit, system_extra=COMPUTE_RULE)
    return await _vote(att.answer, lambda: run_c(ALT_MODEL, ALT_COMPUTE_MAX_TOKENS),
                       lambda: run_c(TIE_MODEL, TIE_MAX_TOKENS))


async def run_second_pass(first: dict[str, tuple[Task, Attempt]], budget: Budget, model: str,
                          sem: asyncio.Semaphore, reserve: float, time_left, traj_dir, on_answer, record, log,
                          task_timeout: float, compute: bool = True, prechecked=()) -> None:
    cands = candidates(first, compute=compute, prechecked=prechecked)
    kinds = {}
    for c in cands:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    log(f"second pass: {len(cands)} candidates {kinds}, budget left {budget.remaining:.0f}, reserve {reserve:.0f}")

    async def one(c: Candidate) -> None:
        async with sem:
            tid = c.task.task_id
            if time_left() <= 0:
                return
            keep = 0.0 if c.kind in NO_RESERVE_KINDS else reserve
            if budget.remaining - keep < c.est:
                log(f"{tid}: second pass skipped, surplus {budget.remaining - keep:.0f} < est {c.est:.0f}")
                return
            before = budget.by_task.get(tid, 0.0)
            t0 = time.monotonic()
            try:
                answer, result, calls = await asyncio.wait_for(recheck(c, budget, model, traj_dir),
                                                               min(task_timeout * 2, max(time_left(), 1.0)))
            except Exception as e:  # перепроверка не должна ронять прогон: исходный ответ остаётся
                log(f"{tid}: second pass error {type(e).__name__}: {e}")
                return
            changed = result == "changed" and answer is not None and not same(answer, c.att.answer)
            if changed:
                on_answer(c.task, answer)
            cost = budget.by_task.get(tid, 0.0) - before
            record({"task": tid, "type": c.task.task_type, "reward": c.task.reward,
                    "answer": answer if changed else c.att.answer, "prev": c.att.answer,
                    "result": f"p2_{c.kind}_{'changed' if changed else result}", "cost": round(cost, 1),
                    "calls": calls, "sec": round(time.monotonic() - t0, 1), "pass": 2})
            log(f"{tid}: second pass {c.kind}: {result} {c.att.answer!r} -> {answer!r} cost={cost:.0f} "
                f"spent={budget.spent:.0f}")

    # return_exceptions: сбой записи ответа или журнала у одной задачи не обрывает перепроверку остальных.
    for res in await asyncio.gather(*(one(c) for c in cands), return_exceptions=True):
        if isinstance(res, Exception):
            log(f"second pass: task error {type(res).__name__}: {res}")
