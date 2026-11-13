"""Точка входа харнесса: python run_agent.py --tasks-dir DIR

В контейнере: python /app/run_agent.py --tasks-dir /data/tasks
Окружение: LLM_BASE_URL, LLM_API_KEY (обязательно), OUT (необязательно: файл или каталог для сводки решений).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Для локальной разработки: подхватить .env, если переменные не заданы. В контейнере .env нет."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env_budget(default: float = 100_000.0) -> float:
    """Бюджет из окружения: TOKEN_BUDGET (наше имя) или BUDGET (так бюджет передают при проверке).
    Кривое значение не должно ронять прогон: тогда 100k."""
    for name in ("TOKEN_BUDGET", "BUDGET"):
        raw = os.environ.get(name, "").strip().replace("_", "").replace(" ", "")
        if not raw:
            continue
        mult = 1000.0 if raw[-1] in "kK" else 1.0
        try:
            value = float(raw.rstrip("kK")) * mult
        except ValueError:
            continue
        if value > 0 and value != float("inf"):
            return value
    return default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks-dir", required=True, type=Path)
    p.add_argument("--out", default=os.environ.get("OUT"), help="файл или каталог для сводки решений")
    p.add_argument("--budget", type=float, default=env_budget())
    p.add_argument("--model", default=os.environ.get("MODEL", "qwen3.8-27b-noreason"))
    # Регламент разрешает 5 сессий, но у тарифа NeuralDeep потолок 3 одновременных запроса на аккаунт.
    p.add_argument("--parallel", type=int, default=int(os.environ.get("PARALLEL", 3)))
    p.add_argument("--task-timeout", type=float, default=240.0, help="секунд на одну задачу")
    p.add_argument("--deadline", type=float, default=3300.0, help="после стольких секунд новые задачи не берём")
    p.add_argument("--only", nargs="*", help="решать только эти task_id (для отладки)")
    p.add_argument("--log-dir", type=Path, default=os.environ.get("HARNESS_LOG_DIR"))
    # Второй проход (budget_agent/second.py): перепроверка ответов с сигналом сомнения из излишка бюджета.
    # Включён по умолчанию с v9: на честном hold-c5 +3 балла (102 -> 105 из 113), испорченных ответов 0.
    # --escalate — старое имя (раньше второй проход был только для застрявшего coding).
    p.add_argument("--second-pass", "--escalate", dest="second_pass", action="store_true",
                   default=os.environ.get("SECOND_PASS", "1") == "1",
                   help="второй проход по сигналам сомнения из излишка бюджета (по умолчанию включён)")
    p.add_argument("--no-second-pass", dest="second_pass", action="store_false", help="без второго прохода")
    p.add_argument("--no-compute-verify", dest="compute_verify", action="store_false",
                   default=os.environ.get("COMPUTE_VERIFY", "1") == "1",
                   help="во втором проходе не перепроверять math и data")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[harness {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def solutions_path(out: str | None) -> Path | None:
    if not out:
        return None
    p = Path(out)
    return p / "solutions.json" if p.is_dir() else p


def save_solutions(path: Path | None, solutions: dict[str, str]) -> None:
    if path is None:
        return
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(solutions, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        log(f"cannot write solutions to {path}: {e}")


async def main_async(args: argparse.Namespace) -> None:
    started = time.monotonic()
    # В контейнере всё пишем только в /tmp: корень read-only.
    log_dir = args.log_dir or Path(tempfile.gettempdir()) / "budget_agent_runs" / time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # До импорта остального openJiuwen: иначе он попытается создать ./logs в read-only /app.
    from budget_agent.ojw_logging import setup_openjiuwen_logging
    setup_openjiuwen_logging()

    from budget_agent.budget import Budget
    from budget_agent.planner import TIGHT_ON, estimate_cost, order_tasks, set_budget_pressure, task_cap
    from budget_agent.precheck import doubt_reason, infeasible_reason
    from budget_agent.solver import Attempt, solve
    from budget_agent.planner import ESCALATE_MIN_FRACTION, ESCALATE_MIN_LEFT
    from budget_agent.tasks import load_tasks, write_answer
    from budget_agent.answers import apply_decimals, fail_word

    tasks = load_tasks(args.tasks_dir)
    if args.only:
        tasks = [t for t in tasks if t.task_id in set(args.only)]
    budget = Budget(limit=args.budget, log_path=log_dir / "calls.jsonl")
    sol_path = solutions_path(args.out)
    solutions: dict[str, str] = {}
    log(f"{len(tasks)} tasks, budget {args.budget:.0f}, model {args.model}, logs in {log_dir}")
    if os.environ.get("BASE_URL_PROBE", "1") == "1":
        # Адрес без «/v1» дал бы 404 на каждом вызове (budget_agent/endpoint.py).
        from budget_agent.endpoint import resolve_base_url
        base, why = await asyncio.to_thread(resolve_base_url, os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"])
        if base != os.environ["LLM_BASE_URL"].strip().rstrip("/"):
            os.environ["LLM_BASE_URL"] = base
        log(f"llm base url: {why}")

    sem = asyncio.Semaphore(args.parallel)

    def record(row: dict) -> None:
        with (log_dir / "results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def mark_unsolvable(task) -> None:
        # Условие само говорит, что писать, если решить нельзя («write exactly UNSOLVABLE»): пишем это слово.
        # Без такого условия невыполнимую оставляем без ответа. Второй проход может заменить его настоящим ответом.
        word = fail_word(task.instruction)
        if word:
            write_answer(task, word, log)
            solutions[task.task_id] = word
            save_solutions(sol_path, solutions)
            log(f"{task.task_id}: no answer, the task says to write {word!r}")

    # Проверки «невыполнимо» без LLM. Жёсткие (противоречивые тесты, нет модуля) — задачу не запускаем вообще.
    # Мягкие (значение, имя или файл из условия не найдены в файлах задачи) — догадка о формулировке: такую
    # задачу решаем, но последней в очереди. PRECHECK_SOFT_SKIP=1 возвращает прежний пропуск.
    skipped = {}
    doubted = {}
    for t in tasks:
        try:
            reason = infeasible_reason(t)
        except Exception as e:  # проверка не должна мешать решать
            log(f"{t.task_id}: precheck error {type(e).__name__}: {e}")
            reason = None
        if not reason:
            try:
                doubt = doubt_reason(t)
            except Exception as e:
                log(f"{t.task_id}: precheck error {type(e).__name__}: {e}")
                doubt = None
            if doubt and os.environ.get("PRECHECK_SOFT_SKIP") == "1":
                reason = doubt
            elif doubt:
                doubted[t.task_id] = doubt
                log(f"{t.task_id}: may be infeasible ({doubt}): moved to the end of the queue")
        if reason:
            skipped[t.task_id] = reason
            log(f"{t.task_id}: skip, infeasible: {reason}")
            mark_unsolvable(t)
            record({"task": t.task_id, "type": t.task_type, "reward": t.reward, "answer": None,
                    "result": "precheck", "reason": reason, "cost": 0.0, "calls": 0, "retries": 0, "sec": 0.0})

    first: dict = {}  # task_id -> (task, Attempt) первого прохода: из них второй проход берёт сомнительные

    def on_answer(task, answer: str) -> None:
        answer = apply_decimals(task.instruction, answer)  # «round to 2 decimals» -> ровно 2 знака (answers.py)
        write_answer(task, answer, log)
        solutions[task.task_id] = answer
        save_solutions(sol_path, solutions)

    async def worker(task) -> None:
        async with sem:
            est = estimate_cost(task)
            if time.monotonic() - started > args.deadline:
                log(f"{task.task_id}: skip, deadline")
                return
            if budget.remaining < est:
                log(f"{task.task_id}: skip, budget left {budget.remaining:.0f} < est {est:.0f}")
                return
            t0 = time.monotonic()
            try:
                att = await asyncio.wait_for(solve(task, budget, args.model, task_cap=task_cap(task),
                                                   traj_dir=log_dir / "traj"),
                                             args.task_timeout)
            except Exception as e:  # одна упавшая задача не должна ронять прогон
                log(f"{task.task_id}: error {type(e).__name__}: {e}")
                # Задача осталась без ответа: второй проход решит её заново обычным путём (second.py, rerun).
                first[task.task_id] = (task, Attempt(
                    answer=None, result_type="timeout" if isinstance(e, asyncio.TimeoutError) else "exception",
                    cost=budget.by_task.get(task.task_id, 0.0), calls=0))
                return
            if att.answer is not None:
                on_answer(task, att.answer)
            elif att.result_type in ("unanswerable", "escalated_impossible"):
                mark_unsolvable(task)
            first[task.task_id] = (task, att)
            record({"task": task.task_id, "type": task.task_type, "reward": task.reward,
                    "answer": att.answer, "result": att.result_type, "cost": round(att.cost, 1),
                    "calls": att.calls, "retries": att.retries, "sec": round(time.monotonic() - t0, 1),
                    "pass": 1, "flags": list(att.flags)})
            log(f"{task.task_id}: {att.result_type} answer={att.answer!r} cost={att.cost:.0f} "
                f"calls={att.calls} spent={budget.spent:.0f}" + (f" flags={','.join(att.flags)}" if att.flags else ""))

    async def safe_worker(task) -> None:
        # Запись ответа и журнала стоит вне try в worker: сбой там (переполненный /tmp, кодировка) не должен
        # обрывать остальные задачи.
        try:
            await worker(task)
        except Exception as e:
            log(f"{task.task_id}: worker error {type(e).__name__}: {e}")

    queue = [t for t in order_tasks(tasks) if t.task_id not in skipped]
    queue = [t for t in queue if t.task_id not in doubted] + [t for t in queue if t.task_id in doubted]
    ratio = set_budget_pressure(args.budget, queue)
    if ratio < TIGHT_ON:
        log(f"tight budget: {args.budget:.0f} is {ratio:.2f} of the first-pass estimate, task caps scaled down")
    await asyncio.gather(*(safe_worker(t) for t in queue))
    log(f"first pass: {len(solutions)}/{len(tasks)} answered, spent {budget.spent:.0f} in {budget.calls} calls, "
        f"{time.monotonic() - started:.0f}s")

    # Второй проход: только сомнительные ответы и только из излишка (остаток минус резерв). Первый проход уже
    # записал все ответы, поэтому отключение прокси здесь ничего решённого не отнимает.
    if args.second_pass:
        from budget_agent.second import run_second_pass
        reserve = max(ESCALATE_MIN_LEFT, ESCALATE_MIN_FRACTION * args.budget)
        await run_second_pass(first, budget, args.model, sem, reserve,
                              lambda: args.deadline - (time.monotonic() - started), log_dir / "traj",
                              on_answer, record, log, args.task_timeout, compute=args.compute_verify,
                              prechecked=[t for t in tasks if t.task_id in skipped])
    log(f"done: {len(solutions)}/{len(tasks)} answered, spent {budget.spent:.0f} in {budget.calls} calls, "
        f"{time.monotonic() - started:.0f}s")
    from budget_agent.solver import RATE
    if RATE.waited:
        log(f"rate limit {RATE.rpm}/min: waited {RATE.waited:.0f}s in total")


def main() -> None:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    args = parse_args()
    for var in ("LLM_BASE_URL", "LLM_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"{var} is not set")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
