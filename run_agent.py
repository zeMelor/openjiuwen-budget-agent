"""Точка входа харнесса: python run_agent.py --tasks-dir DIR

В контейнере: python /app/run_agent.py --tasks-dir /data/tasks
Окружение: LLM_BASE_URL, LLM_API_KEY (обязательно), OUT (необязательно: файл или каталог для сводки решений;
в каталог пишутся ещё <каталог задачи>/answer.txt), BUDGET, TASKS (если нет --tasks-dir).
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


def parse_budget(raw: str | None) -> float | None:
    """«100000», «100_000», «100k» -> число; кривое значение -> None."""
    raw = (raw or "").strip().replace("_", "").replace(" ", "")
    if not raw:
        return None
    mult = 1000.0 if raw[-1] in "kK" else 1.0
    try:
        value = float(raw.rstrip("kK")) * mult
    except ValueError:
        return None
    return value if 0 < value < float("inf") else None


def env_budget(default: float = 100_000.0) -> float:
    """Бюджет из окружения: TOKEN_BUDGET (наше имя) или BUDGET (так бюджет передают при проверке).
    Кривое значение не должно ронять прогон: тогда 100k."""
    for name in ("TOKEN_BUDGET", "BUDGET"):
        value = parse_budget(os.environ.get(name))
        if value:
            return value
    return default


# Неизвестный параметр с одним из этих слов в имени считаем файлом или каталогом для решений: регламент говорит,
# что проект «получает в параметрах имя папки с задачами и имя файла для сохраненных решений», а имя параметра
# не названо.
OUT_WORDS = ("out", "answer", "solution", "result")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбор терпимый: лишний или незнакомый параметр в команде запуска не должен ронять прогон (строгий argparse
    завершился бы с кодом 2 до первой задачи). Незнакомое пишется в журнал; каталог задач, бюджет и файл решений
    узнаются и по другим именам, каталог задач — ещё из TASKS и /data/tasks."""
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--tasks-dir", "--tasks", "--task-dir", dest="tasks_dir", type=Path, default=None)
    p.add_argument("--out", "--output", "--out-file", "--output-file", "-o", dest="out", default=None,
                   help="файл или каталог для сводки решений (по умолчанию OUT)")
    p.add_argument("--budget", default=None, help="бюджет токенов (по умолчанию BUDGET или 100k)")
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
    args, extra = p.parse_known_args(argv)
    notes: list[str] = []
    args.budget = parse_budget(args.budget) if args.budget is not None else None
    positional: list[str] = []
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok.startswith("-") and len(tok) > 1:
            name, eq, val = tok.partition("=")
            if not eq and i + 1 < len(extra) and not extra[i + 1].startswith("-"):
                i += 1
                val = extra[i]
            low = name.lower()
            if val and "task" in low and args.tasks_dir is None:
                args.tasks_dir = Path(val)
                notes.append(f"{name} {val}: tasks dir")
            elif val and "budget" in low and parse_budget(val) and args.budget is None:
                args.budget = parse_budget(val)
                notes.append(f"{name} {val}: budget")
            elif val and any(w in low for w in OUT_WORDS) and args.out is None:
                args.out = val
                notes.append(f"{name} {val}: solutions file")
            else:
                notes.append(f"ignored {name}" + (f" {val}" if val else ""))
        else:
            positional.append(tok)
        i += 1
    for tok in positional:
        if args.tasks_dir is None and Path(tok).is_dir():
            args.tasks_dir = Path(tok)
            notes.append(f"{tok}: tasks dir")
        elif args.out is None:
            args.out = tok
            notes.append(f"{tok}: solutions file")
        else:
            notes.append(f"ignored {tok}")
    if args.tasks_dir is None:
        for cand in (os.environ.get("TASKS"), "/data/tasks"):
            if cand and Path(cand).is_dir():
                args.tasks_dir = Path(cand)
                notes.append(f"no --tasks-dir: using {cand}")
                break
    if args.out is None:
        args.out = os.environ.get("OUT") or None
    if args.budget is None:
        args.budget = env_budget()
    args.arg_notes = notes
    return args


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


def mirror_answer(out_dir: Path | None, task_dir_name: str, text: str) -> None:
    """OUT — каталог: кроме solutions.json пишем туда <каталог задачи>/answer.txt, как в каталоге задач. Страховка
    на случай, если каталог задач смонтирован без права записи для контейнера."""
    if out_dir is None:
        return
    try:
        d = out_dir / task_dir_name
        d.mkdir(parents=True, exist_ok=True)
        (d / "answer.txt").write_text(text.strip() + "\n", encoding="utf-8")
    except OSError as e:
        log(f"cannot write {out_dir / task_dir_name / 'answer.txt'}: {e}")


def can_write(d: Path) -> bool:
    """Можно ли создать файл в каталоге: пробный файл создаём и сразу удаляем."""
    probe = d / f".write_probe_{os.getpid()}"
    try:
        probe.write_text("", encoding="utf-8")
    except OSError:
        return False
    try:
        probe.unlink()
    except OSError:
        pass
    return True


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
    out_dir = Path(args.out) if args.out and Path(args.out).is_dir() else None
    solutions: dict[str, str] = {}
    unwritten: set[str] = set()  # задачи, чей answer.txt не записался ни в каталог задачи, ни в workspace/
    log(f"{len(tasks)} tasks, budget {args.budget:.0f}, model {args.model}, logs in {log_dir}")
    if tasks and not (can_write(tasks[0].dir) or can_write(tasks[0].workdir)):
        log(f"WARNING: cannot create files in {tasks[0].dir}: answer.txt may not be saved; answers also go to "
            f"{sol_path or 'nowhere else (OUT is not set)'} and to this log")
    if sol_path is not None:
        log(f"solutions also go to {sol_path}" + (f" and {out_dir}/<task>/answer.txt" if out_dir else ""))
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

    def emit(task, answer: str) -> None:
        """Ответ — в каталог задачи и workspace/, в сводку OUT и в зеркало OUT/<задача>/answer.txt."""
        if write_answer(task, answer, log):
            unwritten.discard(task.task_id)
        else:
            unwritten.add(task.task_id)
        solutions[task.task_id] = answer
        save_solutions(sol_path, solutions)
        mirror_answer(out_dir, task.dir.name, answer)

    def mark_unsolvable(task) -> None:
        # Условие само говорит, что писать, если решить нельзя («write exactly UNSOLVABLE»): пишем это слово.
        # Без такого условия невыполнимую оставляем без ответа. Второй проход может заменить его настоящим ответом.
        word = fail_word(task.instruction)
        if word:
            emit(task, word)
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
        emit(task, answer)

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
    if unwritten:
        log(f"WARNING: answer.txt not saved for {len(unwritten)} tasks: {sorted(unwritten)}")
    # Все ответы одной строкой: если файлы записать не удалось, ответы остаются хотя бы в журнале контейнера.
    log("answers: " + json.dumps(solutions, ensure_ascii=False, sort_keys=True))
    from budget_agent.solver import RATE
    if RATE.waited:
        log(f"rate limit {RATE.rpm}/min: waited {RATE.waited:.0f}s in total")


def main() -> None:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    args = parse_args()
    for note in args.arg_notes:
        log(f"args: {note}")
    if args.tasks_dir is None:
        sys.exit("tasks dir is not set: --tasks-dir DIR (or TASKS)")
    for var in ("LLM_BASE_URL", "LLM_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"{var} is not set")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
