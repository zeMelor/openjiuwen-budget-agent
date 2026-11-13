"""Загрузка задач и запись ответов."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Task:
    task_id: str
    task_type: str
    instruction: str
    reward: float
    dir: Path

    @property
    def workdir(self) -> Path:
        """Каталог, где агент работает: workspace/, если он есть, иначе сам каталог задачи."""
        ws = self.dir / "workspace"
        return ws if ws.is_dir() else self.dir


def load_tasks(tasks_dir: Path) -> list[Task]:
    tasks = []
    for d in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        meta_path = d / "task.json"
        if not meta_path.exists():
            continue
        # Одна кривая задача (не JSON, не объект, битая кодировка, reward не число) не должна ронять весь прогон:
        # без этой обёртки харнесс падал с exit 1 до первого вызова модели (bench/robust/REPORT.md).
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(meta, dict):
                continue
            task = Task(
                task_id=str(meta.get("task_id") or d.name),
                task_type=str(meta.get("task_type") or "unknown"),
                instruction=str(meta.get("instruction") or ""),
                reward=_reward(meta.get("reward")),
                dir=d,
            )
            if task.task_type not in KNOWN_TYPES:
                task.task_type = infer_type(task)
            tasks.append(task)
        except Exception:
            continue
    return tasks


KNOWN_TYPES = ("coding", "search", "math", "data_analysis")
_DATA_SUFFIXES = {".csv", ".tsv", ".psv", ".json", ".jsonl", ".xlsx", ".xls", ".sqlite", ".sqlite3", ".db", ".parquet",
                  ".xml", ".yaml", ".yml", ".log", ".gz", ".txt", ".dat"}


def infer_type(task: Task) -> str:
    """Тип не из регламента («sql», «shell», «logic» в чужих наборах): решатель выбираем по файлам задачи. Раньше
    такие задачи шли к обычному агенту — на 36 задачах sql 15/112 баллов за 11 660 токенов (jul-os), хотя это
    задачи по данным в SQLite, которые решатель data уже умеет читать."""
    root = task.workdir
    files = [p for p in root.rglob("*") if p.is_file() and p.name not in ("task.json", "answer.txt")
             and "__pycache__" not in p.parts] if root.is_dir() else []
    rels = [p.relative_to(root) for p in files]
    if any(r.name.startswith("test_") and r.suffix == ".py" or r.name.endswith("_test.py")
           or {"tests", "test"} & set(r.parts[:-1]) for r in rels):
        return "coding"
    if (root / "kb").is_dir():
        return "search"
    if files:
        return "data_analysis"
    return "math"


def _reward(value) -> float:
    """reward как число; если не число или не конечное — 0 (задача всё равно попадёт в очередь)."""
    try:
        r = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return r if math.isfinite(r) else 0.0


def write_answer(task: Task, answer: str, log=None) -> bool:
    """Пишет answer.txt в каталог задачи и, если есть, в workspace/. True, если записали хоть куда-то.

    Регламент говорит «в каталоге задачи», инструкция задачи говорит «в текущей директории» (это workspace/).
    Пока организаторы не уточнили, пишем в оба места. Ошибка записи в одно место не мешает второму и не
    роняет прогон: при --cap-drop ALL root пишет только туда, где это разрешено правами каталога.
    """
    text = answer.strip() + "\n"
    ok = False
    for d in {task.dir, task.workdir}:
        try:
            (d / "answer.txt").write_text(text, encoding="utf-8")
            ok = True
        except OSError as e:
            if log:
                log(f"{task.task_id}: cannot write {d / 'answer.txt'}: {e}")
    return ok