"""Первый шаг поиска делает харнесс: grep по именам из вопроса, без LLM и без токенов на вызов.

В search-задачах модель почти всегда начинает с `grep -ri -A4 '<имя из вопроса>' kb/`. Этот вызов стоит
~45 единиц (вход всего промпта + выход команды), а результат предсказуем. Харнесс выполняет такой поиск сам
и кладёт вывод в запрос; модель начинает со второго шага. Если имён в вопросе нет, ничего не делаем.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .precheck import _NAME, _NOT_NAMES
from .tasks import Task

AFTER = 4
MAX_CHARS = 900
MAX_FILE_BYTES = 200_000


def _grep(root: Path, needle: str) -> list[str]:
    """Как grep -ri -A4 needle: строки «путь:строка» и «путь-строка» для контекста после совпадения."""
    out: list[str] = []
    n = needle.lower()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in ("task.json", "answer.txt") or p.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        last = -1
        for i, line in enumerate(lines):
            if n in line.lower():
                if out and last >= 0 and i > last + 1:
                    out.append("--")
                for j in range(max(i, last + 1), min(len(lines), i + AFTER + 1)):
                    out.append(f"{rel}{':' if j == i else '-'}{lines[j]}")
                last = min(len(lines) - 1, i + AFTER)
    return out


MAX_LISTED = 40


def prefetch_files(task: Task, max_files: int = 4, head: int = 2) -> str | None:
    """Для задач по данным: первые строки и число строк каждого файла (как head -2 и wc -l).
    Модель иначе тратит отдельный вызов на `head -1 sales.csv`, чтобы увидеть заголовок и разделитель.

    Файлов больше max_files (помесячные csv, логи по дням): список файлов и заголовки по одному файлу каждого типа —
    раньше в этом случае модель не видела ничего. Двоичные файлы (SQLite, gzip, xlsx) описываются по структуре."""
    root = task.workdir
    # Без workspace/ файлы лежат прямо в каталоге задачи (chillcoders): раньше тогда не показывалось ничего, и модель
    # угадывала колонки (product и date вместо category и quarter, data_easy_0003).
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.name not in ("task.json", "answer.txt")
             and "__pycache__" not in p.parts]
    if not files:
        return None
    blocks = []
    shown_files = files
    if len(files) > max_files:
        listing = "\n".join(f"{p.relative_to(root).as_posix()}  ({_size(p)})" for p in files[:MAX_LISTED])
        more = f"\n... {len(files) - MAX_LISTED} more files" if len(files) > MAX_LISTED else ""
        blocks.append(f"[{len(files)} files]\n{listing}{more}")
        by_suffix: dict[str, list[Path]] = {}
        for p in files:
            by_suffix.setdefault(p.suffix.lower(), []).append(p)
        # Сначала самый частый тип: это и есть данные задачи (7 csv по дням важнее одного readme). Из него —
        # несколько файлов с разной структурой: помесячные csv «от разных инструментов» отличались порядком колонок
        # и форматом дат, модель видела один файл и молча теряла строки другого формата (hold-c8).
        groups = sorted(by_suffix.values(), key=lambda g: -len(g))
        shown_files = _distinct_layouts(groups[0]) + [g[0] for g in groups[1:3]]
    for p in shown_files:
        block = describe_file(p, root, head)
        if block:
            blocks.append(block)
    return ("Already run for you:\n" + "\n".join(blocks)) if blocks else None


MAX_LAYOUTS = 4


def _shape(value: str) -> str:
    """Форма значения: цифры -> 9, буквы -> a («2025-04-04T14:08» -> «9-9-9a9:9»)."""
    s = re.sub(r"[A-Za-z]+", "a", re.sub(r"\d+", "9", value.strip()))
    return s or "_"


def _layout_signature(p: Path) -> tuple:
    try:
        with p.open(encoding="utf-8", errors="replace") as f:
            first, second = f.readline().strip(), f.readline().strip()
    except OSError:
        return ("?",)
    delim = max((",", ";", "\t", "|"), key=first.count)
    fields = tuple(sorted(x.strip().lower() for x in first.split(delim)))
    shapes = tuple(sorted(_shape(v) for v in second.split(delim)))
    return delim, fields, shapes


def _distinct_layouts(group: list[Path]) -> list[Path]:
    """Файлы группы с разными заголовками или формой значений первой строки данных, не больше MAX_LAYOUTS."""
    out, seen = [], set()
    for p in group:
        sig = _layout_signature(p)
        if sig not in seen:
            seen.add(sig)
            out.append(p)
        if len(out) >= MAX_LAYOUTS:
            break
    return out


def _size(p: Path) -> str:
    try:
        n = p.stat().st_size
    except OSError:
        return "?"
    return f"{n} bytes" if n < 10_000 else f"{n // 1000} KB"


_SQLITE_MAGIC = b"SQLite format 3\x00"


def describe_file(p: Path, root: Path, head: int = 2) -> str | None:
    rel = p.relative_to(root).as_posix()
    try:
        start = p.open("rb").read(2048)
    except OSError:
        return None
    suffix = p.suffix.lower()
    if start.startswith(_SQLITE_MAGIC):
        return _describe_sqlite(p, rel)
    if suffix == ".gz" or start[:2] == b"\x1f\x8b":
        import gzip
        try:
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                lines = [next(f, "").rstrip("\n") for _ in range(head)]
        except (OSError, EOFError):
            return f"[{rel}: gzip, {_size(p)}, cannot read]"
        return f"[{rel}: gzip-compressed text, {_size(p)}; first {head} lines after gzip -dc]\n" + \
            "\n".join(ln[:200] for ln in lines)
    if suffix in (".xlsx", ".xlsm"):
        return _describe_xlsx(p, rel)
    if b"\x00" in start:
        return f"[{rel}: binary file, {_size(p)}]"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    skeleton = json_skeleton(text) if suffix == ".json" else None
    if skeleton:
        # Первые строки JSON («{», «"warehouses": [») не показывают полей: модель угадывала имена и падала
        # на KeyError (hold-c4: «quantity» в условии, «qty» в файле).
        return f"[structure of {rel}: first item of each list, values shortened; {len(lines)} lines]\n{skeleton}"
    if suffix in (".jsonl", ".ndjson") and lines:
        first = json_skeleton(lines[0])
        if first:
            return f"[{rel}: JSON Lines, {len(lines)} lines; structure of line 1]\n{first}"
    if suffix == ".xml":
        return f"[first 400 characters of {rel}; file has {len(lines)} lines]\n{text[:400]}"
    table = _first_md_table(lines)
    if table is not None and suffix not in (".csv", ".tsv"):
        # Отчёт в Markdown с таблицей внутри: первые строки — заголовок отчёта, таблицы модель не видела и угадывала
        # номер колонки (hold-c8, ответ 7 вместо 1942313).
        start, rows = table
        intro = [ln[:200] for ln in lines[:start] if ln.strip()][:2]
        return (f"[{rel}: {len(lines)} lines; first lines, then the first table (line {start + 1}), header and 2 rows]\n"
                + "\n".join(intro + ["..."] + [ln[:200] for ln in rows]))
    if suffix not in (".csv", ".tsv"):
        head = max(head, 4)  # текст и логи: двух строк мало, чтобы увидеть формат
    shown = "\n".join(ln[:200] for ln in lines[:head])
    return f"[output of head -{head} {rel}; file has {len(lines)} lines]\n{shown}"


def _first_md_table(lines: list[str]) -> tuple[int, list[str]] | None:
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            return i, lines[i:i + 4]
    return None


def _describe_sqlite(p: Path, rel: str) -> str:
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'view') "
                               "ORDER BY name").fetchall()
            parts = []
            for name, sql in rows[:12]:
                try:
                    n = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                except sqlite3.Error:
                    n = "?"
                parts.append(f"{(sql or name)[:300]}  -- {n} rows")
        finally:
            con.close()
    except sqlite3.Error as e:
        return f"[{rel}: SQLite database, {_size(p)}; cannot open: {e}]"
    return f"[{rel}: SQLite database (use sqlite3), {_size(p)}; schema]\n" + "\n".join(parts)


def _describe_xlsx(p: Path, rel: str) -> str:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        parts = []
        for ws in wb.worksheets[:4]:
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i >= 2:
                    break
                rows.append(", ".join("" if v is None else str(v)[:30] for v in row[:15]))
            parts.append(f"sheet {ws.title!r} ({ws.max_row} rows): " + " | ".join(rows))
        wb.close()
        return f"[{rel}: Excel workbook (use openpyxl or pandas), {_size(p)}]\n" + "\n".join(parts)
    except Exception as e:  # noqa: BLE001 — описание файла не должно ронять решение
        return f"[{rel}: Excel workbook, {_size(p)}; cannot read: {type(e).__name__}]"


def json_skeleton(text: str, limit: int = 700) -> str | None:
    """Структура JSON: у каждого списка первый элемент, строки обрезаны, не больше 12 ключей на объект."""
    try:
        data = json.loads(text)
    except ValueError:
        return None

    def sk(x, depth=0):
        if depth > 8:
            return "..."
        if isinstance(x, dict):
            return {k: sk(v, depth + 1) for k, v in list(x.items())[:12]}
        if isinstance(x, list):
            return [sk(x[0], depth + 1), f"... {len(x)} items"] if x else []
        if isinstance(x, str):
            return x[:24]
        return x

    out = json.dumps(sk(data), ensure_ascii=False)
    return out if len(out) <= limit else out[:limit] + " ...[cut]"


def prefetch_search(task: Task) -> str | None:
    root = task.workdir
    if root == task.dir:
        return None
    names = []
    for name in _NAME.findall(task.instruction):
        if name.split()[0] not in _NOT_NAMES and name not in names:
            names.append(name)
    blocks = []
    for name in names[:2]:
        hits = _grep(root, name)
        if hits:
            text = "\n".join(hits)
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + "\n...[cut]"
            blocks.append(f"[output of grep -ri -A{AFTER} '{name}' .]\n{text}")
    if not blocks:
        return None
    return "Already run for you:\n" + "\n".join(blocks)
