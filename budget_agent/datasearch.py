"""search по файлам данных: вопрос о таблице или конфиге решает код, а не чтение глазами.

В базе знаний бывают не только карточки, но и таблицы и конфиги (csv, tsv, json, yaml, SQLite, xml). На вопросах
«какой sku с наименьшим остатком», «сумма цена × остаток», «версия сервиса в config/services.yaml» модель без
рассуждений ошибалась, хотя нужный файл был в контексте целиком: брала соседнюю запись YAML (jul search_0033,
4.2.0 вместо 1.7.3), считала сумму в уме (search_0014, 10453 вместо 39803), а агент с `grep -A4` видел 4 строки
таблицы из 8 (search_0032). На большой таблице глазами не ответить вовсе: контекст search — 12 000 символов, из
файла без имени в вопросе берутся только строки со словами вопроса.

Маршрут выбирается без токенов:
- named: вопрос называет файл данных («in config/services.yaml»);
- aggregate: в вопросе подсчёт или выбор по значению (наименьший, сумма, сколько) и поле таблицы из базы, а все
  отличительные слова вопроса (имена, коды) есть в файлах данных. Если слово есть только в карточках («компании
  со штаб-квартирой в Tallinn»), вопрос смешанный: его решает обычный search по карточкам;
- fields: карточки по вопросу не собрались (в вопросе нет имён), а поле таблицы из базы в нём есть — код пробуем
  до агента с grep.

База только из карточек (как в примерах организаторов) этот маршрут не включает: файлов данных нет.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import search as direct_search
from .prefetch import _size, describe_file, json_skeleton

MAX_DATA_BYTES = 2_000_000
MAX_LISTED = 30
MAX_DESCRIBED = 4
DATA_SUFFIXES = {".csv", ".tsv", ".psv", ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".xml", ".ini", ".properties",
                 ".cfg", ".conf", ".toml", ".xlsx", ".xlsm", ".db", ".sqlite", ".sqlite3"}
_SQLITE_MAGIC = b"SQLite format 3\x00"

# Подсчёт или выбор по значению. «phone number of X» — не подсчёт.
_AGG = re.compile(r"\b(?:smallest|largest|lowest|highest|biggest|greatest|least|most|fewest|maximum|minimum|max|min|"
                  r"total|sum|summed|average|mean|median|cheapest|count|top \d+|how many|"
                  r"(?<!phone )(?<!build )(?<!serial )(?<!part )(?<!version )(?<!model )number of)\b", re.I)
# Поля, которые есть почти в любой таблице и ничего не говорят о вопросе.
_GENERIC = {"name", "names", "id", "ids", "key", "value", "values", "type", "title", "text", "description", "notes",
            "note", "data", "item", "items", "info", "details", "label", "code", "row", "rows"}
_STOP = {"the", "and", "for", "with", "which", "what", "who", "whom", "whose", "how", "many", "much", "are", "was",
         "were", "has", "have", "had", "does", "did", "its", "their", "this", "that", "these", "those", "write",
         "answer", "only", "final", "file", "files", "directory", "current", "knowledge", "base", "find", "use",
         "tools", "bash", "without", "decimals", "there", "than", "all", "each", "any", "one", "per", "into", "out",
         "not", "but", "after", "before", "over", "under", "from", "into", "list", "listed", "give", "number", "total",
         "sum", "most", "least", "max", "min", "count", "top", "whole", "integer", "round", "rounded"}
_MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


@dataclass
class DataFile:
    path: Path
    rel: str
    fields: set[str]
    records: int  # записей в самой большой таблице файла; 0 — конфиг «ключ = значение»
    text: str
    raw: set[str] = field(default_factory=set)  # имена полей как в файле


@dataclass
class Route:
    kind: str  # named | aggregate | fields
    files: list[DataFile]


def _variants(word: str) -> set[str]:
    """Формы слова для сравнения с полем: «on-call» ~ «oncall», «errors» ~ «error», «max_temp_c» ~ «max», «temp»."""
    w = word.lower()
    out = {w, re.sub(r"[-_.\s]", "", w)}
    out |= {p for p in re.split(r"[-_.\s]+", w) if len(p) >= 3}
    for x in list(out):
        if len(x) > 4 and x.endswith("ies"):
            out.add(x[:-3] + "y")
        elif len(x) > 4 and x.endswith("es"):
            out |= {x[:-2], x[:-1]}
        elif len(x) > 3 and x.endswith("s") and not x.endswith("ss"):
            out.add(x[:-1])
    return out


def _field_tokens(names) -> set[str]:
    out: set[str] = set()
    for n in names:
        if isinstance(n, str) and 0 < len(n) <= 60:
            out |= _variants(n)
    return out - _GENERIC


def _walk(x, fields: set, depth: int = 0) -> int:
    """Ключи записей и число записей в самой большой таблице (список словарей или словарь словарей). Ключи словаря
    словарей — имена записей («firmware-update:»), а не поля: их не берём."""
    if depth > 5:
        return 0
    best = 0
    if isinstance(x, dict):
        sub = [v for v in x.values() if isinstance(v, dict)]
        if len(x) >= 2 and len(sub) >= 2 and len(sub) >= len(x) // 2:
            best = len(sub)
        else:
            fields.update(k for k in x if isinstance(k, str))
        for v in list(x.values())[:50]:
            best = max(best, _walk(v, fields, depth + 1))
    elif isinstance(x, list):
        dicts = [v for v in x if isinstance(v, dict)]
        if len(dicts) >= 2:
            best = len(dicts)
        for v in x[:3]:
            best = max(best, _walk(v, fields, depth + 1))
    return best


def _delimited(text: str) -> tuple[set[str], int] | None:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    first = lines[0]
    delim = max((",", "\t", "|", ";"), key=first.count)
    if first.count(delim) < 1:
        return None
    header = next(csv.reader(io.StringIO(first), delimiter=delim))
    return {h.strip() for h in header if h.strip()}, len(lines) - 1


def inspect(p: Path, root: Path) -> DataFile | None:
    rel = p.relative_to(root).as_posix()
    try:
        if p.stat().st_size > MAX_DATA_BYTES:
            return None
        raw = p.read_bytes()
    except OSError:
        return None
    suf = p.suffix.lower()
    fields: set[str] = set()
    records = 0
    text = ""
    try:
        if raw.startswith(_SQLITE_MAGIC):
            con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro&immutable=1", uri=True)
            try:
                for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall():
                    cur = con.execute(f'SELECT * FROM "{t}" LIMIT 2000')
                    rows = cur.fetchall()
                    fields |= {t} | {d[0] for d in cur.description}
                    records = max(records, len(rows))
            finally:
                con.close()
            text = direct_search._sqlite_text(p) or ""
        elif suf in (".xlsx", ".xlsm"):
            import openpyxl
            wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
            try:
                for ws in wb.worksheets[:6]:
                    first = next(ws.iter_rows(values_only=True, max_row=1), ())
                    fields |= {str(v) for v in first if v is not None} | {ws.title}
                    records = max(records, (ws.max_row or 1) - 1)
            finally:
                wb.close()
        elif suf not in DATA_SUFFIXES or b"\0" in raw[:2048]:
            return None
        else:
            text = raw.decode("utf-8", errors="replace")
            if suf in (".csv", ".tsv", ".psv"):
                d = _delimited(text)
                if d:
                    fields, records = d
            elif suf == ".json":
                records = _walk(json.loads(text), fields)
            elif suf in (".jsonl", ".ndjson"):
                lines = [ln for ln in text.splitlines() if ln.strip()]
                for ln in lines[:3]:
                    _walk(json.loads(ln), fields)
                records = len(lines)
            elif suf in (".yaml", ".yml"):
                import yaml
                records = _walk(yaml.safe_load(text), fields)
            elif suf == ".xml":
                import xml.etree.ElementTree as ET
                tree = ET.fromstring(text)
                for el in tree.iter():
                    fields.add(el.tag)
                    fields |= set(el.attrib)
                    tags = [c.tag for c in el]
                    if tags:
                        records = max(records, max(tags.count(t) for t in set(tags)))
            else:  # ini, properties, cfg, conf, toml: «ключ = значение», секции
                fields = set(re.findall(r"^\s*([\w.-]+)\s*[=:]", text, re.M)) | \
                    set(re.findall(r"^\s*\[([^\]\n]+)\]", text, re.M))
    except Exception:  # noqa: BLE001 — битый файл не должен ронять задачу: просто не файл данных
        return None
    return DataFile(p, rel, _field_tokens(fields), records, text, {f for f in fields if isinstance(f, str)})


def data_files(workdir: Path) -> list[DataFile]:
    out = []
    for p in sorted(workdir.rglob("*")):
        if not p.is_file() or p.name in ("task.json", "answer.txt") or "__pycache__" in p.parts:
            continue
        if p.suffix.lower() not in DATA_SUFFIXES:
            try:
                with p.open("rb") as f:
                    if f.read(16) != _SQLITE_MAGIC:
                        continue
            except OSError:
                continue
        d = inspect(p, workdir)
        if d is not None:
            out.append(d)
    return out


def question_words(question: str) -> dict[str, set[str]]:
    """Слова вопроса с их формами: {«errors»: {«errors», «error»}, …}."""
    out: dict[str, set[str]] = {}
    for w in _WORD.findall(question):
        if len(w) >= 3 and w.lower() not in _STOP:
            out[w.lower()] = _variants(w) - _STOP
    return out


def _hits(d: DataFile, words: dict[str, set[str]]) -> int:
    """Сколько разных слов вопроса совпало с полями файла."""
    return sum(1 for v in words.values() if v & d.fields)


def _distinct_terms(question: str) -> list[str]:
    return [t for t in direct_search.question_terms(question) if t not in _MONTHS and not re.fullmatch(r"\d{1,3}", t)]


# Запасной маршрут (карточки не собрались) — только при двух и больше совпавших полях: одно слово «product» в
# вопросе про архив заметок (jul search_0044) уводило к таблице товаров, а ответ был в notes.txt.gz.
FIELDS_MIN_HITS = 2


def route(question: str, workdir: Path) -> Route | None:
    files = data_files(workdir)
    if not files:
        return None
    named = {m.lower().lstrip("./") for m in direct_search._FILE_MENTION.findall(question)}
    hit = [d for d in files if named and (d.path.name.lower() in named or any(d.rel.lower().endswith(n) for n in named))]
    if hit:
        return Route("named", hit)
    words = question_words(question)
    scored = [(_hits(d, words), d) for d in files if d.records >= 2]
    # При равном числе совпавших полей — сначала большая таблица: «product» есть и в products.xml на 20 товаров,
    # и в products.csv на 60, вопрос «в списке товаров» — про csv.
    matched = [d for n, d in sorted(scored, key=lambda x: (-x[0], -x[1].records, x[1].rel)) if n]
    if not matched:
        return None
    # Отличительное слово вопроса (имя, код, «Toys») только в карточках или нигде: вопрос не о таблице, а код
    # посчитал бы по всей таблице и ответил на невыполнимую задачу.
    blob = "\n".join(d.text for d in files).lower()
    if not all(t.lower() in blob for t in _distinct_terms(question)):
        return None
    if _AGG.search(question):
        return Route("aggregate", matched[:MAX_DESCRIBED])
    if max(n for n, _ in scored) >= FIELDS_MIN_HITS:
        return Route("fields", [d for d in matched if _hits(d, words) >= FIELDS_MIN_HITS][:MAX_DESCRIBED])
    return None


def _describe(d: DataFile, root: Path) -> str | None:
    if d.path.suffix.lower() in (".yaml", ".yml") and d.text:
        # Первые строки YAML не показывают, где кончается одна запись и начинается другая: показываем структуру.
        try:
            import yaml
            sk = json_skeleton(json.dumps(yaml.safe_load(d.text), ensure_ascii=False, default=str))
        except Exception:  # noqa: BLE001
            sk = None
        if sk:
            return f"[structure of {d.rel} (YAML, read it with yaml.safe_load), values shortened]\n{sk}"
    if d.path.suffix.lower() in (".ini", ".properties", ".cfg", ".conf", ".toml") and len(d.text) <= 600:
        return f"[{d.rel}]\n{d.text.strip()}"
    return describe_file(d.path, root, head=3)


SEARCH_DATA_RULE = (" Paths below are relative to the current directory (some names start with a dot). Read the "
                    "data file(s) shown below in code: never copy values by eye. If the question goes through a link "
                    "(find a record by one field, then take another field or look it up in another file), do every "
                    "step in code. Print only the final value asked for (a name, id or number) exactly as written in "
                    "the data.")
# Пустой результат выборки — не ответ: «[]» записывался ответом на вопрос о несуществующем сервисе (стресс-набор).
EMPTY_RESULT = re.compile(r"\[\s*\]|\{\s*\}|\(\s*\)|set\(\)|none|null|nan|''|\"\"", re.I)
_PERSON_LIKE = re.compile(r"[A-Z][\w'.-]*\s+[A-Z]")


def not_a_name(question: str, answer: str) -> bool:
    """Вопрос «кто», а ответ не похож на имя: код остановился на промежуточном шаге и напечатал ключ записи
    («audit-log» вместо дежурного инженера этого сервиса, стресс-набор)."""
    return bool(direct_search._WHO.search(question)) and not _PERSON_LIKE.search(answer)


def prefetch(workdir: Path, files: list[DataFile], question: str = "") -> str:
    """Список файлов данных (без карточек: их десятки, а коду они не нужны) и структура выбранных файлов."""
    allf = [p for p in sorted(workdir.rglob("*")) if p.is_file() and p.name not in ("task.json", "answer.txt")
            and "__pycache__" not in p.parts]
    data = {d.path for d in data_files(workdir)}
    shown = [p for p in allf if p in data][:MAX_LISTED]
    listing = "\n".join(f"{p.relative_to(workdir).as_posix()}  ({_size(p)})" for p in shown)
    other: dict[str, int] = {}
    for p in allf:
        if p not in data:
            other[p.suffix.lower() or p.name] = other.get(p.suffix.lower() or p.name, 0) + 1
    rest = ", ".join(f"{n} {s}" for s, n in sorted(other.items(), key=lambda x: -x[1]))
    blocks = [f"[data files, hidden ones included]\n{listing}" + (f"\n(other files: {rest})" if rest else "")]
    # Какие поля совпали со словами вопроса: без этого модель взяла owner из YAML за дежурного инженера и не
    # заглянула в services.psv с колонкой oncall (стресс-набор, вопрос «кто дежурит у сервиса версии X»).
    qv = set().union(*question_words(question).values()) if question else set()
    hints = []
    for d in files[:MAX_DESCRIBED]:
        m = sorted(n for n in d.raw if (_variants(n) - _GENERIC) & qv)
        if m:
            hints.append(f"{d.rel}: {', '.join(m[:8])}")
    if hints:
        blocks.append("[fields matching words of the question]\n" + "\n".join(hints))
    for d in files[:MAX_DESCRIBED]:
        b = _describe(d, workdir)
        if b:
            blocks.append(b)
    return "Already run for you:\n" + "\n".join(blocks)
