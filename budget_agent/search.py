"""search в один вызов: карточки собирает харнесс, модель только отвечает.

Обычный агент тратит на search 1–4 вызова (grep, cat, снова grep), 100–270 на выполнимую задачу и 240–400 на
невыполнимую. Здесь харнесс без токенов берёт карточки, названные в вопросе, и идёт по ссылкам: имя другой
карточки в тексте — ссылка. Модель видит только эти карточки и отвечает одним вызовом.

Что учтено (наш набор search-x1, трудные варианты):
- заголовок «# Profile: Clara Varga» — имя после двоеточия;
- псевдонимы: «Also known as: Bill Okafor», «Née | Smith» (Nadia Muller = Nadia Smith);
- человек без своей карточки (строка в таблице команды): берём карточки, где его имя упомянуто;
- обратный поиск и подсчёт («кто из X учится у Y», «сколько человек работает в X»): карточки, ссылающиеся
  на имена из вопроса, сначала те, что ссылаются на все сразу;
- глубина по числу отношений в вопросе (ментор ментора работодателя — три шага).

Если в вопросе нет имён из базы или карточек слишком много, решатель возвращает None, и задачу берёт обычный
агент с grep.
"""
from __future__ import annotations

import gzip
import json
import re
import sqlite3
import zlib
from dataclasses import dataclass, field
from pathlib import Path

MAX_NOTE_BYTES = 50_000
# Базы на hold-c7 — 5–9,4 тысячи символов. При лимите 7 000 подсчёт по всей базе (49 карточек из 54, 8 006 символов)
# уходил к агенту с grep и терялся (search_0005). 12 000 символов — ~3 000 токенов входа, ~300 по формуле.
MAX_CONTEXT_CHARS = 12000
MAX_MENTIONS = 3  # карточек на одно имя без своей карточки
MAX_VALUE_SEEDS = 12  # карточек с значением из вопроса, когда имён в нём нет
# Карточки могут быть не только в Markdown: формулировки и раскладки финала не как в примерах (ответ 10.11).
# Заголовка «#» у json/yaml/csv нет — имя карточки берётся из имени файла.
_TEXT_SUFFIXES = {".md", ".txt", ".markdown", ".json", ".yaml", ".yml", ".csv", ".tsv", ".html", ".htm", ".xml",
                  ".rst", ".org"}
_REVERSE_Q = re.compile(r"\bhow many\b|\bwhich (?:person|people|employee|engineer|one)\b|\bwho (?:currently )?"
                        r"(?:works?|worked|is employed|was born|has|have)\b|\bwhose\b|\bname the person\b|"
                        r"\bwho is the\b|\b(?:oldest|youngest|earliest|latest|most|fewest|largest|smallest)\b|"
                        # «Several people are mentored by X. Exactly one of them works at Y» (hold-c7): обратная
                        # ссылка без «who/how many»; «Count the people ...».
                        r"\b(?:mentored|managed|supervised|coached|advised|employed|hired|founded|led|run|owned|"
                        r"headed) by\b|\b(?:one|which|who|each|all|none|any) of (?:them|those|these)\b|"
                        r"\bcount (?:the|all|of)\b|\bnumber of\b", re.I)
_MENTOR_HOP = re.compile(r"\bmentor|\bmentee|\bmanager|\bboss\b", re.I)
_EMPLOYER_HOP = re.compile(r"\bemploy|\bworks?\b|\bcompany (?:where|that)", re.I)
_TITLE_PREFIX = re.compile(r"^(?:notes? on|about|profile(?: of)?|bio(?:graphy)? of|company profile|person|company|"
                           r"card)\b\W*", re.I)
_DROP_SENTENCES = re.compile(r"(?:The directory [^.]*\.|Use the file tools[^.]*\.|Work only inside the current "
                             r"directory\..*$)", re.S)
_ALIAS = re.compile(r"(?:also known as|a\.k\.a\.?|aka|nickname|goes by|known as)\W{0,6}"
                    r"([A-Z][\w'-]+(?: [A-Z][\w'-]+){0,3})", re.I)
_NEE = re.compile(r"\b(?:n[ée]e|maiden name)\W{0,6}([A-Z][\w'-]+)", re.I)
_VALUE = re.compile(r"\b(?:[A-Z][a-z]{2,}|\d{4})\b")
_NAME = re.compile(r"\b[A-Z][a-z-]+(?: [A-Z][a-z-]+)+\b")


@dataclass
class Note:
    rel: str
    title: str
    text: str
    names: list[str] = field(default_factory=list)


# Поле с именем в YAML front matter, YAML, JSON, «Name: …» (карточки по id: «p063.md» с «name: Yusuf Okafor», hold-c8).
# Только name: «title» в карточке человека — должность, «company» — работодатель.
_NAME_FIELD = re.compile(r"""^\s*["']?(?:name|full[ _]name)["']?\s*[:=]\s*["']?"""
                         r"""([^"'\n,{}\[\]]{2,80}?)["']?\s*,?\s*$""", re.I | re.M)
_ID_FIELD = re.compile(r"""^\s*["']?(?:id|key|slug)["']?\s*[:=]\s*["']?([\w.-]{2,40})["']?\s*,?\s*$""", re.I | re.M)


def _json_fields(path: Path, text: str) -> dict:
    """Карточка JSON не обязана содержать переносы строк. Массивы/таблицы здесь не разбираем."""
    if path.suffix.lower() != ".json":
        return {}
    try:
        record = json.loads(text)
    except (ValueError, RecursionError):
        return {}
    return {re.sub(r"[ _]", "", k.lower()): v for k, v in record.items()} if isinstance(record, dict) else {}


def _title(path: Path, text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            if line.startswith("#"):
                t = line.lstrip("#").strip()
                t = t.rsplit(":", 1)[-1].strip() or t  # «Profile: Clara Varga» -> «Clara Varga»
                return _TITLE_PREFIX.sub("", t).strip() or t  # «Notes on Lars Tanaka» -> «Lars Tanaka»
            break
    fields = _json_fields(path, text)
    for key in ("name", "fullname"):
        value = fields.get(key)
        if isinstance(value, str) and 2 <= len(value.strip()) <= 80:
            return value.strip()
    m = _NAME_FIELD.search("\n".join(text.splitlines()[:25]))
    if m and m.group(1).strip():
        return m.group(1).strip()
    stem = re.sub(r"^(?:person|company|people|org|entry)[_-]", "", path.stem)
    return re.sub(r"[_-]+", " ", stem).strip().title()


def _ids(path: Path, text: str) -> list[str]:
    """Идентификаторы карточки, по которым на неё ссылаются другие («mentor: p020», «employer: org-24»): поле id
    и имя файла, если в нём есть цифры. Раньше «org-24» обрезалось до «24» и короче 4 символов ни с чем не
    связывалось (hold-c8)."""
    out = [m.group(1) for m in _ID_FIELD.finditer("\n".join(text.splitlines()[:25]))]
    fields = _json_fields(path, text)
    out += [v for k in ("id", "key", "slug") if isinstance(v := fields.get(k), str)
            and re.fullmatch(r"[\w.-]{2,40}", v)]
    if re.search(r"\d", path.stem):
        out.append(path.stem)
    # С буквой: имя файла «2024.md» иначе совпало бы с любым годом в тексте.
    return [x for x in dict.fromkeys(out) if len(x) >= 3 and re.search(r"[A-Za-z]", x)]


def _aliases(title: str, text: str) -> list[str]:
    out = [m.strip() for m in _ALIAS.findall(text)]
    first = title.split()[0] if title.split() else ""
    out += [f"{first} {m}" for m in _NEE.findall(text) if first]
    return [a for a in dict.fromkeys(out) if a.lower() != title.lower() and len(a) >= 4]


_INDEX_STEMS = {"readme", "index", "toc", "contents", "about"}


def _is_entity_title(path: Path, title: str) -> bool:
    """Заголовок — имя сущности, а не описание базы: «# Knowledge base» в README.md совпадал со словами вопроса
    «described in the knowledge base», и README становился опорной карточкой вместо поиска по значению (hold-c7,
    search_0047). Имя людей и компаний кончается словом с заглавной («Bank of Lyon» — да, «Knowledge base» — нет)."""
    if path.stem.lower() in _INDEX_STEMS:
        return False
    words = re.sub(r"\([^)]*\)", " ", title).split()  # «Hana Silva (engineer)»
    return bool(words) and (words[-1][:1].isupper() or words[-1][:1].isdigit())


def load_notes(workdir: Path) -> list[Note]:
    notes = []
    for p in sorted(workdir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _TEXT_SUFFIXES or p.stat().st_size > MAX_NOTE_BYTES:
            continue
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            title = " ".join(_title(p, text).split())
            names = [title, *_aliases(title, text)] if _is_entity_title(p, title) else []
            names += [i for i in _ids(p, text) if i.lower() not in {n.lower() for n in names}] \
                if p.stem.lower() not in _INDEX_STEMS else []
            notes.append(Note(p.relative_to(workdir).as_posix(), title, text, names))
    return notes


def _name_re(name: str) -> re.Pattern:
    # Пробелы — любые: заголовок «# Marta  Ruiz» с двумя пробелами не находился по «Marta Ruiz» (сторонний набор).
    return re.compile(r"(?<![\w])" + r"\s+".join(re.escape(w) for w in name.split()) + r"(?![\w])", re.I)


def question_text(instruction: str) -> str:
    return " ".join(_DROP_SENTENCES.sub(" ", instruction).split())


def _depth(question: str) -> int:
    """Число переходов: каждый «mentor» — шаг, работодатель — ещё один. Не меньше двух."""
    hops = len(_MENTOR_HOP.findall(question)) + (1 if _EMPLOYER_HOP.search(question) else 0)
    return max(2, min(4, hops))


def select_notes(question: str, notes: list[Note], depth: int | None = None) -> list[Note] | None:
    """Карточки из вопроса и по ссылкам. None — в вопросе нет имён из базы."""
    depth = depth or _depth(question)
    pats = {i: [_name_re(nm) for nm in n.names if len(nm) >= 4] for i, n in enumerate(notes)}
    known = {nm.lower() for n in notes for nm in n.names}

    def mentions(text: str) -> list[int]:
        return [j for j, ps in pats.items() if any(p.search(text) for p in ps)]

    seeds = mentions(question)
    by_value = False
    if not seeds:
        # Своей карточки у человека из вопроса нет, но имя есть в чужих: их и показываем.
        names = [nm for nm in _NAME.findall(question) if nm.split()[0] not in _NOT_NAMES]
        seeds = [i for i, n in enumerate(notes) for nm in names if _name_re(nm).search(n.text)][:MAX_MENTIONS * 2]
        if not seeds and names:
            # Полного имени нет нигде: показываем карточки с тем же именем или фамилией, модель увидит, что
            # нужного человека нет. Это один дешёвый вызов вместо агента с grep.
            parts = {w.lower() for nm in names for w in (nm.split()[0], nm.split()[-1])}
            near = [i for i, n in enumerate(notes) if parts & {w.lower() for w in n.title.split()}][:MAX_MENTIONS * 2]
            return [notes[i] for i in near] or None
        if not seeds:
            # Имён в вопросе нет, есть значение («компании со штаб-квартирой в Lyon», «родился в 1987»):
            # опора — карточки, где оно встречается; для подсчёта и обратного поиска ниже добавятся ссылающиеся.
            # Слова с заглавной (страна, город) — опора; число («born before 1975») — порог, а не опора: оно
            # заполняло лимит карточками людей этого года (hold-c4). Числа берём, только если слов нет.
            values = [w for w in _VALUE.findall(question) if w not in _NOT_NAMES]
            words = [v for v in values if not v.isdigit()]
            for group in (words, [v for v in values if v.isdigit()]):
                seeds = [i for i, n in enumerate(notes) for v in group if _name_re(v).search(n.text)][:MAX_VALUE_SEEDS]
                if seeds:
                    break
            if not seeds:
                return None
            by_value = True
    seen = list(dict.fromkeys(seeds))
    frontier = list(seen)
    for _ in range(depth):
        nxt = []
        for i in frontier:
            linked = [j for j in mentions(notes[i].text) if j != i]
            # Имя без своей карточки (строка в таблице команды и т. п.): карточки, где оно упомянуто.
            for nm in _NAME.findall(notes[i].text):
                if nm.lower() not in known and nm.split()[0] not in _NOT_NAMES:
                    pat = _name_re(nm)
                    linked += [j for j, n in enumerate(notes) if j != i and pat.search(n.text)][:MAX_MENTIONS]
            for j in linked:
                if j not in seen:
                    seen.append(j)
                    nxt.append(j)
        frontier = nxt
    if by_value or _REVERSE_Q.search(question):
        # Обратный обход: карточки, ссылающиеся на опорные, сначала на несколько сразу. Второй уровень — от
        # карточек не о людях («страна -> город -> компания -> люди»): «кто в компаниях из Германии» (hold-c4).
        level = list(seeds)
        for step in range(2):
            lp = [p for i in level for p in pats[i]]
            scored = [(sum(bool(p.search(n.text)) for p in lp), j) for j, n in enumerate(notes) if j not in seen]
            nxt = [j for s, j in sorted(scored, key=lambda x: -x[0]) if s]
            seen += nxt
            level = [j for j in nxt if not _is_person(notes[j])]
            if not level:
                break
    return [notes[i] for i in seen]


_PERSON = re.compile(r"\b(?:born|birth|year of birth|mentor|age)\b", re.I)


def _is_person(note: Note) -> bool:
    return bool(_PERSON.search(note.text))


def build_context(question: str, workdir: Path) -> str | None:
    notes = load_notes(workdir)
    sel = select_notes(question, notes) if notes else None
    ctx = "\n\n".join(f"[{n.rel}]\n{n.text}" for n in sel) if sel else ""
    if len(ctx) > MAX_CONTEXT_CHARS:
        return None
    titles = [n.title for n in notes if n.names]
    grep = keyword_snippets(question, workdir, {n.rel for n in sel or []}, MAX_CONTEXT_CHARS - len(ctx), titles)
    if grep:
        ctx = (ctx + "\n\n" if ctx else "") + "Other files mentioning words of the question (matching lines):\n" + grep
    return ctx or None


# Поиск по словам вопроса во всех файлах базы, включая скрытые и форматы без карточек (.log, .ics, .vcf, .ini,
# .properties, .env): в стороннем наборе ответы лежали в архивных отчётах, заметках встреч, календаре, логах и
# скрытом .vault, а контекст собирался только по карточкам с именами — 7 задач «нет ответа» (−25 баллов).
MAX_GREP_FILES = 4
MAX_GREP_FILE_CHARS = 400
MAX_NAMED_FILE_CHARS = 3000
MAX_GREP_BYTES = 200_000
_TERM = re.compile(r"[\"«]([^\"»\n]{2,40})[\"»]|'([^'\n]{2,40})'(?!\w)|\b[A-Z][\w-]*(?:[ \t]+[A-Z][\w-]*)*|"
                   r"\b[\w-]*\d[\w-]*|\b[A-Z][A-Z0-9_]{2,}\b")
_COMMON = {"Under", "Where", "Whose", "Whom", "Is", "Are", "Was", "Were", "Does", "Do", "Did", "On", "At", "By", "From",
           "To", "Of", "And", "Or", "Name", "List", "Find", "Give", "Show", "Tell", "Note", "Notes", "Project", "Company",
           "Person", "File", "Files", "Directory", "Knowledge", "Base", "Partner", "Team", "Summit", "Report"}


def question_terms(question: str, titles: list[str] = ()) -> list[str]:
    """Отличительные слова вопроса: имена и названия с заглавной, идентификаторы с цифрами («S-331», даты),
    КОНСТАНТЫ, слова в кавычках. Из названия из нескольких слов — и отдельные слова (в архиве «Chimera» без
    «Project»), но не те, что входят в имя другой карточки: «Muller» из «Hiro Muller» тянул однофамильцев, и на
    невыполнимой задаче (человека нет) модель отвечала по чужой карточке (регресс 12.11)."""
    out = []
    for m in _TERM.finditer(question):
        t = (m.group(1) or m.group(2) or m.group(0)).strip()
        words = t.split()
        if not words:
            continue
        if len(words) == 1 and (t in _NOT_NAMES or t in _COMMON or (len(t) < 3 and not t.isdigit())):
            continue
        if re.fullmatch(r"\d{1,2}", t):
            continue
        # Слово с заглавной в начале предложения («Start at ...», «Stop as soon as ...») — не название.
        if len(words) == 1 and not m.group(1) and not m.group(2) and t[:1].isupper() and not t.isupper() \
                and not re.search(r"\d", t) and question[:m.start()].rstrip()[-1:] in ("", ".", "?", "!", ":"):
            continue
        out.append(t)
        if len(words) > 1:
            for w in words:
                if len(w) < 4 or w in _NOT_NAMES or w in _COMMON:
                    continue
                other = [x for x in titles if x.lower() != t.lower() and w.lower() in x.lower().split()]
                if not other:
                    out.append(w)
    return list(dict.fromkeys(out))


def _text_file(p: Path) -> str | None:
    """Текст файла для поиска по словам. Сжатый .gz распаковываем, SQLite выводим строками «таблица: поле=значение»
    (в стороннем наборе ответы были в archive/notes.txt.gz и registry.db). Прочие двоичные пропускаем."""
    try:
        if p.stat().st_size > MAX_GREP_BYTES:
            return None
        raw = p.read_bytes()
    except OSError:
        return None
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)[:MAX_GREP_BYTES]
        except (OSError, EOFError, zlib.error):
            return None
    if raw[:16] == b"SQLite format 3\x00":
        return _sqlite_text(p)
    if b"\0" in raw[:2048]:
        return None
    return raw.decode("utf-8", errors="replace")


def _sqlite_text(p: Path, max_rows: int = 500) -> str | None:
    try:
        con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error:
        return None
    lines = []
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            cur = con.execute(f'SELECT * FROM "{t}" LIMIT {max_rows}')
            cols = [d[0] for d in cur.description]
            lines += [f"{t}: " + ", ".join(f"{c}={v}" for c, v in zip(cols, row)) for row in cur]
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return "\n".join(lines) or None


_FILE_MENTION = re.compile(r"[\w./-]+\.(?:ini|properties|ya?ml|json|jsonl|csv|tsv|psv|xml|txt|md|html?|log|ics|vcf|db|"
                           r"sqlite3?|gz|env|cfg|conf|toml)\b", re.I)


def keyword_snippets(question: str, workdir: Path, skip: set[str], room: int, titles: list[str] = ()) -> str:
    """Файлы, названные в вопросе («the build number in build.properties»), целиком, затем строки с отличительными
    словами вопроса (и по строке вокруг) из прочих файлов."""
    terms = question_terms(question, titles)
    named = {m.lower().lstrip("./") for m in _FILE_MENTION.findall(question)}
    if not (terms or named) or room < 200:
        return ""
    pats = [(t, _name_re(t)) for t in terms]
    found, whole = [], []
    for p in sorted(workdir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(workdir).as_posix()
        if rel in skip or p.name in ("task.json", "answer.txt") or "__pycache__" in p.parts:
            continue
        if named and (p.name.lower() in named or any(rel.lower().endswith(n) for n in named)):
            text = _text_file(p)
            if text:
                whole.append(f"[{rel}]\n" + (text[:MAX_NAMED_FILE_CHARS] + " ...[cut]"
                                              if len(text) > MAX_NAMED_FILE_CHARS else text))
                continue
        text = _text_file(p)
        if not text:
            continue
        lines = text.splitlines()
        hit_terms, hit_lines = set(), []
        for i, ln in enumerate(lines):
            ts = {t for t, pat in pats if pat.search(ln)}
            if ts:
                hit_terms |= ts
                hit_lines.append(i)
        if hit_terms:
            # Сначала файлы с большим числом разных слов вопроса, затем с самым длинным совпавшим названием.
            found.append((len(hit_terms), max(len(t) for t in hit_terms), rel, lines, hit_lines))
    found.sort(key=lambda x: (-x[0], -x[1], x[2]))
    out, used = [], 0
    for block in whole:
        if used + len(block) > room:
            break
        out.append(block)
        used += len(block) + 2
    for _, _, rel, lines, hit_lines in found[:MAX_GREP_FILES]:
        keep = sorted({j for i in hit_lines for j in (i - 1, i, i + 1) if 0 <= j < len(lines)})
        body = "\n".join(lines[j].rstrip() for j in keep if lines[j].strip())
        if len(body) > MAX_GREP_FILE_CHARS:
            body = body[:MAX_GREP_FILE_CHARS] + " ...[cut]"
        block = f"[{rel}]\n{body}"
        if used + len(block) > room:
            break
        out.append(block)
        used += len(block) + 2
    return "\n\n".join(out)


SYSTEM_PROMPT_SEARCH = (
    "Answer the question from the notes only. Follow references between notes; names must match exactly, a "
    "similar name is a different person. First write one line with the "
    "chain of lookups you followed (e.g. X -> mentor Y -> employer Z -> headquarters W), then the answer alone "
    "on the last line. If the notes do not contain the answer, the last line is UNANSWERABLE."
)
# Подсчёт и обратный поиск: модель перечисляла найденных списком и не успевала дойти до ответа в 120 токенов
# (hold-c4, search-x1). Имена — одной строкой через запятую, лимит выше.
REVERSE_RULE = (" For a count or a 'who/which' question: first line - the names that match, comma-separated, "
                "instead of a chain (no lists, no explanations); last line - the answer alone (for a count, the "
                "number).")
_HOW_MANY = re.compile(r"\bhow many\b|\bcount (?:the|all|of)\b|\b(?:total )?number of (?:people|persons|employees|"
                       r"companies|mentees|staff|workers|members|engineers)\b", re.I)


_TRAILING_COUNT = re.compile(r"[,;:=]\s*(\d+)\s*\.?\s*$")


def count_from_names(question: str, answer: str | None) -> str | None:
    """Модель на «сколько» вернула список имён вместо числа: считаем имена сами. Имена и число одной строкой
    («Emil Fischer, Grace Berg, 6», hold-c4): берём число в конце."""
    if answer and _HOW_MANY.search(question) and not answer.strip().isdigit():
        m = _TRAILING_COUNT.search(answer)
        if m:
            return m.group(1)
        if "," in answer:
            items = [x for x in answer.split(",") if x.strip()]
            if all(_NAME.fullmatch(x.strip()) for x in items):
                return str(len(items))
    return answer


def is_reverse(question: str) -> bool:
    return bool(_REVERSE_Q.search(question))


# Сравнение с порогом: здесь модель ошибается чаще всего («родился в 1975» засчитан как «до 1975», hold-c4).
_COMPARE = re.compile(r"\b(?:before|after|older|younger|earlier|later|more than|less than|fewer than|at least|"
                      r"at most|over|under|between|exceed\w*)\b", re.I)
_YEAR_Q = re.compile(r"\b(?:what|which|in which) year\b|\bwhen\b", re.I)
_WHO = re.compile(r"\bwho\b|\bwhich (?:person|employee|engineer)\b|\bname the\b", re.I)


def doubt_flags(question: str, context: str, answer: str | None) -> list[str]:
    """Сигналы сомнения в ответе search без токенов (второй проход, budget_agent/second.py):
    - reverse: обратный поиск, подсчёт или сравнение — там промахи одного вызова;
    - unanswerable: модель не нашла ответа, хотя опорные карточки собраны;
    - type: тип ответа не тот (на «сколько» не число, на «в каком году» не год, на «кто» число);
    - not_in_context: ответа-слова нет в собранных карточках (имя придумано или искажено)."""
    flags = []
    if is_reverse(question) or _COMPARE.search(question):
        flags.append("reverse")
    if answer is None:
        return flags + ["unanswerable"]
    a = answer.strip()
    if _HOW_MANY.search(question):
        if not a.isdigit():
            flags.append("type")
    elif _YEAR_Q.search(question):
        if not re.fullmatch(r"\d{4}", a):
            flags.append("type")
    elif _WHO.search(question) and re.fullmatch(r"[\d\s.,-]+", a):
        flags.append("type")
    if not a.isdigit() and a.lower() not in context.lower():
        flags.append("not_in_context")
    return flags


def build_prompt(question: str, context: str) -> str:
    return f"Notes:\n{context}\n\nQuestion: {question}"


def final_line(text: str | None) -> str | None:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    if "->" in last:  # ответ дописан в конец цепочки: «... -> born 1977» -> «1977»
        last = re.sub(r"^(?:[a-z][a-z' ]*?\s)(?=[A-Z0-9])", "", last.rsplit("->", 1)[1].strip())
    return last.strip()


# Слова, с которых начинаются предложения условия: «In Which», «The Directory» и т. п. не имена.
_NOT_NAMES = {"In", "The", "Use", "Work", "When", "What", "Which", "Who", "How", "Give", "Run", "Fix", "Compute",
              "Then", "A", "An", "If", "For", "Write", "Answer", "Born", "Before", "After", "Since", "Profile"}
