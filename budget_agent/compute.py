"""math и data_analysis в один вызов: модель пишет только код, харнесс его запускает.

Через агента с инструментом sh модель отвечала ```bash python3 -c "print(...)" | tee answer.txt```: из ~33 токенов
выхода ~19 уходило на обёртку, а выход в 10 раз дороже входа. Плюс системный промпт агента и подсказка на входе.
Здесь короткий системный промпт, модель присылает выражение или короткую программу без огораживания, харнесс
дописывает печать результата, запускает код и берёт последнюю строку вывода. Код упал — один повтор с текстом
ошибки; снова не вышло — задачу берёт обычный агент.
"""
from __future__ import annotations

import ast
import re

PRELUDE = ("import csv, json, math, re, statistics, collections, itertools, datetime, sqlite3, glob, os, gzip\n"
           "import xml.etree.ElementTree as ET\n"
           "from collections import Counter, defaultdict\n")
_TOOL_SENTENCE = re.compile(r"\s*(?:Compute it with the bash tool[^.]*\.|Use bash with python3[^.]*\.|"
                            r"Use the file tools[^.]*\.)", re.I)
# Только саму фразу: раньше срезалось и всё, что после неё, — а там может стоять условие («Round to 2 decimal
# places»). Фразу про answer.txt после неё убирает _drop_answer_file_sentences.
_BOILERPLATE = re.compile(r"\s*Work only inside the current directory\.")
_FENCE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.S)

SYSTEM_PROMPT_COMPUTE = (
    "Reply with only Python code, no fences or comments: one expression if the answer takes one or two "
    "operations, otherwise a short program with one statement per step (s=1752 / s//=2 / s*=2 on separate "
    "lines). Name variables after what they count and print exactly the quantity the question asks for. "
    "When the cases can be enumerated (under about 10**7), count them with a direct loop instead of a formula. "
    "If the unknown is defined by conditions (ages, counts, amounts), loop over candidate values and check "
    "every condition; never state a number you did not compute. "
    "csv, json, math, re, statistics and collections are imported. Use // where "
    "the text says rounded down, otherwise round() a float result, never int(). Print the answer; never write files. "
    "If the data needed is missing or contradictory, reply only UNANSWERABLE."
)
# Для data пошаговое правило вредно: модель писала программу с циклом на 65–79 токенов вместо одного выражения
# на 35 (gen-s1, 09.11).
SYSTEM_PROMPT_DATA = (
    "Reply with only Python code that prints the answer, no fences or comments. Keep the logic straightforward "
    "but the style compact: short names, no blank lines, no with-blocks. csv, json, math, re, statistics, "
    "collections, sqlite3, glob, os, gzip and xml.etree.ElementTree as ET are imported; import anything else you "
    "need (openpyxl, yaml). Convert csv fields to int or float before comparing numbers. Files may differ in "
    "layout or value formats (column order, date formats, case): handle every variant shown, never skip rows "
    "with a bare except. "
    "Print the answer; never write files. If the files cannot answer the question, reply only UNANSWERABLE."
)
DATA_HINT = "Files are in the current directory; mind the delimiter and header shown below."


def system_prompt(task_type: str) -> str:
    return SYSTEM_PROMPT_DATA if task_type == "data_analysis" else SYSTEM_PROMPT_COMPUTE



_SENTENCE_SPLIT = re.compile(r"((?<=[.!?])\s+)")
_PAREN = re.compile(r"\([^)]*\)")
# Слова, из которых состоит фраза «запиши ответ в файл» без единого слова о том, ЧТО считать. Список нарочно
# узкий: слов line, word, digit, unit, text здесь нет — «Write the line number to answer.txt» должно остаться.
# Лишняя оставленная фраза стоит нескольких токенов, вырезанный вопрос — задачи.
_GENERIC_WORDS = frozenset("""
a an the this that it its itself your you are is be been done finished when once after then finally now please and or
but finish by write writing written save saving saved put putting store storing stored record recording place placing
output print report give provide enter leave create make sure
only just alone exactly single one sole nothing else other extra any no without with except
final resulting computed numeric bare plain raw answer result value number integer
to in into inside as on of for at from there
file named called current directory folder working checked graded read
example e g eg bash echo using use
must should will shall can contain contains containing hold holds
explanation explanations comment comments commentary
""".split())
# «with no words or units», «no extra text»: слова word, unit, text допустимы только после отрицания.
_FORMAT_NOUN = r"(?:words?|units?|text|characters?|punctuation|quotes?|whitespace|spaces?|newlines?|labels?|formatting)"
_NEGATED_FORMAT = re.compile(
    r"\b(?:no|without|not)\s+(?:(?:any|other|extra|additional|surrounding|leading|trailing)\s+)*" + _FORMAT_NOUN
    + r"(?:\s*(?:,|or|and|nor)\s*" + _FORMAT_NOUN + r")*"
    + r"|\bon\s+(?:(?:a|one|its|own|single|separate)\s+)+line\b", re.I)
_ANSWER_FILE = re.compile(r"answer\.txt|\bfile\b", re.I)


def _is_generic_file_sentence(sentence: str) -> bool:
    s = _NEGATED_FORMAT.sub(" ", _PAREN.sub(" ", sentence).replace("answer.txt", " "))
    return all(w in _GENERIC_WORDS for w in re.findall(r"[A-Za-z]+", s.lower()))


def _drop_answer_file_sentences(text: str) -> str:
    """Фраза про answer.txt («Finish by writing the answer alone into a file named answer.txt», hold-c7) толкала
    модель писать файл самой: `open('answer.txt','w').write(str(n))` вместо печати. Убираем предложение про файл
    ответа, только если в нём нет ни слова о том, что считать: «Write the name of the region with the highest
    revenue to answer.txt» — это сам вопрос, и раньше он вырезался, потому что в нём нет how/what/total (проба
    13.11: модель получала одно описание файла). Остальной текст не трогаем, пробелы сохраняются."""
    parts = _SENTENCE_SPLIT.split(text)  # предложение, разделитель, предложение, ...
    out, dropped, kept = [], 0, 0
    for i in range(0, len(parts), 2):
        s = parts[i]
        if not s.strip():
            continue
        if _ANSWER_FILE.search(s) and _is_generic_file_sentence(s):
            dropped += 1
            continue
        kept += 1
        out.append(s + (parts[i + 1] if i + 1 < len(parts) else ""))
    return "".join(out).strip() if kept and dropped else text


def question_text(instruction: str) -> str:
    return _drop_answer_file_sentences(_BOILERPLATE.sub("", _TOOL_SENTENCE.sub("", instruction)).strip())


def strip_code(text: str | None) -> str:
    t = (text or "").strip()
    m = _FENCE.search(t)
    return (m.group(1) if m else t.strip("`")).strip()


def _writes_file(code: str) -> bool:
    """Программа сама пишет ответ в файл (`f.write(str(n))`, в том числе внутри with): печатать нечего, ответ харнесс
    берёт из файла (solver.solve_compute_direct). Обёртка print(f.write(...)) печатала число записанных символов:
    2 вместо 26 (hold-c7), а программу с with харнесс отбрасывал как «не Python»."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("write", "write_text")
               for n in ast.walk(tree))


def to_program(code: str) -> str | None:
    """Выражение -> print(выражение). Программа без print: печатаем последнее выражение или переменную."""
    if _writes_file(code):
        return PRELUDE + code + "\n"
    try:
        expr = ast.parse(code, mode="eval").body
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "print":
            return PRELUDE + code + "\n"
        return PRELUDE + f"print({code})\n"
    except SyntaxError:
        pass
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if not tree.body:
        return None
    has_print = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
                    for n in ast.walk(tree))
    if has_print:
        return PRELUDE + code + "\n"
    last = tree.body[-1]
    lines = code.splitlines()
    if isinstance(last, ast.Expr):
        expr = "\n".join(lines[last.lineno - 1:last.end_lineno])
        return PRELUDE + "\n".join(lines[:last.lineno - 1] + [f"print({expr.strip()})"]) + "\n"
    target = None
    if isinstance(last, ast.Assign) and len(last.targets) == 1 and isinstance(last.targets[0], ast.Name):
        target = last.targets[0].id
    elif isinstance(last, (ast.AugAssign, ast.AnnAssign)) and isinstance(last.target, ast.Name):
        target = last.target.id
    return PRELUDE + code + (f"\nprint({target})\n" if target else "\n") if target else None


def last_output_line(out: str) -> str | None:
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def is_literal(code: str) -> bool:
    """Код без вычисления: только присваивания и печать констант («s = 112; print(s)», hold-c5). Модель посчитала
    в уме и часто ошиблась. Без токенов, по AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    ops = (ast.BinOp, ast.AugAssign, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.For, ast.While, ast.comprehension,
           ast.Subscript, ast.Attribute, ast.IfExp, ast.If)
    for node in ast.walk(tree):
        if isinstance(node, ops):
            if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
                continue  # -5
            return False
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id == "print"):
            return False
    return bool(tree.body)


_COUNT_Q = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b", re.I)
_INT = re.compile(r"-?\d+")


def doubt_flags(question: str, out: str, answer: str) -> list[str]:
    """Сигналы сомнения в ответе math/data без токенов (второй проход, budget_agent/second.py):
    - multi: программа напечатала несколько строк, а ответом взята последняя;
    - type: вопрос «сколько», а ответ не целое неотрицательное число."""
    flags = []
    if len([ln for ln in out.splitlines() if ln.strip()]) > 1:
        flags.append("multi")
    if _COUNT_Q.search(question) and not (_INT.fullmatch(answer.strip()) and not answer.strip().startswith("-")):
        flags.append("type")
    return flags


_KEY_ERROR = re.compile(r"KeyError: ['\"]([^'\"]+)['\"]")
_NO_FILE = re.compile(r"(?:FileNotFoundError|No such file or directory)[^'\"]*['\"]([^'\"]+)['\"]")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _headers(workdir) -> set[str] | None:
    """Названия колонок таблиц задачи (первая строка csv/tsv с любым разделителем). None — в задаче есть данные
    не только в таблицах (json, логи): там поле в условии и в данных часто называется по-разному
    («quantity» и «qty», hold-c4), и вывод «колонки нет» делать нельзя."""
    out: set[str] = set()
    for p in workdir.rglob("*"):
        if not p.is_file() or p.stat().st_size > 50_000_000:
            continue
        suf = p.suffix.lower()
        if suf in (".json", ".jsonl", ".log", ".xml", ".yaml", ".yml"):
            return None
        try:
            if suf in (".csv", ".tsv"):
                with p.open(encoding="utf-8", errors="replace") as f:
                    first = f.readline()
                out |= {_norm(x) for x in re.split(r"[,;\t|]", first) if x.strip()}
        except OSError:
            continue
    return out or None


def missing_data_reason(err: str, workdir, question: str) -> str | None:
    """Код модели упал, потому что колонки или файла, о которых спрашивает условие, нет в данных: это
    невыполнимая задача, а не ошибка кода. Считаем так, только если имя есть в условии (выдуманное моделью имя —
    это ошибка кода, пусть будет повтор) и его нет в шапках таблиц (Units вместо units — тоже не считаем)."""
    q = _norm(question)
    m = _KEY_ERROR.search(err)
    headers = _headers(workdir) if m else None
    if m and headers and len(_norm(m.group(1))) >= 3 and _norm(m.group(1)) in q and _norm(m.group(1)) not in headers:
        return f"column {m.group(1)!r} is not in the data"
    m = _NO_FILE.search(err)
    if m:
        name = re.split(r"[\\/]", m.group(1))[-1].lower()
        if name and _norm(name) in q and not any(p.name.lower() == name for p in workdir.rglob("*")):
            return f"file {m.group(1)!r} does not exist"
    return None
