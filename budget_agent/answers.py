"""Нормализация ответа перед записью в answer.txt. Работает без LLM и без токенов.

Что чиним (по разбору прогонов):
- целое записано как float: «16537.0» -> «16537»;
- модель написала команду вместо ответа: «echo Lyon > answer.txt» -> «Lyon»;
- кавычки, обратные кавычки, «Answer: X», точка в конце одного слова, разделители тысяч.
"""
from __future__ import annotations

import re

# Модель пишет это слово, если по данным ответить нельзя (см. SYSTEM_PROMPT в solver.py).
NO_ANSWER_MARK = "UNANSWERABLE"
_NO_ANSWER = {"UNANSWERABLE", "IMPOSSIBLE", "CANNOT ANSWER", "CANNOT BE ANSWERED", "NOT ANSWERABLE"}

# Только строчными: команда echo пишется так, а «Echo Labs» — название, и раньше от него оставалось «Labs».
_ECHO = re.compile(r"""^(?:echo|printf)\s+(?:-[a-zA-Z]+\s+)?(.*?)\s*(?:>>?\s*\S+)?\s*$""")
_PREFIX = re.compile(r"^(?:final\s+answer|answer|result)\s*[:=]\s*", re.I)
# Подпись поля из карточки: «Mentor: Lars Novak», «- **Born:** 1977».
_FIELD = re.compile(r"^[-*\s]*\**([A-Z][A-Za-z]+(?: [A-Za-z]+){0,2}):\**\s+(.+)$")
# Строка вывода grep: «./kb/person_x.md:Mentor: Y» или «kb/person_x.md-Born: 1977».
_GREP_PREFIX = re.compile(r"^\.?/?[\w./-]+\.(?:md|txt|csv|tsv|json|log|py|ya?ml)[:-]")
_THOUSANDS = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_INT_FLOAT = re.compile(r"(-?\d+)\.0+")


def _unquote(s: str) -> str:
    s = s.strip().strip("`").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1].strip()
    return s


# Ответ из нескольких строк: «Write one company name per line», «Write exactly 3 rows», «print one per line»,
# «: 4 rows». Раньше всегда бралась последняя строка — на стороннем наборе это ~90 баллов (sql и coding).
# Ищем только в предложениях с глаголом вывода: «The file numbers.txt contains 20 integers, one per line» — это
# описание входа, а не ответа.
_MULTI_LINE = re.compile(r"\bper (?:line|row)\b|\bseparate (?:rows?|lines?)\b|\bon (?:separate|its own|their own|"
                         r"new) lines?\b|\b(?:exactly|write|print|output|show|list)\s+\d+\s+(?:rows|lines)\b|"
                         r":\s*\d+\s+(?:rows|lines)\b", re.I)
_OUTPUT_VERB = re.compile(r"\b(?:write|print|output|show|list|return|give|display|report)\b", re.I)


def wants_lines(instruction: str | None) -> bool:
    for sent in re.split(r"(?<=[.!?])\s+", instruction or ""):
        if _OUTPUT_VERB.search(sent) and _MULTI_LINE.search(sent):
            return True
    return False


# Слово-ответ для coding: «write ONLY the word DONE», «write exactly PASS to answer.txt» (сторонний набор:
# мы всегда писали DONE). Только заглавные: «write ONLY the final answer» не совпадает.
_DONE_WORD = re.compile(r"\bwrite\s+(?:(?:only|exactly|just)\s+)?(?:the\s+word\s+)?[\"'`]?((?-i:[A-Z][A-Z_]{1,19}))"
                        r"[\"'`]?\s+(?:to|into|in)\s+(?:the\s+file\s+)?(?:\./)?answer\.txt", re.I)
# Что писать, если задачу решить нельзя, когда условие это говорит: «If the requested value cannot be determined
# ..., write exactly UNSOLVABLE to answer.txt» (сторонний набор). Без такого условия невыполнимую оставляем без ответа.
_FAIL_CUE = (r"(?:cannot|can't|can not|could not|couldn't|unable to|impossible|not possible|inconsistent|"
             r"contradict\w*|insufficient|not enough|undetermin\w*|unsolvable|not solvable|no solution|ill-posed)")
_FAIL_WORD = re.compile(
    r"\b" + _FAIL_CUE + r"[^.]{0,140}?\bwrite\s+(?:(?:only|exactly|just)\s+)?(?:the\s+word\s+)?[\"'`]?"
    r"((?-i:[A-Z][A-Z_]{2,19}))\b"
    r"|\bwrite\s+(?:(?:only|exactly|just)\s+)?(?:the\s+word\s+)?[\"'`]?((?-i:[A-Z][A-Z_]{2,19}))[\"'`]?"
    r"(?:\s+(?:to|into|in)\s+(?:the\s+file\s+)?answer\.txt)?\s+(?:if|when)\b[^.]{0,100}?" + _FAIL_CUE, re.I)


_CAPS_WORD = re.compile(r"(?<![\w.])[A-Z][A-Z_]{1,19}(?!\w|\.\w)")  # не часть имени файла: README.md
# Заглавные слова-усилители и расширения: это не слово-ответ.
_NOT_A_WORD = {"ONLY", "THE", "NOT", "NO", "AND", "OR", "ALL", "ANY", "MUST", "NEVER", "ALWAYS", "DO", "JUST",
               "EXACTLY", "NOTE", "IMPORTANT", "TXT", "PY", "CSV", "JSON"}


def done_word(instruction: str | None) -> str:
    """Слово, которое условие просит записать после починки. Сначала прежнее правило («write … WORD to answer.txt»).
    Если оно не сработало — единственное слово заглавными в предложениях про answer.txt: «save the word FINISHED
    in answer.txt», «answer.txt must contain PASSED», «put OK into answer.txt» (проба 13.11: во всех трёх случаях
    писалось DONE). Слово для невыполнимой задачи (fail_word) не считается. Неоднозначно — DONE, как раньше."""
    text = instruction or ""
    skip = _NOT_A_WORD | {fail_word(text) or ""}
    for m in _DONE_WORD.finditer(text):
        if m.group(1) not in skip:
            return m.group(1)
    found = {w for s in re.split(r"(?<=[.!?;])\s+", text) if "answer.txt" in s
             for w in _CAPS_WORD.findall(s) if w not in skip}
    return found.pop() if len(found) == 1 else "DONE"


def fail_word(instruction: str | None) -> str | None:
    m = _FAIL_WORD.search(instruction or "")
    return (m.group(1) or m.group(2)) if m else None


# «Round to 2 decimals», «rounded to two decimal places», «one decimal point», «exactly two digits after the decimal
# point»: ответ — ровно столько знаков. Так размечены такие задачи в двух сторонних наборах
# («1102.50», «0.3010», «5.0»; «47701.00», «38.50») и на наших hold-c7 («6027.31»). Мы же
# писали «47701» и «38.5» — все 6 потерь data на одном из этих наборов (13.11). «decimal digits», «21 decimal numbers» — не то.
_N_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_N = r"(\d|one|two|three|four|five|six)"
_DECIMALS = re.compile(r"\bround(?:ed|ing)?\b[^.;?!]{0,60}?\b" + _N + r"\s+decimals?\b"
                       r"|\b" + _N + r"\s+decimal\s+places?\b"
                       r"|\b" + _N + r"\s+digits?\s+after\s+the\s+(?:decimal\s+)?point\b", re.I)
_PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def decimals_wanted(instruction: str | None) -> int | None:
    found = set()
    for m in _DECIMALS.finditer(instruction or ""):
        w = next(g for g in m.groups() if g)
        found.add(int(w) if w.isdigit() else _N_WORDS[w.lower()])
    return found.pop() if len(found) == 1 and 0 < min(found) <= 10 else None


def apply_decimals(instruction: str | None, answer: str | None) -> str | None:
    """Одно число при условии «N знаков после запятой» — ровно N знаков, половина вверх. Остальное как есть."""
    n = decimals_wanted(instruction)
    if n is None or answer is None or not _PLAIN_NUMBER.fullmatch(answer.strip()):
        return answer
    from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
    try:
        return str(Decimal(answer.strip()).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return answer


def _normalize_lines(lines: list[str]) -> str | None:
    """Многострочный ответ: строки как есть (без пустых и краевых пробелов), кроме обёрток вокруг всего ответа."""
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        lines = lines[1:-1]
    return "\n".join(lines) or None


def normalize_answer(raw: str | None, multiline: bool = False) -> str | None:
    """Возвращает чистый ответ, None если ответа нет. Признак «невыполнимо» возвращается как NO_ANSWER_MARK.
    multiline — условие просит ответ в несколько строк (wants_lines): тогда строки не отбрасываются."""
    if raw is None:
        return None
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    # Многострочный вывод grep (модель перенаправила его в answer.txt) — это не ответ.
    if len(lines) > 1 and sum(bool(_GREP_PREFIX.match(ln)) for ln in lines) >= 2:
        return None
    if multiline and len(lines) > 1:
        return _normalize_lines(lines)
    s = _GREP_PREFIX.sub("", lines[-1]).strip()
    m = _ECHO.match(s)
    if m:
        s = m.group(1)
    s = _PREFIX.sub("", _unquote(s))
    m = _FIELD.match(s)
    if m:
        s = m.group(2)
    s = _unquote(s.strip("*").strip())
    if not s:
        return None
    if s.upper().rstrip(".!") in _NO_ANSWER:
        return NO_ANSWER_MARK
    if _THOUSANDS.fullmatch(s):
        s = s.replace(",", "")
    m = _INT_FLOAT.fullmatch(s)
    if m:
        s = m.group(1)
    if " " not in s and s.endswith(".") and s.count(".") == 1 and len(s) > 1:
        s = s[:-1]
    return s or None
