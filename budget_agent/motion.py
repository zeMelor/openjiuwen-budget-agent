"""Два равномерно движущихся объекта: модель извлекает данные, встречу считает харнесс.

Объект стоит в p[i] до s[i], затем движется со скоростью v[i]. Между моментами старта разность координат
линейна: её первый ноль находится точно через Fraction, без численного поиска и формулы от модели.
"""
from __future__ import annotations

import json
import math
import re
from fractions import Fraction

SYSTEM_PROMPT = (
    "Extract initial conditions only; do NOT calculate a meeting/catching time. Return only JSON with "
    "keys p, v, s, origin. Arrays p,v,s each have two numbers, one per moving object in order of mention. "
    "p: initial positions in km (if only separation is given, first object starts at 0, second at that "
    "separation; if both depart the same place both positions are 0). v: signed speeds in km/hour; "
    "positive toward increasing position, negative toward decreasing position. s: departure times in "
    "minutes from time zero; objects stay at their initial positions until departure. origin: index "
    "0 or 1 if the question measures elapsed time after that object departs, or -1 for time zero. "
    "Use the stated marker coordinates when provided. No explanation or calculations."
)
_RATE = re.compile(r"\bkm\s*/\s*h\b", re.I)
_EVENT = re.compile(r"\b(?:later|after|before|delay\w*|starts?|begins?|waits?)\b", re.I)
_TARGET = re.compile(r"\b(?:meet|catch|overtake|same (?:position|location))\b", re.I)
_TIME = re.compile(r"\b(?:how many|at what time|what time)\b[^?.]*\bminutes?\b", re.I)
_UNSUPPORTED = re.compile(
    r"\b(?:accelerat\w*|decelerat\w*|stops?|pauses?|rests?|turns?|revers\w*|returns?|"
    r"circular|laps?|wind|current|third|three|four|round\w*|nearest|decimal\w*|integer|"
    r"slows?|faster|slower|doubles?|triples?|halfway)\b|"
    r"\bchang\w*\s+(?:its\s+|their\s+|the\s+)?speed\b|"
    r"\bspeed\s+(?:changes?|increases?|decreases?)\b", re.I)


def applies(question: str) -> bool:
    return (not _UNSUPPORTED.search(question)
            and all(p.search(question) for p in (_RATE, _EVENT, _TARGET, _TIME)))


def solve(reply: str, question: str) -> str | None:
    """None — схема/данные непригодны или подходящей встречи нет; тогда нужен обычный решатель."""
    try:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
        d = json.loads(text)
        if not isinstance(d, dict) or set(d) != {"p", "v", "s", "origin"}:
            return None
        for key in ("p", "v", "s"):
            if (not isinstance(d[key], list) or len(d[key]) != 2
                    or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e9 for x in d[key])):
                return None
        if type(d["origin"]) is not int or d["origin"] not in (-1, 0, 1):
            return None
        p, v, s = ([Fraction(str(x)) for x in d[key]] for key in ("p", "v", "s"))
        if min(s) < 0:
            return None
        origin = Fraction(0) if d["origin"] == -1 else s[d["origin"]]
        # При догонянии с одного места совпадение позиций до начала погони ответом не является.
        begin = max(s) if re.search(r"\b(?:catch|overtake)\b", question, re.I) else Fraction(0)
        cuts = sorted({begin, *[x for x in s if x >= begin]})
        for i, t in enumerate(cuts):
            gap = (p[0] + v[0] * max(0, t - s[0]) / 60) - (p[1] + v[1] * max(0, t - s[1]) / 60)
            rate = ((v[0] if t >= s[0] else 0) - (v[1] if t >= s[1] else 0)) / 60
            if gap == 0:
                meet = t
            elif rate:
                meet = t - gap / rate
            else:
                continue
            if meet < t or (i + 1 < len(cuts) and meet > cuts[i + 1]):
                continue
            value = meet - origin
            if value < 0:
                return None
            return str(value.numerator) if value.denominator == 1 else format(float(value), ".12g")
    except (TypeError, ValueError, OverflowError):
        return None
    return None
