"""Факты о литеральных таблицах unittest и атомарное применение предложенных исправлений ожиданий.

Не исполняет тесты и не решает, верен ли контракт. Только считает длины и запрещает менять входы/удалять случаи.
"""
from __future__ import annotations

import ast
import collections
import math


def _scalar(value):
    return type(value) in (str, int, bool, type(None)) or (type(value) is float and math.isfinite(value))


def _cases(code):
    tree = ast.parse(code)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    counts = collections.Counter(n.name for n in functions)
    result = {}
    for fn in functions:
        if counts[fn.name] != 1:
            continue
        assignments = [n for n in fn.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                       and isinstance(n.targets[0], ast.Name)]
        for loop in fn.body:
            if not (isinstance(loop, ast.For) and isinstance(loop.iter, ast.Name)
                    and isinstance(loop.target, (ast.Tuple, ast.List))
                    and all(isinstance(n, ast.Name) for n in loop.target.elts)):
                continue
            names = [n.id for n in loop.target.elts]
            if len(set(names)) != len(names):
                continue
            matches = [n.value for n in assignments if n.targets[0].id == loop.iter.id]
            expectations = {n.args[1].id for n in ast.walk(loop) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute) and n.func.attr == "assertEqual"
                            and len(n.args) >= 2 and isinstance(n.args[1], ast.Name)
                            and n.args[1].id in names}
            if len(matches) != 1 or len(expectations) != 1 or not isinstance(matches[0], (ast.List, ast.Tuple)):
                continue
            expected_index = names.index(next(iter(expectations)))
            for i, row in enumerate(matches[0].elts):
                if not isinstance(row, (ast.List, ast.Tuple)) or len(row.elts) != len(names):
                    continue
                try:
                    values = ast.literal_eval(row)
                except (ValueError, TypeError, SyntaxError):
                    continue
                if not all(_scalar(v) for v in values):
                    continue
                key = (fn.name, i)
                # Несколько таблиц в одном тесте с одинаковыми индексами неоднозначны.
                if key in result:
                    result[key] = None
                    continue
                result[key] = (row.elts[expected_index], values[expected_index],
                               {k: v for j, (k, v) in enumerate(zip(names, values)) if j != expected_index})
    return {k: v for k, v in result.items() if v is not None}


def literal_cases(code: str) -> list[dict]:
    out = []
    for (test, row), (_, expected, inputs) in _cases(code).items():
        out.append({"test": test, "row": row, "inputs": inputs, "expected": expected,
                    "input_string_lengths": {k: len(v) for k, v in inputs.items() if isinstance(v, str)},
                    "expected_length": len(expected) if isinstance(expected, str) else None})
    return out


def apply_corrections(code: str, review: dict) -> str:
    """При ошибке отклоняет весь ответ; меняются только ожидаемые литералы в существующих строках таблиц."""
    if not isinstance(review, dict) or set(review) != {"corrections"} or not isinstance(review["corrections"], list):
        raise ValueError("expected an object with a corrections array")
    cases = _cases(code)
    lines = code.splitlines(keepends=True)

    def offset(row, byte_col):
        return sum(map(len, lines[:row - 1])) + len(lines[row - 1].encode("utf-8")[:byte_col].decode("utf-8"))

    edits, seen = [], set()
    for item in review["corrections"]:
        if not isinstance(item, dict) or set(item) != {"test", "row", "expected", "reason"}:
            raise ValueError("invalid correction fields")
        if type(item["test"]) is not str or type(item["row"]) is not int:
            raise ValueError("invalid test or row")
        key = (item["test"], item["row"])
        if key not in cases or key in seen:
            raise ValueError("unknown, ambiguous or duplicate test row")
        seen.add(key)
        node, old, _ = cases[key]
        value = item["expected"]
        if not _scalar(value) or type(value) is not type(old):
            raise ValueError("expected value changed type")
        if value == old:
            raise ValueError("correction leaves the expected value unchanged")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError("correction has no reason")
        edits.append((offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset), repr(value)))
    for start, end, replacement in sorted(edits, reverse=True):
        code = code[:start] + replacement + code[end:]
    ast.parse(code)
    return code
