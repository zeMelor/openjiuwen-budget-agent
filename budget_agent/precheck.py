"""Проверки «задача невыполнима» до вызова LLM: ноль токенов.

Два уровня. Жёсткий (infeasible_reason) — задача не запускается вовсе:
- видимые тесты требуют от одного и того же вызова разных результатов в разных тестах: пройти их нельзя;
- тест импортирует модуль, которого нет ни в файлах задачи, ни в окружении.
Мягкий (doubt_reason) — задача уходит в конец очереди первого прохода, но решается:
- значение в одинарных кавычках из вопроса нигде не встречается в файлах задачи;
- имя «Имя Фамилия» из вопроса нигде не встречается в файлах задачи (ни в тексте, ни в именах файлов);
- условие ссылается на файл с данными (.csv, .json, …), которого нет в каталоге задачи.
Если проверка не уверена, она молчит: ложное срабатывание стоит баллов, пропуск только токенов.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from .tasks import Task

MAX_SCAN_BYTES = 5_000_000
# Кавычка — только если рядом с ней нет буквы: иначе апострофы двух притяжательных («Boris Kaya's mentor …
# Boris Kaya's») давали «значение» «s mentor was Boris Kaya» и ложное «невыполнимо» (hold-c5).
_QUOTED = re.compile(r"(?<!\w)'([A-Za-z][\w\- ]{1,40})'(?!\w)")
_NAME = re.compile(r"\b([A-Z][a-z]+(?:[ -][A-Z][a-z]+)+)\b")
# Слова, с которых часто начинаются предложения условия: «In Which», «The Directory» и т. п. не имена.
_NOT_NAMES = {"In", "The", "Use", "Work", "When", "What", "Which", "Who", "How", "Give", "Run", "Fix", "Compute",
              "Then", "A", "An", "If", "For", "Write"}


def _workspace_text(task: Task) -> tuple[str, str] | None:
    """Весь текст файлов задачи (кроме task.json) и имена файлов, в нижнем регистре."""
    root = task.workdir
    if root == task.dir and not any(p.name != "task.json" for p in root.iterdir()):
        return None
    texts, names, size = [], [], 0
    for p in root.rglob("*"):
        if not p.is_file() or p.name in ("task.json", "answer.txt"):
            continue
        names.append(p.stem.replace("_", " ").replace("-", " "))
        if size < MAX_SCAN_BYTES:
            try:
                t = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            size += len(t)
            texts.append(t)
    return "\n".join(texts).lower(), "\n".join(names).lower()


def _test_files(task: Task) -> list[Path]:
    from .coding import _py_files, is_test_file
    return [p for p in _py_files(task.workdir) if is_test_file(p, task.workdir)]


def _module_aliases(tree: ast.AST) -> set[str]:
    """Имена, под которыми тест импортировал модули: import mylib, import mylib as m."""
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return out


def _pure_call(expr: ast.AST, modules: set[str], local: frozenset[str] | set[str] = frozenset()) -> bool:
    """Вызов функции с литералами: f(2), mod.f('a'). Только такие ожидания сравниваем: у q.dequeue(),
    len(cart) или self.book.total() между проверками меняется состояние, и разные ответы законны
    (ложные срабатывания на coding-x1 и hold-c3). Нужен хотя бы один аргумент: f() бывает со счётчиком внутри.
    Имя, которому тест сам что-то присваивает (`add = make_adder(2)`), — не функция модуля: в разных тестах под
    ним разные объекты, и add(1) законно даёт 3 и 6 (проба 13.11)."""
    if not isinstance(expr, ast.Call) or not (expr.args or expr.keywords):
        return False
    f = expr.func
    if isinstance(f, ast.Attribute):
        if not (isinstance(f.value, ast.Name) and f.value.id in modules):
            return False
    elif not isinstance(f, ast.Name) or f.id in local:
        return False
    args = [*expr.args, *(k.value for k in expr.keywords)]
    try:
        for a in args:
            ast.literal_eval(a)
    except (ValueError, SyntaxError, TypeError):
        return False
    return True


def _calls_with_scope(tree: ast.AST) -> list[tuple[str, ast.Call]]:
    """Все вызовы с именем функции или метода, внутри которого они стоят («Класс.метод»)."""
    out: list[tuple[str, ast.Call]] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = f"{scope}.{child.name}" if scope else child.name
            if isinstance(child, ast.Call):
                out.append((scope, child))
            visit(child, inner)

    visit(tree, "")
    return out


def _same_value(a, b) -> bool:
    """Как сравнивает assertEqual: 6 и 6.0 — одно ожидание, а не противоречие (проба 13.11)."""
    try:
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


def _contradictory_tests(task: Task) -> str | None:
    """Один и тот же вызов с литералами должен дать разные результаты в разных тестах. Внутри одного теста
    разные ожидания не считаем: между вызовами законно меняется состояние (`next_id('a')` даёт 1, потом 2)."""
    seen: dict[str, list[tuple[object, str, str]]] = {}
    for test in _test_files(task):
        try:
            tree = ast.parse(test.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            return None
        modules = _module_aliases(tree)
        local = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        for scope, node in _calls_with_scope(tree):
            if not isinstance(node.func, ast.Attribute):
                continue
            name = node.func.attr
            if name == "assertEqual" and len(node.args) == 2:
                try:
                    val = ast.literal_eval(node.args[1])
                except (ValueError, SyntaxError, TypeError):
                    continue
                # Только вызовы с литералами: у len(cart) или self.book.total() между проверками меняется
                # состояние, и разные ожидания законны (на нашем наборе coding-x1 это дало 2 ложных срабатывания).
                if not _pure_call(node.args[0], modules, local):
                    continue
                key = "eq:" + ast.dump(node.args[0])
            elif name in ("assertTrue", "assertFalse") and node.args and _pure_call(node.args[0], modules, local):
                key, val = "bool:" + ast.dump(node.args[0]), name == "assertTrue"
            else:
                continue
            for prev, prev_test, prev_scope in seen.get(key, []):
                if (prev_test, prev_scope) != (test.name, scope) and not _same_value(prev, val):
                    where = test.name if prev_test == test.name else f"{prev_test} and {test.name}"
                    if key.startswith("bool:"):
                        return f"{where}: {ast.unparse(node.args[0])} must be both true and false"
                    return f"{where}: {ast.unparse(node.args[0])} expected both {prev!r} and {val!r}"
            seen.setdefault(key, []).append((val, test.name, scope))
    return None


def _missing_module(task: Task) -> str | None:
    """Тест импортирует модуль, которого нет ни в файлах задачи, ни в окружении: такой код не написать
    по условию. Относительные импорты и модули из каталогов задачи не трогаем."""
    import importlib.util
    root = task.workdir
    local = {p.stem for p in root.rglob("*.py")} | {p.name for p in root.rglob("*") if p.is_dir()}
    for test in _test_files(task):
        try:
            tree = ast.parse(test.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            return None
        for n in ast.walk(tree):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else                 [n.module] if isinstance(n, ast.ImportFrom) and n.module and not n.level else []
            for full in names:
                top = full.split(".")[0]
                if top in local:
                    continue
                try:
                    found = importlib.util.find_spec(top) is not None
                except (ImportError, ValueError):
                    found = False
                if not found:
                    return f"{test.name} imports {full}, which exists neither in the task files nor in the environment"
    return None


_DATA_FILE = re.compile(r"\b([\w.-]+\.(?:csv|tsv|json|jsonl|txt|log|xml|ya?ml|parquet|xlsx))\b", re.I)
_SERVICE_FILES = {"answer.txt", "task.json"}


def _missing_file(task: Task) -> str | None:
    """Условие ссылается на файл с данными, которого нет в каталоге задачи (на любой глубине)."""
    present = {p.name.lower() for p in task.workdir.rglob("*") if p.is_file()}
    for name in _DATA_FILE.findall(task.instruction):
        if name.lower() not in _SERVICE_FILES and name.lower() not in present:
            return f"{name} is referenced but does not exist"
    return None


def infeasible_reason(task: Task) -> str | None:
    """Причина, по которой задачу точно не решить, или None. Только проверки по самим тестам: противоречие в
    ожиданиях и модуль, которого нет. Такую задачу харнесс не запускает."""
    if task.task_type == "coding" or _test_files(task):
        return _contradictory_tests(task) or _missing_module(task)
    return None


def doubt_reason(task: Task) -> str | None:
    """Признак «возможно, невыполнима» по тексту условия: файл, значение в кавычках или имя из условия не
    встречаются в файлах задачи. Это догадка о формулировке, а не факт: на переписанных условиях она ошибалась
    («write 'DONE'», «Find Irina Smith's…», ответ 'yes' или 'no' — проба 13.11; до того 2 из 13 отсечений на
    новых наборах были выполнимыми задачами). Поэтому такую задачу харнесс не пропускает, а ставит в конец
    очереди первого прохода: при тесном бюджете она отпадёт первой, при свободном — решается как обычная."""
    if task.task_type != "coding":
        reason = _missing_file(task)
        if reason:
            return reason
    ws = _workspace_text(task)
    if ws is None:
        return None
    text, names = ws
    for value in _QUOTED.findall(task.instruction):
        v = value.strip().lower()
        if v and v not in text and v not in names and not _defined_here(value, task.instruction):
            return f"'{value}' does not occur in the task files"
    # Имена людей — только в search (база карточек). В data «Имя Фамилия» по форме — это и названия форматов,
    # городов, продуктов: «Common Log Format» дало ложное «невыполнимо» (hold-c6); в data правило ничего не ловило.
    for name in _NAME.findall(task.instruction) if task.task_type == "search" else []:
        if name.split()[0] in _NOT_NAMES:
            continue
        n = name.lower()
        if _word_in(n, text) or n.replace("-", " ") in names:
            continue
        # Имя и фамилия есть по отдельности — возможен псевдоним («Nadia Muller, née Smith»): пусть решает поиск.
        first, last = n.split()[0], n.split()[-1]
        if (_word_in(first, text) or first in names) and (_word_in(last, text) or last in names):
            continue
        return f"{name} does not occur in the task files"
    return None


def _defined_here(value: str, instruction: str) -> bool:
    """Термин в кавычках, который условие тут же определяет: «the 'greedy' companies: companies selling ...»
    (сторонний набор, sql_0032 — ложное «невыполнимо»). Это не значение из данных, искать его в файлах не нужно."""
    m = re.search(r"['\"]" + re.escape(value) + r"['\"]([^.]{0,30})", instruction)
    return bool(m and re.search(r":|\(|\bi\.e\.|\bmeans?\b|\bdefined\b|\bthat is\b|\bcalled\b", m.group(1)))


def _word_in(word: str, text: str) -> bool:
    """Целым словом: «omar moreau» не находится внутри «omar moreaux»."""
    return re.search(r"(?<![\w])" + re.escape(word) + r"(?![\w])", text) is not None
