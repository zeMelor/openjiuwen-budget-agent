"""Быстрый путь coding: тесты гоняет харнесс, модель отвечает только правками. Обычно один вызов.

Харнесс без токенов находит команду тестов (из условия или по раскладке) и модули с кодом. Модулей может
быть несколько, в пакете или в src/, тесты могут лежать в tests/. Он прогоняет тесты и кладёт в запрос условие,
код и упавшие проверки. Модель присылает только то, что меняет: функции, методы внутри своего класса,
константы. Харнесс подставляет правки по имени во все модули через AST и снова гоняет тесты.

Почему так (по разбору прогонов):
- выход в 10 раз дороже входа, поэтому модуль целиком не просим, а docstring в ответе разрешаем опустить:
  харнесс вернёт прежний docstring сам;
- скрытые тесты проверяют поведение из docstring, поэтому небольшие модули идут в запрос целиком,
  а условие задачи (там бывает «docstrings describe the intended behaviour») — всегда;
- модуль с синтаксической ошибкой по AST не разобрать: функции в нём заменяем по тексту, а если модель
  прислала файл целиком, заменяем файл.
"""
from __future__ import annotations

import ast
import copy
import io
import re
import shlex
import symtable
import textwrap
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

from .shell import run_bash, truncate

MAX_FAIL_CHARS = 1500
FULL_SOURCE_CHARS = 8000  # до этого размера все модули идут в запрос целиком
MAX_SOURCE_CHARS = 80_000  # больше — не наш случай, пусть разбирается обычный агент
FULL_TESTS_CHARS = 4000
SHORT_CONST_CHARS = 400
# Видимые тесты дольше 45 с — почти наверняка зацикливание или экспоненциальный алгоритм (подвох «tests finish
# quickly», сторонний набор). При 90 с два прогона с моделью между ними не укладывались в таймаут задачи 240 с.
TEST_TIMEOUT_S = 45

_SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache"}
_TEST_DIRS = {"tests", "test"}
_CMD = re.compile(r"`((?:python3?|pytest)(?:\s+[^`]+)?)`")
# pytest есть в образе с 13.11. Кэш не пишем, отчёт короткий: его читает модель.
_PYTEST_FLAGS = ("-q", "--tb=short", "-p no:cacheprovider")
_SAFE_CMD = re.compile(r"^python3?(?:\s+[\w./=:-]+)+$")
_PY_NAME = re.compile(r"[\w./-]+\.py\b")
_PY_BLOCK = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\r?\n(.*?)```", re.S)
_RC = re.compile(r"__RC=(\d+)\n?")
_SEP = re.compile(r"^={40,}[ \t]*$", re.M)
_DASH = re.compile(r"^-{40,}[ \t]*$", re.M)
_FAILED_TEST = re.compile(r"^(?:FAIL|ERROR): (\w+) \(", re.M)
_FAILED_COUNTS = re.compile(r"^FAILED \(([^)]*)\)", re.M)
_RAN = re.compile(r"^Ran \d+ tests?", re.M)
_PYTEST_TAIL = re.compile(r"^.*\b\d+ (?:passed|failed|errors?)\b.*\bin [\d.]+s\b.*$", re.M)
_PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR) \S+?::(?:\w+::)*(\w+)", re.M)
_TOP_DEF = re.compile(r"^(?:async\s+def|def|class)\s+(\w+)", re.M)
_WORD = re.compile(r"[A-Za-z_]\w*")

_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFS = (*_FUNCS, ast.ClassDef)
_IMPORTS = (ast.Import, ast.ImportFrom)
_SETUP = {"setUp", "setUpClass", "asyncSetUp", "setUpModule"}


# ---------------------------------------------------------------- раскладка задачи

def _py_files(workdir: Path) -> list[Path]:
    return sorted(p for p in workdir.rglob("*.py")
                  if not set(p.relative_to(workdir).parts[:-1]) & _SKIP_DIRS)


def is_test_file(path: Path, workdir: Path) -> bool:
    parts = set(path.relative_to(workdir).parts[:-1])
    return path.name.startswith("test_") or path.name.endswith("_test.py") or bool(parts & _TEST_DIRS)


def _imported(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out |= set(a.name.split("."))
        elif isinstance(n, ast.ImportFrom) and n.module:
            out |= set(n.module.split("."))
    return out


def with_pytest_flags(cmd: str) -> str:
    return cmd + "".join(f" {flag}" for flag in _PYTEST_FLAGS if flag not in cmd)


def has_pytest() -> bool:
    import importlib.util
    return importlib.util.find_spec("pytest") is not None


def test_commands(instruction: str, workdir: Path, tests: list[Path]) -> list[str]:
    """Команда из условия (`python3 test_x.py`, `python3 -m unittest discover ...`), иначе по раскладке.
    Остальные тестовые файлы в корне, которых команда не касается, тоже гоняем: проверка их учтёт."""
    root = [t for t in tests if t.parent == workdir]
    m = _CMD.search(instruction or "")
    if m:
        cmd = " ".join(m.group(1).split())
        if cmd.split()[0] == "pytest":  # `pytest -q` из условия -> тот же интерпретатор, что и у остальных команд
            cmd = "python3 -m " + cmd
        if _SAFE_CMD.match(cmd):
            cmd = re.sub(r"^python\s", "python3 ", cmd)
            if "pytest" in cmd.split():
                cmd = with_pytest_flags(cmd)
            if " -m " in f" {cmd} ":
                return [cmd]
            return [cmd] + [f"python3 {t.name}" for t in root if t.name not in cmd]
    if root:
        return [f"python3 {t.name}" for t in root]
    if tests:
        d = tests[0].parent.relative_to(workdir).as_posix()
        return [f"python3 -m unittest discover -s {d} -t ."]
    return []


@dataclass
class Layout:
    workdir: Path
    cmds: list[str]
    tests: list[Path]
    sources: list[Path]  # самые вероятные первыми: названные в условии, потом импортируемые тестами

    def rel(self, p: Path) -> str:
        return p.relative_to(self.workdir).as_posix()

    def read_sources(self) -> dict[str, str]:
        return {self.rel(p): p.read_text(encoding="utf-8", errors="replace") for p in self.sources}

    def write_sources(self, mods: dict[str, str]) -> None:
        for rel, text in mods.items():
            (self.workdir / rel).write_text(text, encoding="utf-8")


def find_layout(workdir: Path, instruction: str = "") -> Layout | None:
    files = _py_files(workdir)
    tests = [p for p in files if is_test_file(p, workdir)]
    if not tests:
        return None
    named = {Path(n).name for n in _PY_NAME.findall(instruction or "")}
    imported: set[str] = set()
    for t in tests:
        imported |= _imported(t)
    sources = [p for p in files if p not in tests and p.name not in ("conftest.py", "setup.py")
               and not (p.name == "__init__.py" and not p.read_text(encoding="utf-8", errors="replace").strip())]
    if not sources or sum(p.stat().st_size for p in sources) > MAX_SOURCE_CHARS:
        return None
    sources.sort(key=lambda p: (0 if p.name in named else 1 if p.stem in imported else 2, p.as_posix()))
    cmds = test_commands(instruction, workdir, tests)
    if not cmds:
        return None
    return Layout(workdir, cmds, tests, sources)


# ---------------------------------------------------------------- прогон тестов

@dataclass
class TestRun:
    passed: bool  # видимые тесты
    summary: str = ""
    nfail: int = 0
    failed_names: set[str] = field(default_factory=set)
    doc_fail: int = 0  # упавшие примеры >>> из docstring: бесплатная замена скрытым тестам
    doc_summary: str = ""

    def key(self) -> tuple[int, int]:
        """Чем меньше, тем лучше: сначала видимые тесты, потом примеры из docstring."""
        return (0 if self.passed else self.nfail, self.doc_fail)


def count_failures(out: str) -> int:
    """Сколько тестов упало. Тесты не запустились (синтаксис, импорт) — 99: это хуже любого числа падений."""
    if not _RAN.search(out):
        tail = _PYTEST_TAIL.findall(out)
        if tail:  # отчёт pytest: «1 failed, 2 passed in 0.03s»
            return sum(int(n) for n in re.findall(r"\b(\d+) (?:failed|errors?)\b", tail[-1])) or 99
    if "_FailedTest" in out or not _RAN.search(out):
        return 99
    n = sum(int(v) for grp in _FAILED_COUNTS.findall(out) for k, v in re.findall(r"(\w+)=(\d+)", grp)
            if k in ("failures", "errors"))
    return n or 99


_FILE_LINE = re.compile(r'\s*File "([^"]+)"')


def _skip_file_line(line: str) -> bool:
    m = _FILE_LINE.match(line)
    if not m:
        return False
    path = Path(m.group(1).replace("\\", "/"))
    return (path.name.startswith("test_") or path.name.endswith("_test.py") or bool({"tests", "test"} & set(path.parts))
            or "unittest" in path.parts or any(p.startswith("python3") or p == "lib" for p in path.parts))


def summarize_failures(out: str, workdir: Path, limit: int = MAX_FAIL_CHARS) -> str:
    """Отчёт unittest в несколько строк на каждый упавший тест: заголовок, проверка, ошибка."""
    for prefix in (str(workdir), workdir.as_posix()):
        out = out.replace(prefix + "/", "").replace(prefix + "\\", "")
    # Построчный diff unittest («- a b c», «+ b a c», «First differing element») дублирует строку AssertionError.
    noise = ("Traceback", "^", "~", "- ", "+ ", "? ", "First differing", "First list contains", "Second list contains",
             "Diff is ", "Items in the")
    blocks = _SEP.split(out)[1:]
    if not blocks:  # упало до запуска тестов: показываем хвост трейсбека
        lines = [ln.rstrip() for ln in out.splitlines() if ln.strip() and not ln.lstrip().startswith(noise)]
        return truncate("\n".join(lines[-14:]), limit)
    parts = []
    for b in blocks:
        head, _, body = b.strip("\n").partition("\n")
        body = _DASH.split(body, 1)[-1]
        body = re.split(r"^-{40,}[ \t]*$|^Ran \d+ tests?", body, flags=re.M)[0]
        # Строки File из тестов и стандартной библиотеки не нужны: исходник упавших тестов и так в запросе.
        keep = [ln.strip() for ln in body.splitlines()
                if ln.strip() and not ln.lstrip().startswith(noise) and not _skip_file_line(ln)]
        head = re.sub(r" \([\w.]+\)$", "", head.strip())
        parts.append(head + ("\n  " + "\n  ".join(keep[-6:]) if keep else ""))
    return truncate("\n".join(parts), limit)


_STARS = re.compile(r"^\*{20,}[ \t]*$", re.M)
_DOC_TOTAL = re.compile(r"\*\*\*Test Failed\*\*\* (\d+) failure")


def summarize_doctest(out: str) -> tuple[int, str]:
    """Отчёт doctest: число упавших примеров и строки «функция: пример -> ожидалось, получено».
    Модуль, который doctest не смог даже импортировать, не считаем: это не ошибка кода задачи."""
    blocks = [b.strip("\n") for b in _STARS.split(out) if "Failed example:" in b]
    if not blocks:
        return 0, ""
    lines = []
    for b in blocks:
        where = re.search(r", in ([\w.]+)", b)
        ex = re.search(r"Failed example:\n(.*?)\n(?:Expected:|Exception raised:|Expected nothing)", b, re.S)
        exp = re.search(r"Expected:\n(.*?)\n(?:Got|Got nothing)", b, re.S)
        got = re.search(r"Got:\n(.*)$", b, re.S)
        exc = b.split("Exception raised:", 1)[1].strip().splitlines()[-1].strip() if "Exception raised:" in b else ""
        item = f"{where.group(1) if where else '?'}: {' '.join((ex.group(1) if ex else '?').split())}"
        if exp:
            item += f" -> expected {' '.join(exp.group(1).split())}"
        item += f", got {' '.join(got.group(1).split())}" if got else (f", raised {exc}" if exc else "")
        lines.append(item)
    total = sum(int(n) for n in _DOC_TOTAL.findall(out)) or len(blocks)
    return total, truncate("\n".join(lines), 800)


async def run_doctests(layout: Layout) -> tuple[int, str]:
    mods = [layout.rel(p) for p in layout.sources if ">>>" in p.read_text(encoding="utf-8", errors="replace")]
    if not mods:
        return 0, ""
    out = await run_bash(" ; ".join(f"python3 -m doctest {m} 2>&1" for m in mods), layout.workdir,
                         timeout=TEST_TIMEOUT_S, limit=0)
    return summarize_doctest(out.replace("\r\n", "\n"))


_RAN_SOME = re.compile(r"^Ran [1-9]\d* tests?|\b[1-9]\d* (?:passed|failed)\b", re.M)
_SCRIPT_CMD = re.compile(r"python3 ([\w./-]+\.py)")
# Тестовые функции без классов (стиль pytest) запускаем через unittest: вывод в привычном для разбора виде.
_FUNC_RUNNER = ("import importlib, sys, unittest\n"
                "m = importlib.import_module({mod!r})\n"
                "fs = [f for n, f in sorted(vars(m).items()) if n.startswith('test') and callable(f) "
                "and not isinstance(f, type) and getattr(f, '__module__', None) == m.__name__]\n"
                "r = unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(map(unittest.FunctionTestCase, fs)))\n"
                "sys.exit(not r.wasSuccessful())\n")


def unrun_fallback(cmd: str, workdir: Path) -> str | None:
    """Команда `python3 test_x.py` вернула 0 и не сообщила ни об одном тесте. Если в файле классы TestCase без
    `unittest.main()` или функции test_* и классы Test* в стиле pytest, которые сам файл не вызывает, то тесты не
    запускались вовсе, и «код 0» ничего не значит: раньше харнесс писал DONE без единого вызова модели (проба
    13.11, 7 из 20 раскладок). Возвращает команду, которая их действительно запустит (unittest, pytest, а без
    pytest — простые функции через unittest), или None, если файл запускает проверки сам (assert на уровне
    модуля, свой main)."""
    m = _SCRIPT_CMD.fullmatch(cmd)
    if not m:
        return None
    rel = Path(m.group(1))
    tree = _parse_file(workdir / rel)
    if tree is None:
        return None
    mod = ".".join(rel.with_suffix("").parts)
    if any(isinstance(n, ast.ClassDef) and any("TestCase" in ast.unparse(b) for b in n.bases) for n in tree.body):
        return f"python3 -m unittest {mod}"
    funcs = [n for n in tree.body if isinstance(n, _FUNCS) and n.name.startswith("test")]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.startswith("Test")
               and any(isinstance(b, _FUNCS) and b.name.startswith("test") for b in n.body)]
    if not funcs and not classes:
        return None
    names = {f.name for f in funcs} | {c.name for c in classes}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    dynamic = {"globals", "locals", "vars", "getattr", "dir"} & used
    if names & used or dynamic:
        return None  # файл сам вызывает свои тесты (в main или перебором по именам)
    if has_pytest():  # функции с фикстурами, параметризация, классы Test* — всё это умеет только pytest
        return with_pytest_flags(f"python3 -m pytest {rel.as_posix()}")
    if classes or any(f.args.args or f.args.posonlyargs or f.args.kwonlyargs for f in funcs):
        return None
    return "python3 -c " + shlex.quote(_FUNC_RUNNER.format(mod=mod))


def _parse_file(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return None


async def run_tests(layout: Layout) -> TestRun:
    script = " ; ".join(f'{c} 2>&1; echo "__RC=$?"' for c in layout.cmds)
    out = (await run_bash(script, layout.workdir, timeout=TEST_TIMEOUT_S, limit=0)).replace("\r\n", "\n")
    doc_fail, doc_summary = await run_doctests(layout)
    codes = _RC.findall(out)
    if len(codes) == len(layout.cmds) and all(c == "0" for c in codes):
        # Код 0 без «Ran N tests»: возможно, тесты не запускались. Перезапускаем такие файлы настоящим раннером
        # и верим ему, только если он выполнил хотя бы один тест; иначе — как раньше, «прошли».
        outputs = _RC.split(out)[0::2]
        again = [fb for cmd, text in zip(layout.cmds, outputs) if not _RAN_SOME.search(text)
                 for fb in [unrun_fallback(cmd, layout.workdir)] if fb]
        if not again:
            return TestRun(True, doc_fail=doc_fail, doc_summary=doc_summary)
        script = " ; ".join(f'{c} 2>&1; echo "__RC=$?"' for c in again)
        out2 = (await run_bash(script, layout.workdir, timeout=TEST_TIMEOUT_S, limit=0)).replace("\r\n", "\n")
        codes2 = _RC.findall(out2)
        if all(c == "0" for c in codes2) or not _RAN_SOME.search(out2):
            return TestRun(True, doc_fail=doc_fail, doc_summary=doc_summary)
        out = out2
    out = _RC.sub("", out)
    if out.startswith("[timeout"):
        return TestRun(False, f"tests did not finish in {TEST_TIMEOUT_S} s: an infinite loop or an algorithm too slow "
                              "for the test inputs (e.g. exponential recursion). Make it fast: iterate or memoize, "
                              "avoid repeated work.", 99, doc_fail=doc_fail, doc_summary=doc_summary)
    summary = summarize_failures(out, layout.workdir)
    failed = set(_FAILED_TEST.findall(out)) | set(_PYTEST_FAILED.findall(out))
    return TestRun(False, summary, count_failures(out), failed, doc_fail, doc_summary)


# ---------------------------------------------------------------- запрос модели

def _parse(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _start(node: ast.AST) -> int:
    decs = getattr(node, "decorator_list", None)
    return decs[0].lineno if decs else node.lineno  # type: ignore[attr-defined]


def _seg(lines: list[str], node: ast.AST) -> str:
    return "\n".join(lines[_start(node) - 1:node.end_lineno])  # type: ignore[attr-defined]


def _names(node: ast.AST) -> set[str]:
    out = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Name):
            out.add(x.id)
        elif isinstance(x, ast.Attribute):
            out.add(x.attr)
    return out


def _assign_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def tests_context(tests: list[Path], failed: set[str], workdir: Path) -> str | None:
    """Исходник только упавших тестов (плюс setUp и импорты). None, если не разобрать — тогда тесты целиком,
    когда они небольшие."""
    parts = []
    for t in tests:
        text = t.read_text(encoding="utf-8", errors="replace")
        tree = _parse(text)
        if tree is None:
            return None
        lines = text.splitlines()
        picked = [n for n in ast.walk(tree) if isinstance(n, _FUNCS) and (n.name in failed or n.name in _SETUP)]
        if not any(n.name in failed for n in picked):
            continue
        header = "\n".join(_seg(lines, n) for n in tree.body if isinstance(n, _IMPORTS))
        body = "\n".join(textwrap.dedent(_seg(lines, n)).rstrip() for n in picked)
        rel = t.relative_to(workdir).as_posix()
        parts.append(f"# {rel} (only failing tests)\n{header}\n{body}".replace("\n\n\n", "\n\n"))
    return "\n\n".join(parts) or None


def module_context(sources: dict[str, str], refs: set[str]) -> str:
    """Небольшие модули целиком. В больших — то, что упоминают упавшие тесты и ошибки, плюс два уровня
    вызываемого, импорты и короткие константы; остальное перечислено по именам."""
    if sum(len(t) for t in sources.values()) <= FULL_SOURCE_CHARS:
        return "\n\n".join(f"{rel}:\n```python\n{text.rstrip()}\n```" for rel, text in sources.items())
    trees = {rel: _parse(text) for rel, text in sources.items()}
    index: dict[str, ast.AST] = {}
    for tree in trees.values():
        for n in tree.body if tree else []:
            if isinstance(n, _DEFS):
                index.setdefault(n.name, n)
            if isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, _FUNCS):
                        index.setdefault(m.name, n)
    selected = {id(index[w]): index[w] for w in refs if w in index}
    frontier = list(selected.values())
    for _ in range(2):
        nxt = []
        for node in frontier:
            for w in _names(node):
                tgt = index.get(w)
                if tgt is not None and id(tgt) not in selected:
                    selected[id(tgt)] = tgt
                    nxt.append(tgt)
        frontier = nxt
    if not selected:
        return "\n\n".join(f"{rel}:\n```python\n{text.rstrip()}\n```" for rel, text in sources.items())
    out = []
    for rel, text in sources.items():
        tree = trees[rel]
        if tree is None:
            out.append(f"{rel}:\n```python\n{text.rstrip()}\n```")
            continue
        lines = text.splitlines()
        kept, omitted = [], []
        for n in tree.body:
            snip = _seg(lines, n)
            if id(n) in selected or isinstance(n, _IMPORTS) or (_assign_name(n) and len(snip) <= SHORT_CONST_CHARS):
                kept.append(snip)
            elif isinstance(n, _DEFS):
                omitted.append(n.name)
        if not any(id(n) in selected for n in tree.body):
            if omitted:
                out.append(f"{rel}: defines {', '.join(omitted)} (not shown)")
            continue
        tail = f"\n# not shown, unchanged: {', '.join(omitted)}" if omitted else ""
        out.append(f"{rel} (relevant part):\n```python\n" + "\n\n".join(kept) + tail + "\n```")
    return "\n\n".join(out)


_BOILERPLATE = re.compile(r"\s*Work only inside the current directory\..*$", re.S)

# Правила короче вдвое (442 -> ~240 символов): на простых задачах они занимали 30 % запроса (09.11).
GENERAL_RULE = ("Hidden tests also check documented behaviour (names, docstrings, edge cases), even where no "
                "visible test fails. Never special-case test inputs.")
# Правило про ответ не сокращать: без «Omit docstrings and comments; do not repeat unchanged code» выход вырос
# с 66 до 102 токенов на вызов (gen-s1, 09.11) — это в разы дороже, чем экономия на входе.
REPLY_RULE = ("Reply with one ```python block holding only what you change: whole functions, methods inside their "
              "class (unchanged methods may be left out), module-level constants. Omit docstrings and comments; "
              "do not repeat unchanged code.")
# Стандартные фразы условия coding: файлы и команда и так видны в запросе.
_STD_SENTENCES = re.compile(r"Run `[^`]+` in the current directory;\s*some tests fail\.\s*|"
                            r"\(do not edit the tests\)\s*|until all tests pass\.?\s*", re.I)
IMPOSSIBLE_RULE = ("If the visible tests cannot all pass with any code (they contradict each other or demand "
                   "something impossible), reply only IMPOSSIBLE.")
SYSTEM_PROMPT_CODING = "You fix Python code. Reply with one ```python block and nothing else."


def parses(code: str) -> bool:
    return _parse(code) is not None


def says_impossible(text: str | None) -> bool:
    return (text or "").strip().strip("`*. ").upper().startswith("IMPOSSIBLE")


def split_patch(blocks: list[str]) -> list[list[str]]:
    """Ответ модели по одному определению (импорты блока идут с каждым): чтобы принять полезную часть
    правки, если вся целиком делает хуже."""
    units = []
    for code in blocks:
        tree = _parse(code)
        if tree is None:
            continue
        lines = code.splitlines()
        imports = [_seg(lines, n) for n in tree.body if isinstance(n, _IMPORTS)]
        for n in tree.body:
            if isinstance(n, _DEFS) or _assign_name(n):
                units.append(["\n".join(imports + [_seg(lines, n)])])
    return units


def defined_names(blocks: list[str]) -> list[str]:
    out = []
    for code in blocks:
        tree = _parse(code)
        for n in tree.body if tree else []:
            name = n.name if isinstance(n, _DEFS) else _assign_name(n)
            if name:
                out.append(name)
    return out


# Сверка с docstring во втором проходе (budget_agent/second.py): видимые тесты прошли, а скрытые проверяют поведение
# из docstring, которого видимые не касаются. Регресс 10.11: все 3 оставшихся провала coding такие — parse_int(None)
# («None gives default»), round_money(-0.125) («halves go away from zero»), slugify('Version 2.0').
AUDIT_MAX_CHARS = 12000
SYSTEM_PROMPT_AUDIT = "You check Python code against its docstrings. Reply OK or one ```python block, nothing else."
AUDIT_RULE = ("All visible tests pass. Hidden tests check every function against its docstring, including inputs the "
              "visible tests skip: None or empty values, negative numbers, rounding of halves, which value gets "
              "rounded (the final result, not an intermediate one, e.g. a price after discount rounded down), "
              "separators, letter case, the documented examples. Check also: off-by-one at boundaries (inclusive or "
              "exclusive ends, ceil versus floor in ranks and percentiles); inputs that lie outside a documented "
              "range (clip to it); unknown keys or items (the documented default instead of an exception); "
              "arguments or stored objects that must not be modified (copy them); validation the docstring requires "
              "(raise the documented exception); documented speed (O(1) operations must not scan a list). Code "
              "comments, in any language, are part of the specification. Fix only functions whose code contradicts their "
              "docstring for some input; do not change behaviour the docstring does not specify. " + REPLY_RULE +
              " If nothing contradicts a docstring, reply only OK.")


def build_audit_prompt(sources: dict[str, str], instruction: str) -> str | None:
    instr = _STD_SENTENCES.sub("", _BOILERPLATE.sub("", instruction or "")).strip()
    body = "\n\n".join(f"{rel}:\n```python\n{text.rstrip()}\n```" for rel, text in sources.items())
    if len(body) > AUDIT_MAX_CHARS:
        return None
    return "\n\n".join(x for x in (instr if len(instr) > 12 else "", body, AUDIT_RULE) if x)


def build_prompt(layout: Layout, sources: dict[str, str], run: TestRun, instruction: str,
                 allow_impossible: bool = False, note: str = "") -> str:
    instr = _STD_SENTENCES.sub("", _BOILERPLATE.sub("", instruction or "")).strip()
    instr = instr if len(instr) > 12 else "Fix the bugs so that the tests pass."
    refs = set(_WORD.findall(run.summary + " " + run.doc_summary))
    tctx = tests_context(layout.tests, run.failed_names, layout.workdir) if run.failed_names else None
    if tctx:
        for t in layout.tests:
            tree = _parse(t.read_text(encoding="utf-8", errors="replace"))
            for n in ast.walk(tree) if tree else []:
                if isinstance(n, _FUNCS) and (n.name in run.failed_names or n.name in _SETUP):
                    refs |= _names(n)
    parts = [instr, GENERAL_RULE, module_context(sources, refs)]
    summary = run.summary
    if tctx:
        parts.append(f"Failing tests:\n```python\n{tctx.rstrip()}\n```")
        # Строки assert уже есть в исходнике упавших тестов: в сводке оставляем имена и фактические значения.
        summary = "\n".join(ln for ln in summary.splitlines()
                            if not ln.strip().startswith(("self.assert", "assert ")))
    else:
        texts = {layout.rel(t): t.read_text(encoding="utf-8", errors="replace") for t in layout.tests}
        if sum(len(x) for x in texts.values()) <= FULL_TESTS_CHARS:
            parts += [f"{rel}:\n```python\n{x.rstrip()}\n```" for rel, x in texts.items()]
    if not run.passed:
        parts.append(f"Failing now:\n{summary}")
    if run.doc_fail:
        parts.append(f"Docstring examples that fail (python3 -m doctest):\n{run.doc_summary}")
    broken = [rel for rel, text in sources.items() if _parse(text) is None]
    if broken:
        parts.append(f"{broken[0]} has a syntax error: fix it as well.")
    if note:
        parts.append(note)
    parts.append(REPLY_RULE + (" " + IMPOSSIBLE_RULE if allow_impossible else ""))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- подстановка ответа

# Заголовок файла внутри блока: «# inventory/models.py».
_FILE_HEADER = re.compile(r"(?m)^[ \t]*#[ \t]*[\w./-]+\.py[ \t]*$")


def _split_by_files(block: str) -> list[str]:
    """Один блок на несколько файлов пакета, метод с отступом и без своего класса (hold-c6, c4 pkg_cart): блок не
    разбирается, и правка терялась трижды подряд. Режем по заголовкам файлов и снимаем отступ у каждого куска:
    подстановка (_Patcher) сама находит класс метода."""
    if parses(block) or not _FILE_HEADER.search(block):
        return [block]
    segs = [textwrap.dedent(p).strip("\n") for p in _FILE_HEADER.split(block) if p.strip()]
    return segs if segs and all(parses(s) for s in segs) else [block]


def extract_blocks(text: str | None) -> list[str]:
    if not text:
        return []
    blocks = [textwrap.dedent(b).strip("\n") for b in _PY_BLOCK.findall(text)]
    if blocks:
        return [s for b in blocks if b.strip() for s in _split_by_files(b)]
    t = textwrap.dedent(text).strip()
    return [t] if t.startswith(("def ", "class ", "import ", "from ", "async def ", "@")) else []


def extract_code(text: str | None) -> str | None:
    """Совместимость: самый длинный блок кода."""
    blocks = extract_blocks(text)
    return max(blocks, key=len) if blocks else None


def _block(lines: list[str], node: ast.AST) -> list[str]:
    """Строки узла со сдвигом так, чтобы он начинался с колонки 0."""
    col = node.col_offset  # type: ignore[attr-defined]
    return [ln[min(col, len(ln) - len(ln.lstrip())):] for ln in lines[_start(node) - 1:node.end_lineno]]  # type: ignore[attr-defined]


def _indent(block: list[str], col: int) -> list[str]:
    return [" " * col + ln if ln.strip() else ln for ln in block]


def _is_placeholder(node: ast.AST) -> bool:
    """Тело из `...` или `pass` означает «не менялось» и не должно затирать настоящий код."""
    body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)  # type: ignore[attr-defined]
                                          and isinstance(b.value.value, str))]
    return len(body) == 1 and (isinstance(body[0], ast.Pass) or (
        isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and body[0].value.value is Ellipsis))


def _docstring_node(node: ast.AST) -> ast.Expr | None:
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[0]
    return None


def _header_span(text: str, node: ast.AST) -> tuple[int, int]:
    """Объявление def/class до двоеточия; не захватывает декораторы, комментарии и тело."""
    lines = text.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
    first = node.body[0]
    body_line = _start(first)
    # AST считает столбцы в байтах UTF-8, tokenize — в символах. Декоратор первого метода тоже часть тела.
    body_col = len(lines[body_line - 1].encode("utf-8")[:first.col_offset].decode("utf-8"))
    body_start = sum(map(len, lines[:body_line - 1])) + body_col
    fragment = text[start:body_start]
    lines = fragment.splitlines(keepends=True)
    end = None
    for token in tokenize.generate_tokens(io.StringIO(fragment).readline):
        if token.type == tokenize.OP and token.string == ":":
            end = start + sum(map(len, lines[:token.end[0] - 1])) + token.end[1]
    if end is not None:
        return start, end
    raise ValueError("definition has no suite")


def _restore_docstring(block: list[str], new: ast.AST, old_lines: list[str], old: ast.AST) -> list[str]:
    """Модель опустила docstring (так дешевле): возвращаем прежний на место."""
    if not isinstance(new, _FUNCS) or _docstring_node(new) is not None:
        return block
    doc = _docstring_node(old)
    if doc is None or not new.body:
        return block
    # Компактный ответ «def f(x): return ...» прежде терял контракт. Вставка строки между заголовком и
    # телом здесь невозможна: разворачиваем только такой ответ, сохраняя docstring как узел AST.
    text = "\n".join(block)
    parsed = ast.parse(text).body[0]
    _, end = _header_span(text, parsed)
    if parsed.body[0].lineno == text[:end].count("\n") + 1:
        parsed.body.insert(0, copy.deepcopy(doc))
        return ast.unparse(parsed).splitlines()
    first = new.body[0]
    at = first.lineno - _start(new)
    indent = first.col_offset - new.col_offset
    doc_lines = old_lines[doc.lineno - 1:doc.end_lineno]
    doc_lines = [ln[min(doc.col_offset, len(ln) - len(ln.lstrip())):] for ln in doc_lines]
    return block[:at] + _indent(doc_lines, indent) + block[at:]


def _splice(text: str, old: ast.AST, block: list[str]) -> str:
    lines = text.splitlines()
    lines[_start(old) - 1:old.end_lineno] = _indent(block, old.col_offset)  # type: ignore[attr-defined]
    return "\n".join(lines) + "\n"


def _text_span(text: str, name: str) -> tuple[int, int] | None:
    """Границы определения верхнего уровня по тексту — для файла, который не разбирается."""
    lines = text.splitlines()
    pat = re.compile(rf"^(?:async\s+def|def|class)\s+{re.escape(name)}\b")
    for i, ln in enumerate(lines):
        if pat.match(ln):
            s = i
            while s > 0 and lines[s - 1].startswith("@"):
                s -= 1
            e = i + 1
            while e < len(lines) and (not lines[e].strip() or lines[e][0] in " \t#)]}"):
                e += 1
            while e > i + 1 and not lines[e - 1].strip():
                e -= 1
            return s, e
    return None


class _Patcher:
    def __init__(self, sources: dict[str, str]):
        self.mods = dict(sources)

    def _parsed(self):
        for rel, text in self.mods.items():
            tree = _parse(text)
            if tree is not None:
                yield rel, text, tree

    def function(self, node: ast.AST, plines: list[str]) -> str | None:
        block = _block(plines, node)
        for rel, text, tree in self._parsed():
            old = next((n for n in tree.body if isinstance(n, _FUNCS) and n.name == node.name), None)
            if old is not None:
                self.mods[rel] = _splice(text, old, _restore_docstring(block, node, text.splitlines(), old))
                return rel
        for rel, text in self.mods.items():  # файл с синтаксической ошибкой: замена по тексту
            if _parse(text) is None:
                span = _text_span(text, node.name)
                if span:
                    lines = text.splitlines()
                    lines[span[0]:span[1]] = block
                    self.mods[rel] = "\n".join(lines) + "\n"
                    return rel
        args = [*node.args.posonlyargs, *node.args.args]  # type: ignore[attr-defined]
        if not args or args[0].arg not in ("self", "cls"):
            return None
        owners = [(rel, text, m) for rel, text, tree in self._parsed() for c in tree.body
                  if isinstance(c, ast.ClassDef) for m in c.body if isinstance(m, _FUNCS) and m.name == node.name]
        if len(owners) != 1:  # метод без класса принимаем, только если владелец однозначен
            return None
        rel, text, old = owners[0]
        self.mods[rel] = _splice(text, old, _restore_docstring(block, node, text.splitlines(), old))
        return rel

    def klass(self, node: ast.ClassDef, plines: list[str]) -> str | None:
        """Частичный класс обновляет только упомянутые методы и атрибуты, остальное не теряется."""
        for rel, text, tree in self._parsed():
            old = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == node.name), None)
            if old is None:
                continue
            # В частичном классе отсутствие баз означает «оставь прежние». Явные базы/метакласс,
            # напротив, часть правки: без этого корректный LRUCache(OrderedDict) становился обычным object.
            header = copy.deepcopy(old)
            header.bases = node.bases or old.bases
            header.keywords = node.keywords or old.keywords
            if ([ast.dump(n) for n in header.bases + header.keywords]
                    != [ast.dump(n) for n in old.bases + old.keywords]):
                header.body, header.decorator_list = [ast.Pass()], []
                declaration = ast.unparse(header).splitlines()[0]
                start, end = _header_span(text, old)
                text = text[:start] + declaration + text[end:]
            for member in node.body:
                text = self._member(text, node.name, member, plines)
            self.mods[rel] = text
            return rel
        return None

    @staticmethod
    def _member(text: str, cname: str, member: ast.AST, plines: list[str]) -> str:
        owner = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == cname)
        if isinstance(member, _FUNCS):
            if _is_placeholder(member):
                return text
            old = next((n for n in owner.body if isinstance(n, _FUNCS) and n.name == member.name), None)
        elif _assign_name(member):
            old = next((n for n in owner.body if _assign_name(n) == _assign_name(member)), None)
        else:
            return text  # docstring класса, pass и прочее
        block = _block(plines, member)
        if old is not None:
            return _splice(text, old, _restore_docstring(block, member, text.splitlines(), old))
        lines = text.splitlines()
        lines[owner.end_lineno:owner.end_lineno] = ["", *_indent(block, owner.body[0].col_offset)]
        return "\n".join(lines) + "\n"

    def constant(self, node: ast.AST, plines: list[str]) -> str | None:
        name = _assign_name(node)
        for rel, text, tree in self._parsed():
            old = next((n for n in tree.body if _assign_name(n) == name), None)
            if old is not None:
                self.mods[rel] = _splice(text, old, _block(plines, node))
                return rel
        return None

    def add_new(self, home: str, nodes: list[ast.AST], plines: list[str]) -> None:
        """Новое из ответа: импорты после импортов модуля, константы перед определениями, функции в конец."""
        text = self.mods[home]
        tree = _parse(text)
        if tree is None or not nodes:
            return
        imports, consts, defs = [], [], []
        for n in nodes:
            blk = _block(plines, n)
            if isinstance(n, _IMPORTS):
                if "\n".join(blk).strip() not in text:
                    imports += blk
            elif isinstance(n, _DEFS):
                defs += ["", "", *blk]
            elif _assign_name(n):
                consts += blk
        lines = text.splitlines()
        first_def = next((_start(n) - 1 for n in tree.body if isinstance(n, _DEFS)), len(lines))
        if consts:
            lines[first_def:first_def] = [*consts, ""]
        if imports:
            after = 0
            for n in tree.body:
                if isinstance(n, _IMPORTS) or (n is tree.body[0] and _docstring_node(tree) is n):
                    after = n.end_lineno
                elif isinstance(n, _DEFS):
                    break
            lines[after:after] = imports
        self.mods[home] = "\n".join(lines + defs) + "\n"

    def whole_file(self, tree: ast.Module, code: str) -> bool:
        """Сломанный файл, присланный целиком (все его определения на месте), заменяем целиком."""
        offered = {n.name for n in tree.body if isinstance(n, _DEFS)}
        for rel, text in self.mods.items():
            if _parse(text) is None:
                need = set(_TOP_DEF.findall(text))
                if offered and need <= offered:
                    self.mods[rel] = code.rstrip() + "\n"
                    return True
        return False


def _always_exits(body: list[ast.stmt]) -> bool:
    """Консервативная проверка: следующий оператор недостижим на любом пути."""
    for node in body:
        if isinstance(node, (ast.Return, ast.Raise)):
            return True
        if isinstance(node, ast.If) and _always_exits(node.body) and _always_exits(node.orelse):
            return True
        if isinstance(node, ast.Try) and (_always_exits(node.finalbody) or (
                _always_exits(node.body) and all(_always_exits(h.body) for h in node.handlers))):
            return True
    return False


def _lift_stranded_methods(code: str, sources: dict[str, str]) -> str:
    """Исправить лишний отступ метода после return/raise другой функции в ответе модели.

    Поднимаем только недостижимый self/cls-метод с единственным существующим владельцем. Замыкания,
    зависимые от внешних локальных значений заголовки и неоднозначные классы оставляем как есть.
    """
    tree = _parse(code)
    if tree is None:
        return code
    owners, classes = {}, {}
    for text in sources.values():
        source = _parse(text)
        for cls in source.body if source else []:
            if isinstance(cls, ast.ClassDef):
                classes[cls.name] = classes.get(cls.name, 0) + 1
                for fn in cls.body:
                    if isinstance(fn, _FUNCS):
                        owners.setdefault(fn.name, []).append((cls.name, fn))
    explicit = {(cls.name, fn.name) for cls in tree.body if isinstance(cls, ast.ClassDef)
                for fn in cls.body if isinstance(fn, _FUNCS)}
    try:
        symbols = symtable.symtable(code, "model_patch", "exec")
    except SyntaxError:
        return code
    lifted = []
    for parent in tree.body:
        if not isinstance(parent, _FUNCS):
            continue
        scope = next((t for t in symbols.get_children()
                      if t.get_name() == parent.name and t.get_lineno() == parent.lineno), None)
        if scope is None:
            continue
        kept = []
        for fn in parent.body:
            args = [*fn.args.posonlyargs, *fn.args.args] if isinstance(fn, _FUNCS) else []
            matches = owners.get(fn.name, []) if isinstance(fn, _FUNCS) else []
            if not (args and args[0].arg in ("self", "cls") and len(matches) == 1 and _always_exits(kept)):
                kept.append(fn)
                continue
            owner, old = matches[0]
            nested = next((t for t in scope.get_children()
                           if t.get_name() == fn.name and t.get_lineno() == fn.lineno), None)
            header = [*fn.decorator_list, *fn.args.defaults, *fn.args.kw_defaults, fn.returns]
            header += [a.annotation for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)]
            header += [a.annotation for a in (fn.args.vararg, fn.args.kwarg) if a is not None]
            reads = set().union(*(_names(n) for n in header if n is not None))
            if (classes[owner] != 1 or (owner, fn.name) in explicit or nested is None or nested.get_frees()
                    or reads & set(scope.get_locals())
                    or [ast.dump(n) for n in fn.decorator_list] != [ast.dump(n) for n in old.decorator_list]):
                kept.append(fn)
                continue
            lifted.append(ast.ClassDef(name=owner, bases=[], keywords=[], body=[fn], decorator_list=[]))
            explicit.add((owner, fn.name))
        parent.body = kept
    if not lifted:
        return code
    tree.body.extend(lifted)
    return ast.unparse(ast.fix_missing_locations(tree))


def apply_patch(sources: dict[str, str], blocks: list[str], primary: str | None = None) -> dict[str, str] | None:
    """Новые тексты модулей после правок модели. None — ответ не разобрать или он ничего не меняет."""
    p = _Patcher(sources)
    for code in blocks:
        code = _lift_stranded_methods(code, sources)
        tree = _parse(code)
        if tree is None:
            return None
        if p.whole_file(tree, code):
            continue
        plines = code.splitlines()
        home, new = None, []
        for n in tree.body:
            if isinstance(n, _FUNCS):
                if _is_placeholder(n):
                    continue
                rel = p.function(n, plines)
            elif isinstance(n, ast.ClassDef):
                rel = p.klass(n, plines)
            elif _assign_name(n):
                rel = p.constant(n, plines)
            elif isinstance(n, _IMPORTS):
                new.append(n)
                continue
            else:
                continue  # вызовы для примера, if __name__ == "__main__" и прочее
            if rel is None:
                new.append(n)
            else:
                home = home or rel
        home = home or primary or next((r for r, t in p.mods.items() if _parse(t) is not None), None)
        if home is not None:
            p.add_new(home, new, plines)
    changed = {rel: t for rel, t in p.mods.items() if t != sources[rel]}
    if not changed or any(_parse(t) is None for t in changed.values()):
        return None
    for rel in changed:
        p.mods[rel] = add_missing_imports(p.mods[rel], rel, p.mods)
    return p.mods


# Имена, которые модель берёт из стандартной библиотеки, забывая импорт.
_STD_FROM = {"OrderedDict": "collections", "deque": "collections", "defaultdict": "collections",
             "Counter": "collections", "namedtuple": "collections", "Decimal": "decimal",
             "ROUND_HALF_UP": "decimal", "ROUND_HALF_EVEN": "decimal", "ROUND_DOWN": "decimal",
             "InvalidOperation": "decimal", "Fraction": "fractions", "dataclass": "dataclasses",
             "field": "dataclasses", "replace": "dataclasses", "reduce": "functools", "lru_cache": "functools",
             "total_ordering": "functools", "datetime": "datetime", "date": "datetime", "timedelta": "datetime"}
_STD_MODULES = {"copy", "math", "re", "json", "itertools", "functools", "collections", "datetime", "decimal",
                "fractions", "statistics", "heapq", "bisect", "string", "operator", "time", "random", "csv"}


def add_missing_imports(text: str, rel: str, mods: dict[str, str]) -> str:
    """Имя используется в модуле, но нигде в нём не определено и не импортировано: модель написала `Item(...)` в
    store.py без `from .models import Item`, тесты упали на NameError, и верная правка откатилась (hold-c7). Если
    имя определено на верхнем уровне соседнего модуля того же каталога или это известное имя стандартной
    библиотеки, дописываем импорт."""
    tree = _parse(text)
    if tree is None:
        return text
    import builtins
    bound = set(dir(builtins)) | {"__name__", "__file__", "__doc__"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound |= set(n.names)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = sorted(used - bound)
    if not missing:
        return text
    folder = rel.rsplit("/", 1)[0] if "/" in rel else ""
    in_package = f"{folder}/__init__.py" in mods if folder else False
    lines = []
    for name in missing:
        src = None
        for other, otext in mods.items():
            if other == rel or (other.rsplit("/", 1)[0] if "/" in other else "") != folder:
                continue
            otree = _parse(otext)
            if otree is not None and any(
                    (isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and d.name == name)
                    or _assign_name(d) == name for d in otree.body):
                src = other.rsplit("/", 1)[-1][:-3]
                break
        if src and src != "__init__":
            lines.append(f"from {'.' if in_package else ''}{src} import {name}")
        elif name in _STD_FROM:
            lines.append(f"from {_STD_FROM[name]} import {name}")
        elif name in _STD_MODULES:
            lines.append(f"import {name}")
    if not lines:
        return text
    # После docstring модуля и `from __future__`, чтобы не сломать их порядок.
    body = tree.body
    at = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        at = body[0].end_lineno
    for n in body:
        if isinstance(n, ast.ImportFrom) and n.module == "__future__":
            at = max(at, n.end_lineno)
    src_lines = text.splitlines(keepends=True)
    return "".join(src_lines[:at]) + "\n".join(lines) + "\n" + "".join(src_lines[at:])


def apply_functions(src: str, patch: str) -> str | None:
    """Совместимость со старым интерфейсом: один модуль."""
    out = apply_patch({"m.py": src}, [patch], "m.py")
    return out["m.py"] if out else None
