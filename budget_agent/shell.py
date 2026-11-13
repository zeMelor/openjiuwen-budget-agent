"""Асинхронный запуск bash-команды в каталоге задачи.

Асинхронно, потому что синхронные инструменты openJiuwen выполняются прямо в event loop
и блокировали бы параллельные задачи.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import sys
import tempfile
from pathlib import Path

MAX_OUTPUT_CHARS = 2000
CMD_TIMEOUT_S = 60


def _find_bash() -> str:
    override = os.environ.get("HARNESS_BASH")
    if override:
        return override
    if os.name == "nt":
        # На Windows `bash` в PATH может оказаться WSL-ным, поэтому сначала ищем Git Bash.
        for cand in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if Path(cand).exists():
                return cand
    return shutil.which("bash") or "/bin/bash"


def _python3_shim_dir() -> str | None:
    """Подкладываем python3 -> текущий интерпретатор, если настоящего python3 нет.

    На Windows python3 в PATH обычно заглушка Microsoft Store (печатает «Python», код 49),
    поэтому там обёртка нужна всегда.
    """
    if os.name != "nt" and shutil.which("python3"):
        return None
    d = Path(tempfile.gettempdir()) / "budget_agent_shim"
    d.mkdir(exist_ok=True)
    exe = Path(sys.executable).as_posix()
    shim = d / "python3"
    shim.write_text(f'#!/bin/sh\nexec "{exe}" "$@"\n', encoding="utf-8", newline="\n")
    return str(d)


BASH = _find_bash()
_SHIM = _python3_shim_dir()


def _env() -> dict[str, str]:
    env = dict(os.environ)
    # Ключ прокси командам агента не нужен.
    env.pop("LLM_API_KEY", None)
    env["PYTHONIOENCODING"] = "utf-8"
    # Без байткода: правка того же размера, тест и откат в пределах секунды оставляли в __pycache__ байткод
    # отвергнутой правки, и тесты падали на верном исходнике (проверено 13.11). Заодно каталог задачи чище.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if _SHIM:
        env["PATH"] = _SHIM + os.pathsep + env.get("PATH", "")
    return env


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n...[{len(text) - limit} chars cut]...\n{text[-half:]}"


# Локально (не в контейнере) команды модели идут прямо на машину разработчика.
# Грубый предохранитель от выхода за каталог задачи и рекурсивного удаления; в контейнере не нужен.
# Буква диска только отдельным словом: `best:/if` в sed-выражении путём не считается.
_HOST_GUARD = re.compile(r"\.\.|~|\brm\s+-\w*[rR]|(?<![\w])[A-Za-z]:[\\/]|(^|[\s'\"=])/[a-z]/")


def _kill_tree(proc) -> None:
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass


async def run_bash(cmd: str, cwd: Path, timeout: float = CMD_TIMEOUT_S, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Вывод команды. limit=0 — без обрезки (харнессу для разбора отчёта тестов)."""
    if os.name == "nt" and _HOST_GUARD.search(cmd):
        return "[blocked: stay inside the current directory, use relative paths]"
    # Своя группа процессов: по таймауту убиваем и детей bash. Раньше убивался только bash, а python3 с
    # экспоненциальным fib(90) держал канал вывода, proc.wait() ждал его вечно, и задача умирала по таймауту,
    # не дойдя до модели (чужой набор jul-os, coding_0012/0020).
    proc = await asyncio.create_subprocess_exec(
        BASH, "-c", cmd, cwd=str(cwd), env=_env(),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        **({"start_new_session": True} if os.name != "nt" else {}),
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        _kill_tree(proc)
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except asyncio.TimeoutError:
            pass
        if isinstance(exc, asyncio.CancelledError):
            raise
        return f"[timeout after {timeout:.0f}s]"
    text = out.decode("utf-8", "replace").strip()
    if not limit:
        return text
    return truncate(f"[exit {proc.returncode}]\n{text}" if proc.returncode else text or "[no output]", limit)


async def run_python(code: str, cwd: Path, timeout: float = 30.0) -> tuple[int, str, str]:
    """Запуск кода модели текущим интерпретатором (в контейнере это python3): (код возврата, stdout, stderr).
    -I: модули из каталога задачи не подхватываются. Ключа прокси в окружении нет."""
    if os.name == "nt" and _HOST_GUARD.search(code.replace("...", "")):
        return 1, "", "blocked: stay inside the current directory, use relative paths"
    fd, path = tempfile.mkstemp(suffix=".py", prefix="model_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", path, cwd=str(cwd), env=_env(),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            **({"start_new_session": True} if os.name != "nt" else {}),
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            _kill_tree(proc)
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                pass
            if isinstance(exc, asyncio.CancelledError):
                raise
            return 124, "", f"timeout after {timeout:.0f}s"
        return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
