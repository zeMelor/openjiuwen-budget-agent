"""Логи openJiuwen: по умолчанию он пишет в ./logs/ относительно текущего каталога.

В контейнере организаторов текущий каталог /app, а корень read-only, поэтому импорт
openJiuwen падает с «Read-only file system: '/app/logs'». Переносим логи во временный
каталог и убираем вывод в консоль.

Вызывать до первого импорта модулей openJiuwen, которые пишут в лог.
"""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path


def setup_openjiuwen_logging(level: str = "WARNING") -> Path:
    """Логи всегда во временный каталог (в контейнере /tmp), а не в --log-dir.

    openJiuwen делает chmod своему лог-файлу. На смонтированном с хоста каталоге под --cap-drop ALL
    это запрещено, и импорт падает с «Operation not permitted». В /tmp файл наш, chmod проходит.
    """
    from openjiuwen.core.common.logging.default.constant import DEFAULT_INNER_LOG_CONFIG
    from openjiuwen.core.common.logging.log_config import configure_log_config

    log_path = Path(tempfile.gettempdir()) / "openjiuwen_logs"
    cfg = copy.deepcopy(DEFAULT_INNER_LOG_CONFIG)
    cfg.update({
        "level": level,
        "log_path": str(log_path) + "/",
        "output": ["file"],
        "interface_output": ["file"],
        "performance_output": ["file"],
    })
    configure_log_config(cfg)
    return log_path
