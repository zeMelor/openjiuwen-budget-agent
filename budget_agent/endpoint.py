"""Адрес прокси: с /v1 или без.

Клиент OpenAI дописывает к адресу только «/chat/completions». Если организаторы передадут LLM_BASE_URL без «/v1»
(в регламенте формат не указан), а прокси слушает «/v1/chat/completions», все вызовы дадут 404 и прогон обнулится.
Поэтому при старте, только если адрес не кончается на версию, харнесс шлёт на оба пути пустой POST (без модели и
сообщений: прокси отвечает 400/422 и токенов не тратит) и берёт тот, где маршрут есть. Ничего не понятно (оба 404,
сеть, 401 на всё) — адрес не трогаем.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request

_VERSIONED = re.compile(r"/v\d+(?:beta)?$")
_NO_ROUTE = (404, 405)


def _status(url: str, key: str, timeout: float) -> int | None:
    req = urllib.request.Request(url, data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001 — сеть, таймаут, кривой адрес: решения нет, адрес не трогаем
        return None


def resolve_base_url(base: str, key: str, timeout: float = 5.0, status=_status) -> tuple[str, str]:
    """(адрес, пояснение для лога)."""
    b = base.strip().rstrip("/")
    if _VERSIONED.search(b):
        return b, "versioned"
    plain = status(b + "/chat/completions", key, timeout)
    if plain is None or plain not in _NO_ROUTE:
        return b, f"plain path answers {plain}"
    v1 = status(b + "/v1/chat/completions", key, timeout)
    if v1 is not None and v1 not in _NO_ROUTE:
        return b + "/v1", f"plain path {plain}, /v1 path {v1}: using /v1"
    return b, f"plain path {plain}, /v1 path {v1}: unchanged"
