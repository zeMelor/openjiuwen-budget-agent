"""Учёт токенов по правилам регламента: cost = completion + prompt / 10.

Токены рассуждений прокси уже включает в completion_tokens (проверено в docs/proxy.md),
кеш не учитывается.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


def call_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return completion_tokens + prompt_tokens / 10


@dataclass
class Budget:
    limit: float
    log_path: Path | None = None
    spent: float = 0.0
    calls: int = 0
    by_task: dict[str, float] = field(default_factory=dict)

    @property
    def remaining(self) -> float:
        return self.limit - self.spent

    def record(self, task_id: str, model: str, prompt_tokens: int, completion_tokens: int,
               reasoning_tokens: int = 0) -> float:
        cost = call_cost(prompt_tokens, completion_tokens)
        self.spent += cost
        self.calls += 1
        self.by_task[task_id] = self.by_task.get(task_id, 0.0) + cost
        if self.log_path is not None:
            row = {
                "t": round(time.time(), 3), "task": task_id, "model": model,
                "prompt": prompt_tokens, "completion": completion_tokens,
                "reasoning": reasoning_tokens, "cost": cost, "spent": round(self.spent, 1),
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return cost
