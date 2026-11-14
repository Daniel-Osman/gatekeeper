"""Per-stage timing and token accounting.

A stage dict carries its own five fields first — ``stage``, ``latency_ms``,
``tokens_in``, ``tokens_out``, ``model`` — then the payload keys at the same
top level. There is no nested ``payload`` object, and the five reserved keys
always win over a payload key of the same name.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Stage:
    stage: str
    latency_ms: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str | None = None
    payload: dict | None = None

    def as_dict(self) -> dict:
        entry = {
            "stage": self.stage,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "model": self.model,
        }
        for key, value in (self.payload or {}).items():
            if key not in entry:
                entry[key] = value
        return entry


@contextmanager
def timed(stages: list[Stage], name: str, model: str):
    """Append a Stage on entry and stamp its wall time on exit.

    The stage is registered before the work starts, so a stage that raises is
    still present in the trace.
    """
    stage = Stage(stage=name, model=model)
    stages.append(stage)
    started = time.monotonic()
    try:
        yield stage
    finally:
        stage.latency_ms = round((time.monotonic() - started) * 1000.0, 3)
