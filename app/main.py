"""HTTP surface: one gate question, one health check."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import index
from app.gates.decisions import StageError
from app.pipeline import run


@asynccontextmanager
async def lifespan(app: FastAPI):
    index.ensure_loaded()
    yield


app = FastAPI(title="gatekeeper", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    policy: Literal["strict", "fast_fix"]


@app.get("/health")
def health() -> dict:
    index.ensure_loaded()
    return {"ok": True, "corpus_chunks": len(index.chunks)}


@app.post("/ask")
async def ask(request: AskRequest):
    try:
        return await run(request.question, request.policy)
    except StageError as exc:
        trace = exc.trace or {"outcome": "error", "policy": request.policy, "stages": []}
        return JSONResponse(
            status_code=502,
            content={
                "detail": f"stage {exc.stage} failed: {exc.message}",
                "stage": exc.stage,
                "trace": trace,
            },
        )
