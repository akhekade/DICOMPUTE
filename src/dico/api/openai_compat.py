"""OpenAI-compatible consumer façade over collective MLP inference."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Awaitable

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from dico.auth import AuthContext
from dico.billing import BillingService
from dico.config import AppConfig
from dico.protocol import InferenceRequest
from dico.telemetry import METRICS, RequestTimer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "collective-mlp"
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 0.0
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]
    dico: dict[str, Any] = Field(default_factory=dict)


def _features_from_messages(messages: list[ChatMessage], dim: int = 8) -> list[float]:
    """Deterministic bag-of-bytes embedding into fixed feature vector."""
    text = "\n".join(m.content for m in messages)
    raw = text.encode("utf-8")
    feats = [0.0] * dim
    if not raw:
        return feats
    for i, b in enumerate(raw):
        feats[i % dim] += (b / 255.0) - 0.5
    # normalize lightly
    norm = sum(abs(x) for x in feats) or 1.0
    return [x / norm for x in feats]


def create_openai_router(
    *,
    cfg: AppConfig,
    billing: BillingService,
    infer_fn: Callable[[InferenceRequest, str | None], Awaitable[Any]],
    auth_dep: Callable,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["openai"])

    @router.get("/models")
    async def list_models(_auth: AuthContext | None = Depends(auth_dep)) -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": "collective-mlp",
                    "object": "model",
                    "created": 0,
                    "owned_by": "dicompute",
                }
            ],
        }

    @router.get("/pricing")
    async def pricing(_auth: AuthContext | None = Depends(auth_dep)) -> dict:
        return {
            "currency": "USD",
            "scale": 1_000_000,
            "infer_micro_usd": cfg.price_infer_micro_usd,
            "train_micro_usd": cfg.price_train_micro_usd,
        }

    @router.post("/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
        auth: AuthContext | None = Depends(auth_dep),
    ) -> ChatCompletionResponse:
        if body.stream:
            raise HTTPException(400, "streaming not supported yet")
        if body.model not in {"collective-mlp", "dico-collective"}:
            raise HTTPException(404, f"model not found: {body.model}")

        request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        account_id = auth.account_id if auth else "acct-default"
        reserved = billing.reserve(
            account_id, request_id, billing.estimate_infer_cost(1)
        )

        features = _features_from_messages(body.messages)
        infer_req = InferenceRequest(
            features=features,
            request_id=request_id,
            ensemble=True,
            routing="cheapest",
            model="collective-mlp",
            timeout_s=cfg.default_infer_timeout_s,
        )

        with RequestTimer("chat_completions"):
            try:
                result = await infer_fn(infer_req, account_id)
            except Exception as exc:
                # refund full reservation
                billing.settle(
                    reserved,
                    actual_cost=0,
                    kind="chat",
                    units=0,
                    provider_id=None,
                    latency_ms=0,
                    meta={"error": str(exc)},
                )
                METRICS.outcome("client", "chat_error")
                raise

        pred = int(result.prediction)
        probs = result.probabilities
        content = (
            f"class={pred} probs=[{', '.join(f'{p:.3f}' for p in probs)}] "
            f"strategy={result.strategy}"
        )
        settle = billing.settle(
            reserved,
            actual_cost=result.cost_micro_usd or billing.estimate_infer_cost(1),
            kind="chat",
            units=1,
            provider_id=(result.provider_ids[0] if result.provider_ids else None),
            latency_ms=sum(c.latency_ms for c in result.contributors),
            meta={"strategy": result.strategy},
        )
        METRICS.outcome("client", "chat_ok")
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=body.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": max(1, sum(len(m.content) for m in body.messages) // 4),
                "completion_tokens": max(1, len(content) // 4),
                "total_tokens": 0,
            },
            dico={
                "prediction": pred,
                "probabilities": probs,
                "model_version": result.model_version,
                "strategy": result.strategy,
                "cost_micro_usd": settle.charged_micro_usd,
                "providers": result.provider_ids,
            },
        )

    # fill total_tokens in response via model_validator alternative: patch here
    @router.get("/balance")
    async def balance(
        auth: AuthContext | None = Depends(auth_dep),
    ) -> dict:
        account_id = auth.account_id if auth else "acct-default"
        acc = billing.store.get_account(account_id)
        if not acc:
            raise HTTPException(404, "account not found")
        return {
            "account_id": acc.account_id,
            "balance_micro_usd": acc.balance_micro_usd,
            "balance_usd": acc.balance_micro_usd / 1_000_000,
        }

    return router
