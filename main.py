"""
kie.ai Relay — OpenAI-compatible API proxy for kie.ai.

Endpoints:
  Public:
    GET  /health                 → Health check
    GET  /v1/models              → List available models
    POST /v1/images/generations  → Image generation (with billing)
    POST /v1/chat/completions    → Chat completion (with billing)
    GET  /v1/me                  → Check my balance

  Admin (requires RELAY_API_KEY):
    POST /admin/users            → Create user
    GET  /admin/users            → List users
    POST /admin/users/topup      → Add balance
    GET  /admin/usage            → Usage statistics
"""

import base64
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from config import settings
from kie_client import KieClient
import user_manager

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
logger = logging.getLogger("kie_relay")

# ── Lifespan ───────────────────────────────────────────────────
kie_client: Optional[KieClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kie_client
    if not settings.kie_api_key or settings.kie_api_key in ("", "your-kie-api-key-here"):
        logger.warning("KIE_API_KEY 未配置！请在 .env 文件中设置")
    kie_client = KieClient(
        api_key=settings.kie_api_key,
        base_url=settings.kie_api_base,
    )
    logger.info("kie.ai relay started on %s:%s", settings.host, settings.port)
    yield
    await kie_client.close()


app = FastAPI(
    title="kie.ai Relay",
    version="0.2.0",
    description="OpenAI-compatible API proxy for kie.ai with billing",
    lifespan=lifespan,
)


# ── Auth helpers ──────────────────────────────────────────────

def _extract_token(request: Request) -> str:
    """Extract Bearer token from request."""
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


async def verify_admin(request: Request):
    """Verify the request has the master RELAY_API_KEY."""
    token = _extract_token(request)
    if not settings.relay_api_key or token != settings.relay_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin API key")


async def verify_user(request: Request) -> dict:
    """Verify user API key, return user dict. Deducts balance after use."""
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")

    user = user_manager.get_user_by_key(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not user.get("enabled", True):
        raise HTTPException(status_code=403, detail="Account disabled")

    # Store user info in request state for later deduction
    request.state.user_key = token
    request.state.user = user
    return user


async def check_balance(request: Request, cost: float):
    """Check if user has enough balance, raise if not."""
    if request.state.user["balance"] < cost:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient balance. Required: {cost:.1f}, "
                   f"available: {request.state.user['balance']:.1f}. "
                   "Please top up."
        )


# ── Model definitions ────────────────────────────────────────────

MODEL_MAP = {
    "z-image": "z-image",
    "google/imagen-4": "google/imagen4",
    "google/imagen-4-fast": "google/imagen4-fast",
    "ideogram-v3": "ideogram/v3-text-to-image",
    "bytedance/seedream": "bytedance/seedream",
    "grok-imagine": "grok-imagine/text-to-image",
    "recraft-v3": "recraft-v3/text-to-image",
    "black-forest-labs/flux-pro": "black-forest-labs/flux-pro",
    "black-forest-labs/flux-dev": "black-forest-labs/flux-dev",
    "stability-ai/sdxl": "stability-ai/sdxl",
    "hailuo/text-to-video": "hailuo/02-text-to-video-pro",
    "hailuo/image-to-video": "hailuo/02-image-to-video-standard",
    "kling/v2.1-standard": "kling/v2-1-standard",
    "kling/v2.1-turbo": "kling/v2-1-turbo",
}

KIE_TO_FRIENDLY = {v: k for k, v in MODEL_MAP.items()}


def resolve_model(model: str) -> str:
    if model in MODEL_MAP:
        return MODEL_MAP[model]
    if "/" in model or model in KIE_TO_FRIENDLY.values():
        return model
    low = model.lower().replace("-", " ").replace("_", " ")
    for friendly, kie_id in MODEL_MAP.items():
        if low in friendly.lower() or low in kie_id.lower():
            return kie_id
    return model


# ── Pydantic models ────────────────────────────────────────────

class ImageGenerationRequest(BaseModel):
    model: str = Field(default="z-image")
    prompt: str = Field(..., description="Text description")
    n: int = Field(default=1, ge=1, le=4)
    size: Optional[str] = None
    negative_prompt: Optional[str] = None
    response_format: Optional[str] = Field(default="url")
    model_config = {"extra": "allow"}


class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]
    cost: Optional[float] = None
    balance_remaining: Optional[float] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="z-image")
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    model_config = {"extra": "allow"}


# Admin request models
class CreateUserRequest(BaseModel):
    name: str = Field(..., description="User name")
    initial_balance: float = Field(default=10.0, ge=0)


class TopupRequest(BaseModel):
    api_key: str = Field(..., description="User's API key (full)")
    amount: float = Field(..., gt=0, description="Amount to add")


# ── Helpers ─────────────────────────────────────────────────────

STANDARD_RATIOS = {"1:1": "1:1", "4:3": "4:3", "3:4": "3:4", "16:9": "16:9", "9:16": "9:16"}


def _size_to_aspect_ratio(size: Optional[str]) -> str:
    if not size:
        return "1:1"
    parts = size.lower().split("x")
    if len(parts) == 2:
        try:
            w, h = int(parts[0]), int(parts[1])
            ratio = w / h
            closest = min(STANDARD_RATIOS, key=lambda r: abs(
                ratio - int(r.split(":")[0]) / int(r.split(":")[1])
            ))
            return closest
        except ValueError:
            pass
    return "1:1"


def _kie_result_to_openai_images(data: dict, response_format: str = "url", n: int = 1) -> list[dict]:
    result_str = data.get("resultJson") or "{}"
    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    urls = result.get("resultUrls") or []
    images = []
    for i, url in enumerate(urls[:n]):
        if response_format == "b64_json":
            try:
                resp = httpx.get(url, timeout=30)
                resp.raise_for_status()
                b64 = base64.b64encode(resp.content).decode("utf-8")
                images.append({"b64_json": b64})
            except Exception as e:
                logger.warning("Failed to download image %s: %s", url, e)
                images.append({"url": url})
        else:
            images.append({"url": url})
    if not images:
        direct_url = data.get("url") or data.get("output_url") or result.get("url")
        if direct_url:
            images.append({"url": direct_url})
    return images


# ── Public routes ───────────────────────────────────────────────

@app.get("/health")
async def health():
    if kie_client and settings.kie_api_key not in ("", "your-kie-api-key-here"):
        try:
            credit = await kie_client.get_credit()
            users = user_manager.list_users()
            return {
                "status": "ok",
                "credit": credit,
                "users": len(users),
                "total_calls": sum(u["total_calls"] for u in users),
            }
        except Exception as e:
            return {"status": "degraded", "error": str(e)}
    return {"status": "ok", "warning": "KIE_API_KEY not configured"}


@app.get("/v1/models")
async def list_models(request: Request):
    """List available models (requires user API key)."""
    await verify_user(request)
    models = []
    for friendly_name in MODEL_MAP:
        models.append({
            "id": friendly_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "kie-relay",
        })
    return {"object": "list", "data": models}


@app.get("/v1/me")
async def my_info(request: Request):
    """Check my own balance and usage."""
    user = await verify_user(request)
    return {
        "name": user["name"],
        "balance": user["balance"],
        "total_used": user["total_used"],
        "total_calls": user["total_calls"],
        "enabled": user.get("enabled", True),
    }


@app.post("/v1/images/generations")
async def create_image(request: Request, body: ImageGenerationRequest):
    """Generate image(s) with billing."""
    await verify_user(request)
    if not kie_client:
        raise HTTPException(status_code=503, detail="kie.ai client not initialized")

    kie_model = resolve_model(body.model)
    cost = user_manager.get_model_cost(kie_model) * body.n
    await check_balance(request, cost)

    logger.info("Image gen: user=%s model=%s cost=%.1f balance=%.1f",
                request.state.user["name"], body.model, cost,
                request.state.user["balance"])

    input_data = {"prompt": body.prompt}
    if body.negative_prompt:
        input_data["negative_prompt"] = body.negative_prompt
    input_data["aspect_ratio"] = _size_to_aspect_ratio(body.size)
    if body.n > 1:
        input_data["num_images"] = str(body.n)
    extra = getattr(body, 'model_extra', None) or {}
    for key in ("image_size", "style", "seed", "guidance_scale",
                 "nsfw_checker", "expand_prompt", "rendering_speed"):
        if key in extra and extra[key] is not None:
            input_data[key] = extra[key]

    try:
        task_id = await kie_client.create_task(kie_model, input_data)
        result = await kie_client.poll_task(
            task_id,
            interval=settings.poll_interval,
            timeout=settings.poll_timeout,
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"kie.ai error: {e.response.text}")

    images = _kie_result_to_openai_images(result, body.response_format, body.n)
    if not images:
        raise HTTPException(status_code=502, detail="No image URLs in response")

    # Deduct balance only on success
    user_manager.deduct_balance(request.state.user_key, cost, body.model)

    return {
        "created": int(time.time()),
        "data": [{"url": img.get("url"), "b64_json": img.get("b64_json")} for img in images],
        "cost": cost,
        "balance_remaining": round(request.state.user["balance"] - cost, 1),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """Chat completion (wraps image gen for convenience) with billing."""
    await verify_user(request)
    if not kie_client:
        raise HTTPException(status_code=503, detail="Not initialized")

    prompt = ""
    for msg in reversed(body.messages):
        if msg.role == "user":
            prompt = msg.content
            break
    if not prompt:
        raise HTTPException(status_code=400, detail="No user message")

    kie_model = resolve_model(body.model)
    cost = user_manager.get_model_cost(kie_model)
    await check_balance(request, cost)

    try:
        task_id = await kie_client.create_task(kie_model, {"prompt": prompt})
        result = await kie_client.poll_task(
            task_id, interval=settings.poll_interval, timeout=settings.poll_timeout,
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    images = _kie_result_to_openai_images(result, "url", 1)
    image_url = images[0]["url"] if images else ""

    user_manager.deduct_balance(request.state.user_key, cost, body.model)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"![generated image]({image_url})"
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "cost": cost,
        "balance_remaining": round(request.state.user["balance"] - cost, 1),
    }


# ── Admin routes ─────────────────────────────────────────────

@app.post("/admin/users")
async def admin_create_user(request: Request, body: CreateUserRequest):
    """Create a new user with initial balance."""
    await verify_admin(request)
    user = user_manager.create_user(body.name, body.initial_balance)
    return {
        "message": "User created",
        "api_key": user["api_key"],  # Full key shown only at creation
        "name": user["name"],
        "balance": user["balance"],
    }


@app.get("/admin/users")
async def admin_list_users(request: Request):
    """List all users."""
    await verify_admin(request)
    return {"users": user_manager.list_users()}


@app.post("/admin/users/topup")
async def admin_topup(request: Request, body: TopupRequest):
    """Add balance to a user by their full API key."""
    await verify_admin(request)
    user = user_manager.topup_user(body.api_key, body.amount)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "message": f"Topup successful",
        "name": user["name"],
        "new_balance": user["balance"],
    }


@app.get("/admin/usage")
async def admin_usage(request: Request):
    """Get usage statistics."""
    await verify_admin(request)
    return user_manager.get_usage_summary()


# ── Run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
