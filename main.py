"""
kie.ai Relay — OpenAI-compatible API proxy for kie.ai.

Translates OpenAI-format requests into kie.ai's task-based API,
polls for completion, and returns OpenAI-format responses.

Endpoints:
  GET  /v1/models              → List available models
  POST /v1/images/generations  → Image generation
  GET  /health                 → Health check
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
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from kie_client import KieClient

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
    version="0.1.0",
    description="OpenAI-compatible API proxy for kie.ai",
    lifespan=lifespan,
)


# ── Auth middleware ──────────────────────────────────────────────

async def verify_auth(request: Request):
    """If ``relay_api_key`` is set, reject requests without a matching Bearer token."""
    if not settings.relay_api_key:
        return  # no auth required
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if token != settings.relay_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Model definitions ────────────────────────────────────────────

# Mapping from "friendly name" → kie.ai model identifier.
# Users can use these names in their requests; the relay translates them.
MODEL_MAP = {
    # Image generation
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
    # Video generation
    "hailuo/text-to-video": "hailuo/02-text-to-video-pro",
    "hailuo/image-to-video": "hailuo/02-image-to-video-standard",
    "kling/v2.1-standard": "kling/v2-1-standard",
    "kling/v2.1-turbo": "kling/v2-1-turbo",
}

# Reverse map: kie.ai model → friendly name
KIE_TO_FRIENDLY = {v: k for k, v in MODEL_MAP.items()}


def resolve_model(model: str) -> str:
    """Resolve a user-provided model name to a kie.ai model ID."""
    if model in MODEL_MAP:
        return MODEL_MAP[model]
    # If it's already a raw kie.ai model ID, pass through
    if "/" in model or model in KIE_TO_FRIENDLY.values():
        return model
    # Try word-based fuzzy match
    low = model.lower().replace("-", " ").replace("_", " ")
    for friendly, kie_id in MODEL_MAP.items():
        if low in friendly.lower() or low in kie_id.lower():
            return kie_id
    return model  # pass through


# ── OpenAI-format Pydantic models ──────────────────────────────

class ImageGenerationRequest(BaseModel):
    model: str = Field(default="z-image", description="Model name")
    prompt: str = Field(..., description="Text description of the desired image")
    n: int = Field(default=1, ge=1, le=4, description="Number of images to generate")
    size: Optional[str] = Field(default=None, description="Image size, e.g. '1024x1024'")
    negative_prompt: Optional[str] = Field(default=None)
    response_format: Optional[str] = Field(default="url", description="'url' or 'b64_json'")

    model_config = {"extra": "allow"}


class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]


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


# ── Helpers ─────────────────────────────────────────────────────

# Standard aspect ratios supported by most kie.ai models
STANDARD_RATIOS = {
    "1:1": "1:1",
    "4:3": "4:3",
    "3:4": "3:4",
    "16:9": "16:9",
    "9:16": "9:16",
}


def _size_to_aspect_ratio(size: Optional[str]) -> str:
    """Convert OpenAI-style '1024x1024' to nearest standard kie.ai aspect ratio."""
    if not size:
        return "1:1"
    parts = size.lower().split("x")
    if len(parts) == 2:
        try:
            w, h = int(parts[0]), int(parts[1])
            ratio = w / h
            # Map to closest standard ratio
            closest = min(STANDARD_RATIOS, key=lambda r: abs(
                ratio - int(r.split(":")[0]) / int(r.split(":")[1])
            ))
            return closest
        except ValueError:
            pass
    return "1:1"


def _kie_result_to_openai_images(data: dict,
                                  response_format: str = "url",
                                  n: int = 1) -> list[dict]:
    """Extract image URLs from kie.ai result and return OpenAI-format data list."""
    result_str = data.get("resultJson") or "{}"
    if isinstance(result_str, str):
        result = json.loads(result_str)
    else:
        result = result_str

    urls = result.get("resultUrls") or []
    images = []

    for i, url in enumerate(urls[:n]):
        if response_format == "b64_json":
            # Download and base64-encode
            try:
                import httpx
                resp = httpx.get(url, timeout=30)
                resp.raise_for_status()
                b64 = base64.b64encode(resp.content).decode("utf-8")
                images.append({"b64_json": b64})
            except Exception as e:
                logger.warning("Failed to download image %s: %s", url, e)
                images.append({"url": url})
        else:
            images.append({"url": url})

    # If no URLs found, try direct url field
    if not images:
        direct_url = data.get("url") or data.get("output_url") or result.get("url")
        if direct_url:
            images.append({"url": direct_url})

    return images


# ── Routes ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    if kie_client and settings.kie_api_key and settings.kie_api_key not in ("", "your-kie-api-key-here"):
        try:
            credit = await kie_client.get_credit()
            return {"status": "ok", "credit": credit}
        except Exception as e:
            return {"status": "degraded", "error": str(e)}
    return {"status": "ok", "warning": "KIE_API_KEY not configured"}


@app.get("/v1/models")
async def list_models(request: Request):
    """Return models available through this relay (OpenAI-compatible)."""
    await verify_auth(request)
    models = []
    for friendly_name in MODEL_MAP:
        models.append({
            "id": friendly_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "kie-relay",
        })
    return {"object": "list", "data": models}


@app.post("/v1/images/generations")
async def create_image(request: Request, body: ImageGenerationRequest):
    """Generate image(s) — translates OpenAI format → kie.ai createTask."""
    await verify_auth(request)

    if not kie_client:
        raise HTTPException(status_code=503, detail="kie.ai client not initialized")

    # Resolve model
    kie_model = resolve_model(body.model)
    logger.info("Image generation: user_model=%s → kie_model=%s prompt=%s",
                body.model, kie_model, body.prompt[:80])

    # Build kie.ai input
    input_data = {
        "prompt": body.prompt,
    }
    if body.negative_prompt:
        input_data["negative_prompt"] = body.negative_prompt

    # Map size → aspect_ratio (kie.ai 大部分模型要求必传)
    input_data["aspect_ratio"] = _size_to_aspect_ratio(body.size)

    # Pass n (number of images) to kie.ai
    if body.n > 1:
        input_data["num_images"] = str(body.n)

    # Pass through extra model-specific params
    extra = getattr(body, 'model_extra', None) or {}
    for key in ("image_size", "style", "seed", "guidance_scale",
                 "nsfw_checker", "expand_prompt", "rendering_speed"):
        if key in extra and extra[key] is not None:
            input_data[key] = extra[key]

    # Create and poll
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
        raise HTTPException(status_code=502, detail=f"kie.ai HTTP error: {e.response.text}")

    # Convert back to OpenAI format
    images = _kie_result_to_openai_images(result, body.response_format, body.n)

    if not images:
        raise HTTPException(
            status_code=502,
            detail="No image URLs in kie.ai response. Check task logs.",
        )

    return ImageGenerationResponse(
        created=int(time.time()),
        data=[ImageData(**img) for img in images],
    ).model_dump()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """
    Chat completions endpoint.

    For kie.ai's image-generation models, this endpoint extracts the last
    user message as the prompt and generates an image (convenience wrapper).
    For future LLM models on kie.ai, this will forward streaming completions.
    """
    await verify_auth(request)
    if not kie_client:
        raise HTTPException(status_code=503, detail="kie.ai client not initialized")

    # Get the last user message content as the prompt
    prompt = ""
    for msg in reversed(body.messages):
        if msg.role == "user":
            prompt = msg.content
            break

    if not prompt:
        raise HTTPException(status_code=400, detail="No user message found")

    kie_model = resolve_model(body.model)
    logger.info("Chat completion → image gen: model=%s prompt=%s",
                body.model, prompt[:80])

    try:
        task_id = await kie_client.create_task(kie_model, {"prompt": prompt})
        result = await kie_client.poll_task(
            task_id,
            interval=settings.poll_interval,
            timeout=settings.poll_timeout,
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    images = _kie_result_to_openai_images(result, "url", 1)
    image_url = images[0]["url"] if images else ""

    # Return as a markdown image in the assistant message
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"![generated image]({image_url})"
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# ── Run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
