"""
kie.ai Adapter — pure translation layer for One API.

Translates OpenAI-format requests to kie.ai's task-based API.
No user management, no billing — One API handles all of that.

Endpoints:
  POST /v1/images/generations  → Image generation
  GET  /v1/models              → List models
  GET  /health                 → Health check
"""

import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
logger = logging.getLogger("kie_adapter")

# ── Config ────────────────────────────────────────────────────
KIE_API_KEY = ""
KIE_API_BASE = "https://api.kie.ai"
POLL_INTERVAL = 1.0
POLL_TIMEOUT = 55.0

import os
KIE_API_KEY = os.environ.get("KIE_API_KEY", "")
KIE_API_BASE = os.environ.get("KIE_API_BASE", "https://api.kie.ai")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "1"))
POLL_TIMEOUT = float(os.environ.get("POLL_TIMEOUT", "55"))

# ── HTTP client ──────────────────────────────────────────────
_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    if not KIE_API_KEY:
        logger.warning("KIE_API_KEY not set")
    _client = httpx.AsyncClient(
        base_url=KIE_API_BASE,
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={
            "Authorization": f"Bearer {KIE_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    logger.info("kie.ai adapter started")
    yield
    await _client.aclose()


app = FastAPI(title="kie.ai Adapter", version="1.0.0", lifespan=lifespan)

# ── Model mapping ────────────────────────────────────────────
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
    "google/nano-banana-pro": "google/nano-banana",
    "nano-banana-pro": "google/nano-banana",
    "google/nano-banana": "google/nano-banana",
    "nano-banana": "google/nano-banana",
    "google/nano-banana-edit": "google/nano-banana-edit",
    "nano-banana-edit": "google/nano-banana-edit",
    "gpt-image-2-text-to-image": "gpt-image-2-text-to-image",
    "gpt-image-2": "gpt-image-2-text-to-image",
    "gpt-image-2-image-to-image": "gpt-image-2-image-to-image",
    "hailuo/text-to-video": "hailuo/02-text-to-video-pro",
    "hailuo/image-to-video": "hailuo/02-image-to-video-standard",
    "kling/v2.1-standard": "kling/v2-1-standard",
    "kling/v2.1-turbo": "kling/v2-1-turbo",
}

STANDARD_RATIOS = {"1:1": "1:1", "4:3": "4:3", "3:4": "3:4", "16:9": "16:9", "9:16": "9:16"}

# ── Pydantic models ──────────────────────────────────────────

class ImageRequest(BaseModel):
    model: str = "z-image"
    prompt: str
    n: int = Field(default=1, ge=1, le=4)
    size: Optional[str] = None
    negative_prompt: Optional[str] = None
    image: Optional[str] = Field(default=None, description="Image URL for image-to-image (图生图)")
    image_url: Optional[str] = Field(default=None, description="Alias for image field")
    response_format: Optional[str] = "url"
    model_config = {"extra": "allow"}


class ChatRequest(BaseModel):
    model: str = "z-image"
    messages: list[dict]
    model_config = {"extra": "allow"}


# ── Helpers ──────────────────────────────────────────────────

def resolve_model(model: str) -> str:
    if model in MODEL_MAP:
        return MODEL_MAP[model]
    return model


def size_to_ratio(size: Optional[str]) -> str:
    if not size:
        return "1:1"
    parts = size.lower().split("x")
    if len(parts) == 2:
        try:
            w, h = int(parts[0]), int(parts[1])
            ratio = w / h
            return min(STANDARD_RATIOS, key=lambda r: abs(
                ratio - int(r.split(":")[0]) / int(r.split(":")[1])
            ))
        except ValueError:
            pass
    return "1:1"


def parse_result(data: dict, fmt: str = "url", n: int = 1) -> list[dict]:
    result_str = data.get("resultJson") or "{}"
    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    urls = result.get("resultUrls") or []
    images = []
    for url in urls[:n]:
        if fmt == "b64_json":
            try:
                resp = httpx.get(url, timeout=30)
                resp.raise_for_status()
                b64 = base64.b64encode(resp.content).decode("utf-8")
                images.append({"b64_json": b64})
            except Exception:
                images.append({"url": url})
        else:
            images.append({"url": url})
    if not images:
        u = data.get("url") or data.get("output_url") or result.get("url")
        if u:
            images.append({"url": u})
    return images


async def call_kie(model: str, input_data: dict) -> dict:
    """Create task and poll for result."""
    if not _client:
        raise HTTPException(503, "Not initialized")

    # Support model override (e.g. nano-banana-pro -> nano-banana-edit for img2img)
    override_model = input_data.pop("_override_model", None)
    actual_model = override_model or resolve_model(model)

    resp = await _client.post("/api/v1/jobs/createTask", json={
        "model": actual_model,
        "input": input_data,
    })
    body = resp.json()
    if resp.status_code != 200 or body.get("code") != 200:
        raise HTTPException(502, f"kie.ai error: {body.get('msg', body)}")

    task_id = body["data"]["taskId"]
    logger.info("Task created: %s", task_id)

    # Poll
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        resp = await _client.get("/api/v1/jobs/recordInfo", params={"taskId": task_id})
        body = resp.json()
        if resp.status_code != 200 or body.get("code") != 200:
            await _sleep(POLL_INTERVAL)
            continue
        data = body.get("data", {})
        state = (data.get("state") or "").lower()

        if state == "success":
            return data
        if state in ("fail", "failed", "error"):
            raise HTTPException(502, f"Task failed: {data.get('failMsg', 'unknown')}")

        await _sleep(POLL_INTERVAL)

    raise HTTPException(504, "Timeout waiting for generation")


async def _sleep(s):
    import asyncio
    await asyncio.sleep(s)


# ── Routes ───────────────────────────────────────────────────

@app.get("/health")
async def health():
    if _client and KIE_API_KEY:
        try:
            r = await _client.get("/api/v1/chat/credit")
            body = r.json()
            return {"status": "ok", "credit": body.get("data", 0)}
        except Exception as e:
            return {"status": "degraded", "error": str(e)}
    return {"status": "ok", "warning": "KIE_API_KEY not set"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": int(time.time()), "owned_by": "kie-adapter"}
            for name in MODEL_MAP
        ],
    }


@app.post("/v1/images/generations")
async def create_image(body: ImageRequest):
    if not _client:
        raise HTTPException(503, "Not initialized")

    input_data = {"prompt": body.prompt, "aspect_ratio": size_to_ratio(body.size)}
    if body.negative_prompt:
        input_data["negative_prompt"] = body.negative_prompt
    if body.n > 1:
        input_data["num_images"] = str(body.n)

    # ── Image-to-image (图生图) support ──────────────────────
    ref_image_url = body.image or body.image_url or None
    if ref_image_url:
        resolved = resolve_model(body.model)
        if resolved == "google/nano-banana":
            # nano-banana-pro 图生图 → nano-banana-edit
            input_data["image_urls"] = [ref_image_url]
            input_data["_override_model"] = "google/nano-banana-edit"
            logger.info("Switched to nano-banana-edit for image-to-image")
        elif resolved == "gpt-image-2-text-to-image":
            # gpt-image-2 图生图 → gpt-image-2-image-to-image (input_urls)
            input_data["input_urls"] = [ref_image_url]
            input_data["aspect_ratio"] = "auto"
            input_data["_override_model"] = "gpt-image-2-image-to-image"
            input_data.pop("image_urls", None)
            logger.info("Switched to gpt-image-2-image-to-image for image-to-image")
        else:
            input_data["image_urls"] = [ref_image_url]
            logger.info("Using generic image_urls for model=%s", resolved)

    for key in ("image_size", "style", "seed", "guidance_scale", "nsfw_checker"):
        extra = body.model_extra or {}
        if key in extra:
            input_data[key] = extra[key]

    result = await call_kie(body.model, input_data)
    images = parse_result(result, body.response_format, body.n)
    if not images:
        raise HTTPException(502, "No image in response")

    return {"created": int(time.time()), "data": images}


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest):
    if not _client:
        raise HTTPException(503, "Not initialized")

    prompt = ""
    for msg in reversed(body.messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            prompt = content if isinstance(content, str) else json.dumps(content)
            break
    if not prompt:
        raise HTTPException(400, "No user message")

    result = await call_kie(body.model, {"prompt": prompt})
    images = parse_result(result, "url", 1)
    url = images[0]["url"] if images else ""

    return {
        "id": f"chatcmpl-{time.time():.0f}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": f"![image]({url})"},
            "finish_reason": "stop",
        }],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("adapter:app", host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), log_level="info")
