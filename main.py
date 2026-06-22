"""
kie.ai Relay — OpenAI-compatible API proxy for kie.ai.

Endpoints:
  Public:
    GET  /health                 → Health check
    GET  /v1/models              → List available models
    POST /v1/images/generations  → Image generation (with billing)
    POST /v1/chat/completions    → Chat completion (with billing)
    GET  /v1/me                  → Check my balance

  Upload (requires API key):
    POST /api/upload             → Upload image to Tencent COS

  Admin (requires RELAY_API_KEY):
    POST /admin/users            → Create user
    GET  /admin/users            → List users
    POST /admin/users/topup      → Add balance
    GET  /admin/usage            → Usage statistics
"""

import base64
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from kie_client import KieClient
import user_manager
import orders as order_manager
import cos_client

# Alipay (may not be configured)
alipay_client: Optional["AlipayClient"] = None

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
    global kie_client, alipay_client
    if not settings.kie_api_key or settings.kie_api_key in ("", "your-kie-api-key-here"):
        logger.warning("KIE_API_KEY not configured")
    kie_client = KieClient(
        api_key=settings.kie_api_key,
        base_url=settings.kie_api_base,
    )

    # Init Alipay if configured
    if settings.alipay_app_id and settings.alipay_app_id not in ("", "your-app-id"):
        try:
            from alipay_client import AlipayClient
            global alipay_client  # noqa: PLW0602
            alipay_client = AlipayClient(
                app_id=settings.alipay_app_id,
                private_key_path=settings.alipay_private_key_path,
                alipay_public_key_path=settings.alipay_public_key_path,
                notify_url=settings.alipay_notify_url,
                sandbox=settings.alipay_sandbox,
            )
            logger.info("Alipay initialized: app_id=%s", settings.alipay_app_id)
        except Exception as e:
            logger.warning("Alipay init failed: %s (skip)", e)
    else:
        logger.info("Alipay not configured (skip)")

    logger.info("kie.ai relay started on %s:%s", settings.host, settings.port)

    # Init Tencent COS if configured
    if settings.cos_secret_id and settings.cos_secret_id not in ("", "your-cos-secret-id-here"):
        import os as _os
        _os.environ["COS_SECRET_ID"] = settings.cos_secret_id
        _os.environ["COS_SECRET_KEY"] = settings.cos_secret_key
        _os.environ["COS_REGION"] = settings.cos_region
        _os.environ["COS_BUCKET"] = settings.cos_bucket
        if settings.cos_public_domain:
            _os.environ["COS_PUBLIC_DOMAIN"] = settings.cos_public_domain
        cos_client.get_client()  # trigger init
        logger.info("COS configured: bucket=%s region=%s", settings.cos_bucket, settings.cos_region)
        app.state.cos_enabled = True
    else:
        logger.info("COS not configured - will use local file storage")
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        # Mount /uploads for local file serving
        from fastapi.staticfiles import StaticFiles
        app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
        app.state.cos_enabled = False

    yield
    await kie_client.close()
    if alipay_client:
        await alipay_client.close()


app = FastAPI(
    title="kie.ai Relay",
    version="0.3.0",
    description="OpenAI-compatible API proxy for kie.ai with billing",
    lifespan=lifespan,
)

# Serve the web UI and local uploads
STATIC_DIR = Path(__file__).parent / "static"
UPLOADS_DIR = Path(__file__).parent / "uploads"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

# Always mount /uploads for local fallback (created on-demand)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
# We mount lazily below after lifespan to know if COS is configured


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
    image: Optional[str] = Field(default=None, description="Image URL for image-to-image (图生图)")
    image_url: Optional[str] = Field(default=None, description="Alias for image field")
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


# Web page request models
class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=30,
                      description="User nickname")


class VerifyKeyRequest(BaseModel):
    api_key: str = Field(..., description="User's API key")


class RechargeRequest(BaseModel):
    api_key: str = Field(..., description="User's API key")
    amount: float = Field(..., ge=1, le=9999, description="Recharge amount (CNY)")


# ── Web UI routes ──────────────────────────────────────────────

@app.post("/api/register")
async def web_register(body: RegisterRequest):
    """Public registration: creates a user with free initial credits."""
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="昵称不能为空")

    # Give 10 free credits to new users
    user = user_manager.create_user(name, initial_balance=10.0)
    logger.info("New registration: name=%s key=%s", name, user["api_key"][:12]+"...")
    return {
        "message": "注册成功",
        "api_key": user["api_key"],
        "name": user["name"],
        "balance": user["balance"],
    }


@app.post("/api/verify-key")
async def web_verify_key(body: VerifyKeyRequest):
    """Verify an API key and return user info."""
    user = user_manager.get_user_by_key(body.api_key)
    if not user:
        raise HTTPException(status_code=404, detail="无效的 API Key")
    if not user.get("enabled", True):
        raise HTTPException(status_code=403, detail="该账号已被禁用")
    return {
        "name": user["name"],
        "balance": user["balance"],
        "total_used": user["total_used"],
        "total_calls": user["total_calls"],
    }

@app.post("/api/recharge")
async def web_recharge(body: RechargeRequest):
    """Create a payment order for the user to scan with Alipay."""
    user = user_manager.get_user_by_key(body.api_key)
    if not user:
        raise HTTPException(status_code=404, detail="Invalid API key")
    if not user.get("enabled", True):
        raise HTTPException(status_code=403, detail="Account disabled")
    if not alipay_client:
        raise HTTPException(status_code=503, detail="支付宝暂未开通")

    order = order_manager.create_order(body.api_key, body.amount)
    try:
        payment = await alipay_client.create_qr_payment(
            order["order_id"], body.amount
        )
        return {
            "order_id": order["order_id"],
            "qr_code_url": payment["qr_code"],
            "amount": body.amount,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/recharge/{order_id}")
async def web_recharge_status(order_id: str):
    """Check the status of a payment order."""
    order = order_manager.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return {
        "order_id": order["order_id"],
        "status": order["status"],
        "amount": order["amount"],
    }

@app.post("/api/alipay/notify")
async def web_alipay_notify(request: Request):
    """Alipay async notification callback."""
    global alipay_client
    form = await request.form()
    form_data = dict(form)

    if not alipay_client:
        logger.warning("Alipay not configured, ignoring notification")
        return "failure"

    result = alipay_client.parse_notification(form_data)
    if not result:
        logger.warning("Invalid Alipay notification")
        return "failure"

    # Complete the order and add balance
    order = order_manager.complete_order(result["out_trade_no"], result["trade_no"])
    if order:
        user_manager.topup_user(order["api_key"], result["receipt_amount"])
        logger.info(
            "Payment success: order=%s user_key=%s amount=%.2f",
            result["out_trade_no"],
            order["api_key"][:12] + "...",
            result["receipt_amount"],
        )
    return "success"


# ── COS Upload endpoint ──────────────────────────────────────────

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

@app.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Upload an image file to Tencent COS and return a public URL.

    Requires a valid API key (Bearer token).
    The returned URL can be used as ``image`` parameter for image-to-image generation.
    """
    await verify_user(request)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=413, detail="File too large (max 20MB)")

    # Detect content type
    content_type_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")

    url = cos_client.upload_file(content, filename=file.filename, content_type=content_type)
    if not url:
        # Fallback: save locally and serve via /uploads/
        logger.info("COS not configured, saving locally: %s", file.filename)
        local_filename = f"{uuid.uuid4().hex}{ext}"
        local_path = UPLOADS_DIR / local_filename
        with open(local_path, "wb") as f:
            f.write(content)
        # Construct full URL from the request host
        scheme = request.url.scheme
        host = request.headers.get("host", f"localhost:{settings.port}")
        url = f"{scheme}://{host}/uploads/{local_filename}"
        logger.info("Local file saved: %s", url[:80])

    logger.info("File uploaded: user=%s file=%s url=%s",
                request.state.user["name"], file.filename, url[:80])

    return {"url": url, "filename": file.filename, "size": len(content)}


# ── Aspect Ratio helpers ─────────────────────────────────────────
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

    # ── Image-to-image (图生图) support ──────────────────────
    ref_image_url = body.image or body.image_url or None
    if ref_image_url:
        logger.info("Image-to-image enabled: ref_url=%s", ref_image_url[:80])

        if kie_model == "google/nano-banana":
            # nano-banana-pro 图生图 → nano-banana-edit
            input_data["image_urls"] = [ref_image_url]
            kie_model = "google/nano-banana-edit"
            logger.info("Switched model: google/nano-banana -> google/nano-banana-edit")

        elif kie_model == "gpt-image-2-text-to-image":
            # gpt-image-2 图生图 → gpt-image-2-image-to-image
            # 参数名是 input_urls (不是 image_urls)，aspect_ratio 自动
            input_data["input_urls"] = [ref_image_url]
            input_data["aspect_ratio"] = "auto"
            # 去掉可能干扰的 image_urls
            input_data.pop("image_urls", None)
            kie_model = "gpt-image-2-image-to-image"
            logger.info("Switched model: gpt-image-2 -> gpt-image-2-image-to-image (input_urls)")

        else:
            # 其他模型：通用 image_urls
            input_data["image_urls"] = [ref_image_url]
            logger.info("Using generic image_urls for model=%s", kie_model)

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

    # ── Save generated images to COS for persistence ──────────
    # kie.ai temp URLs expire after ~20 min, so we re-upload to COS
    if cos_client.is_configured():
        for img in images:
            if img.get("url") and not img["url"].startswith(cos_client.COS_PUBLIC_DOMAIN):
                cos_url = cos_client.upload_from_url(img["url"], subdir="images")
                if cos_url:
                    img["cos_url"] = cos_url
                    img["url"] = cos_url  # replace temp URL with permanent COS URL

    return {
        "created": int(time.time()),
        "data": [{"url": img.get("url"), "b64_json": img.get("b64_json"),
                  "cos_url": img.get("cos_url")} for img in images],
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
