"""Tencent Cloud COS (Object Storage) client for kie-relay.

Handles:
- Uploading files to COS and returning public URLs
- (Optional) Saving generated images to COS for persistence
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Optional

from qcloud_cos import CosConfig, CosS3Client

logger = logging.getLogger("kie_relay")

# ── Defaults from environment (read dynamically, not at import time) ──
COS_REGION = os.environ.get("COS_REGION", "ap-guangzhou")
COS_BUCKET = os.environ.get("COS_BUCKET", "n8n-results-1302052432")
COS_PUBLIC_DOMAIN = os.environ.get(
    "COS_PUBLIC_DOMAIN",
    f"https://{COS_BUCKET}.cos.{COS_REGION}.myqcloud.com",
)

_client: Optional[CosS3Client] = None


def _get_secret_id() -> str:
    return os.environ.get("COS_SECRET_ID", "")


def _get_secret_key() -> str:
    return os.environ.get("COS_SECRET_KEY", "")


def get_client() -> Optional[CosS3Client]:
    """Lazy-init and return the COS client."""
    global _client
    if _client is not None:
        return _client
    sid = _get_secret_id()
    skey = _get_secret_key()
    if not sid or not skey:
        logger.warning("COS_SECRET_ID / COS_SECRET_KEY not configured")
        return None
    config = CosConfig(
        Region=COS_REGION,
        SecretId=sid,
        SecretKey=skey,
        Scheme="https",
    )
    _client = CosS3Client(config)
    logger.info("COS client initialized: bucket=%s region=%s", COS_BUCKET, COS_REGION)
    return _client


def is_configured() -> bool:
    return bool(_get_secret_id() and _get_secret_key())


def upload_file(
    file_bytes: bytes,
    filename: str = "",
    content_type: str = "image/png",
    subdir: str = "uploads",
) -> Optional[str]:
    """Upload bytes to COS and return the public URL.

    Args:
        file_bytes: Raw file content.
        filename: Original filename (used for extension detection).
        content_type: MIME type.
        subdir: Subdirectory inside the bucket (``uploads``, ``images``, etc.).

    Returns:
        Public URL of the uploaded file, or ``None`` on failure.
    """
    client = get_client()
    if client is None:
        return None

    # Generate a unique object key
    ext = os.path.splitext(filename)[1] if filename else ".png"
    if ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        ext = ".png"
    object_key = f"{subdir}/{datetime.now().strftime('%Y%m%d')}/{uuid.uuid4().hex}{ext}"

    try:
        client.put_object(
            Bucket=COS_BUCKET,
            Body=file_bytes,
            Key=object_key,
            ContentType=content_type,
            ACL="public-read",  # 必须公开读，否则 kie.ai 服务器无法下载参考图
        )
        url = f"{COS_PUBLIC_DOMAIN}/{object_key}"
        logger.info("COS upload OK: %s (%d bytes)", url, len(file_bytes))
        return url
    except Exception as e:
        logger.error("COS upload failed: %s", e)
        return None


def upload_from_url(source_url: str, subdir: str = "images") -> Optional[str]:
    """Download from a URL and re-upload to COS for persistence.

    Useful for saving generated images (kie.ai temp URLs expire).
    """
    import httpx

    try:
        resp = httpx.get(source_url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/png")
        # Derive extension from content-type
        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }
        ext = ext_map.get(content_type.split(";")[0], ".png")
        filename = f"generated{ext}"
        return upload_file(resp.content, filename=filename, content_type=content_type, subdir=subdir)
    except Exception as e:
        logger.error("COS upload_from_url failed for %s: %s", source_url[:80], e)
        return None


def delete_file(url: str) -> bool:
    """Delete a file from COS by its public URL.

    Returns True on success, False on failure.
    """
    client = get_client()
    if client is None:
        return False

    # Extract object key from URL
    prefix = f"{COS_PUBLIC_DOMAIN}/"
    if not url.startswith(prefix):
        logger.warning("Cannot delete: URL not in COS bucket (%s)", url[:60])
        return False
    object_key = url[len(prefix):]

    try:
        client.delete_object(Bucket=COS_BUCKET, Key=object_key)
        logger.info("COS delete OK: %s", object_key)
        return True
    except Exception as e:
        logger.error("COS delete failed: %s", e)
        return False
