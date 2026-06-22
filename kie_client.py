"""kie.ai API client - handles task creation, polling, and result retrieval."""

import json
import time
import logging
from typing import Optional

import httpx

logger = logging.getLogger("kie_relay")


class KieClient:
    """Asynchronous client for the kie.ai task-based API."""

    def __init__(self, api_key: str, base_url: str = "https://api.kie.ai"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self):
        await self._client.aclose()

    # ── Task creation ──────────────────────────────────────────────

    async def create_task(self, model: str, input_data: dict,
                          callback_url: Optional[str] = None) -> str:
        """Create a task on kie.ai and return the task_id."""
        payload = {
            "model": model,
            "input": input_data,
        }
        if callback_url:
            payload["callBackUrl"] = callback_url

        has_ref = "image_urls" in input_data or "input_urls" in input_data
        logger.info("Creating kie.ai task: model=%s ref_img=%s input_keys=%s",
                     model, has_ref, list(input_data.keys()))
        resp = await self._client.post("/api/v1/jobs/createTask", json=payload)
        body = resp.json()

        if resp.status_code != 200 or body.get("code") != 200:
            raise RuntimeError(
                f"kie.ai createTask failed (HTTP {resp.status_code}): "
                f"{body.get('msg', body)}"
            )

        task_id: str = body["data"]["taskId"]
        logger.info("Task created: %s", task_id)
        return task_id

    # ── Polling ────────────────────────────────────────────────────

    async def poll_task(self, task_id: str,
                        interval: float = 2.0,
                        timeout: float = 120.0) -> dict:
        """Poll recordInfo until the task completes or times out.

        Returns the full ``data`` dict from the response.
        Raises TimeoutError or RuntimeError on failure.
        """
        deadline = time.monotonic() + timeout
        last_progress = 0.0

        while time.monotonic() < deadline:
            resp = await self._client.get(
                "/api/v1/jobs/recordInfo",
                params={"taskId": task_id},
            )
            body = resp.json()

            if resp.status_code != 200 or body.get("code") != 200:
                logger.warning("recordInfo HTTP error: %s", body)
                await self._sleep(interval)
                continue

            data = body.get("data", {})
            state = (data.get("state") or "").lower()

            logger.debug("Task %s state=%s", task_id, state)

            # ── Terminal states ──
            if state == "success":
                logger.info("Task %s completed successfully", task_id)
                return data

            if state in ("fail", "failed", "error",
                         "create_task_failed", "generate_failed"):
                err = data.get("failMsg") or data.get("errorMessage") or data.get("failCode") or "unknown error"
                raise RuntimeError(f"Task {task_id} failed: {err}")

            # ── Still running ──
            await self._sleep(interval)

        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

    # ── Helpers ────────────────────────────────────────────────────

    async def get_credit(self) -> int:
        """Return remaining credit balance."""
        resp = await self._client.get("/api/v1/chat/credit")
        body = resp.json()
        if resp.status_code != 200 or body.get("code") != 200:
            raise RuntimeError(f"Failed to get credit: {body}")
        return int(body["data"])

    async def get_download_url(self, url: str) -> str:
        """Get a temporary (20-minute) download URL for a file."""
        resp = await self._client.post(
            "/api/v1/common/download-url",
            json={"url": url},
        )
        body = resp.json()
        if resp.status_code != 200 or body.get("code") != 200:
            raise RuntimeError(f"Failed to get download URL: {body}")
        return str(body["data"])

    @staticmethod
    async def _sleep(seconds: float):
        import asyncio
        await asyncio.sleep(seconds)

    # ── Image download helper ──────────────────────────────────────

    async def download_image(self, url: str) -> bytes:
        """Download raw image bytes from a URL."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=30)
            resp.raise_for_status()
            return resp.content
