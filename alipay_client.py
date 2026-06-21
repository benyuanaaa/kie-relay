"""
Alipay (支付宝) payment client for 当面付 (Face-to-Face).

Endpoints used:
  - alipay.trade.precreate  →  generate payment QR code
  - alipay.trade.query      →  check payment status (fallback)

Requires cryptography library.
"""

import base64
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote, urlencode

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

logger = logging.getLogger("kie_relay")

ALIPAY_GATEWAY = "https://openapi.alipay.com/gateway.do"
ALIPAY_SANDBOX_GATEWAY = "https://openapi-sandbox.dl.alipaydev.com/gateway.do"


class AlipayClient:
    """Alipay payment client for trade.precreate (当面付)."""

    def __init__(
        self,
        app_id: str,
        private_key_path: str,
        alipay_public_key_path: str,
        notify_url: str,
        sandbox: bool = False,
    ):
        self.app_id = app_id
        self.gateway = ALIPAY_SANDBOX_GATEWAY if sandbox else ALIPAY_GATEWAY
        self.notify_url = notify_url.rstrip("/") + "/api/alipay/notify"

        # Load keys
        with open(private_key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)

        with open(alipay_public_key_path, "rb") as f:
            self._alipay_public_key = serialization.load_pem_public_key(f.read())

        self._http = httpx.AsyncClient(timeout=15)

    async def close(self):
        await self._http.aclose()

    # ── Core: sign and verify ─────────────────────────────────

    def _sign(self, params: dict) -> str:
        """Sign params with RSA2 (SHA-256)."""
        content = self._build_sign_content(params)
        signature = self._private_key.sign(
            content.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _verify(self, params: dict, signature: str) -> bool:
        """Verify Alipay's response signature."""
        content = self._build_sign_content(params)
        try:
            self._alipay_public_key.verify(
                base64.b64decode(signature),
                content.encode("utf-8"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _build_sign_content(params: dict) -> str:
        """Build the string to sign: sort keys, URL-encode, join."""
        keys = sorted(k for k in params if k not in ("sign", "sign_type"))
        parts = []
        for k in keys:
            v = params[k]
            if v is None or v == "":
                continue
            parts.append(f"{k}={quote(str(v), safe='')}")
        return "&".join(parts)

    def _build_request_params(self, method: str, biz_content: dict) -> dict:
        """Build common request parameters."""
        params = {
            "app_id": self.app_id,
            "method": method,
            "format": "JSON",
            "charset": "utf-8",
            "sign_type": "RSA2",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            "biz_content": json.dumps(biz_content, ensure_ascii=False),
        }
        if method in ("alipay.trade.precreate",):
            params["notify_url"] = self.notify_url
        params["sign"] = self._sign(params)
        return params

    # ── Payments ─────────────────────────────────────────────

    async def create_qr_payment(
        self, order_id: str, amount: float, subject: str = "API额度充值"
    ) -> dict:
        """
        Create a payment QR code via alipay.trade.precreate.

        Returns:
            {
                "qr_code": "https://qr.alipay.com/xxx",  # payment URL
                "out_trade_no": "ORDER_xxx",
                "total_amount": 10.0,
            }
        Raises RuntimeError on failure.
        """
        biz = {
            "out_trade_no": order_id,
            "total_amount": f"{amount:.2f}",
            "subject": subject,
            "qr_code_timeout_express": "30m",  # 30 min expiry
        }
        params = self._build_request_params("alipay.trade.precreate", biz)
        params_str = urlencode(params, doseq=True)

        resp = await self._http.post(
            self.gateway,
            data=params_str,
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
        )
        body = resp.json()

        # Verify signature in response
        resp_sign = body.get("sign", "")
        resp_params = body.get("alipay_trade_precreate_response", {})

        if not self._verify(resp_params, resp_sign):
            raise RuntimeError("支付宝响应签名验证失败")

        code = resp_params.get("code")
        if code != "10000":
            msg = resp_params.get("msg", "") + ": " + resp_params.get("sub_msg", "")
            raise RuntimeError(f"支付宝创建支付失败: {msg}")

        qr_code = resp_params.get("qr_code", "")
        if not qr_code:
            raise RuntimeError("支付宝未返回二维码")

        logger.info("Payment created: order=%s amount=%.2f", order_id, amount)
        return {
            "qr_code": qr_code,
            "out_trade_no": resp_params["out_trade_no"],
            "total_amount": float(resp_params["total_amount"]),
        }

    async def query_payment(self, order_id: str) -> dict:
        """
        Query payment status via alipay.trade.query.

        Returns:
            {
                "trade_status": "WAIT_BUYER_PAY" | "TRADE_SUCCESS" | "TRADE_CLOSED",
                "out_trade_no": "...",
                "total_amount": 10.0,
                "receipt_amount": 10.0,
            }
        """
        biz = {"out_trade_no": order_id}
        params = self._build_request_params("alipay.trade.query", biz)

        resp = await self._http.post(
            self.gateway,
            data=urlencode(params, doseq=True),
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
        )
        body = resp.json()
        resp_params = body.get("alipay_trade_query_response", {})
        resp_sign = body.get("sign", "")

        if not self._verify(resp_params, resp_sign):
            raise RuntimeError("支付宝查询响应签名验证失败")

        return {
            "trade_status": resp_params.get("trade_status", "UNKNOWN"),
            "out_trade_no": resp_params.get("out_trade_no", ""),
            "total_amount": float(resp_params.get("total_amount", 0)),
            "receipt_amount": float(resp_params.get("receipt_amount", 0)),
            "buyer_id": resp_params.get("buyer_id", ""),
        }

    # ── Notification verification ────────────────────────────

    def verify_notification(self, form_data: dict) -> bool:
        """
        Verify the async notification from Alipay.
        Returns True if the notification is valid.
        """
        sign = form_data.get("sign", "")
        if not sign:
            return False
        # Remove sign and sign_type before verification
        params = {k: v for k, v in form_data.items() if k not in ("sign", "sign_type")}
        return self._verify(params, sign)

    def parse_notification(self, form_data: dict) -> Optional[dict]:
        """
        Parse Alipay notification.
        Returns structured data if notification is valid and trade is successful.
        """
        if not self.verify_notification(form_data):
            logger.warning("Alipay notification signature verification failed")
            return None

        trade_status = form_data.get("trade_status", "")
        if trade_status != "TRADE_SUCCESS":
            logger.info("Alipay notification: status=%s", trade_status)
            return None

        return {
            "out_trade_no": form_data.get("out_trade_no", ""),
            "trade_no": form_data.get("trade_no", ""),
            "total_amount": float(form_data.get("total_amount", 0)),
            "receipt_amount": float(form_data.get("receipt_amount", 0)),
            "buyer_id": form_data.get("buyer_id", ""),
            "gmt_payment": form_data.get("gmt_payment", ""),
        }
