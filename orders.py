"""Simple order management for Alipay payments."""

import json
import logging
import os
import time

logger = logging.getLogger("kie_relay")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ORDERS_FILE = os.path.join(DATA_DIR, "orders.json")


def _ensure():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load() -> dict:
    _ensure()
    if not os.path.exists(ORDERS_FILE):
        return {"orders": [], "next_id": 1}
    try:
        with open(ORDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"orders": [], "next_id": 1}


def _save(data: dict):
    _ensure()
    tmp = ORDERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ORDERS_FILE)


def create_order(api_key: str, amount: float) -> dict:
    """Create a pending order."""
    data = _load()
    order_id = f"RCH{int(time.time())}{data['next_id']}"
    data["next_id"] += 1
    order = {
        "order_id": order_id,
        "api_key": api_key,
        "amount": amount,
        "status": "PENDING",  # PENDING → PAID → COMPLETED
        "alipay_trade_no": "",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paid_at": "",
    }
    data["orders"].append(order)
    _save(data)
    logger.info("Order created: %s amount=%.1f", order_id, amount)
    return order


def get_order(order_id: str) -> dict | None:
    """Get order by ID."""
    data = _load()
    for o in data["orders"]:
        if o["order_id"] == order_id:
            return o
    return None


def complete_order(order_id: str, alipay_trade_no: str) -> dict | None:
    """Mark order as completed and return the order."""
    data = _load()
    for o in data["orders"]:
        if o["order_id"] == order_id and o["status"] == "PENDING":
            o["status"] = "COMPLETED"
            o["alipay_trade_no"] = alipay_trade_no
            o["paid_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _save(data)
            logger.info("Order completed: %s alipay=%s", order_id, alipay_trade_no)
            return o
    return None


def get_user_orders(api_key: str, limit: int = 10) -> list:
    """Get recent orders for a user."""
    data = _load()
    user_orders = [o for o in data["orders"] if o["api_key"] == api_key]
    user_orders.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return user_orders[:limit]
