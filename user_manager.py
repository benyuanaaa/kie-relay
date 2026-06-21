"""User management with JSON file storage for billing."""

import json
import os
import time
import logging
from typing import Optional

logger = logging.getLogger("kie_relay")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
USAGE_FILE = os.path.join(DATA_DIR, "usage.json")

# Model pricing (in kie.ai credits, + our markup)
# Format: model_pattern -> cost_multiplier
MODEL_PRICES = {
    "z-image": 1.0,
    "google/imagen4": 3.0,
    "google/imagen4-fast": 2.0,
    "ideogram": 2.0,
    "seedream": 1.5,
    "grok-imagine": 2.0,
    "recraft": 2.0,
    "flux": 2.0,
    "sdxl": 1.0,
    "hailuo": 5.0,
    "kling": 4.0,
}


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_json(path: str) -> dict:
    _ensure_data_dir()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load %s: %s", path, e)
        return {}


def _save_json(path: str, data: dict):
    _ensure_data_dir()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def get_model_cost(model: str) -> float:
    """Return the cost in credits for a given model."""
    for pattern, price in MODEL_PRICES.items():
        if pattern in model.lower():
            return price
    return 2.0  # default cost


# ── User operations ──────────────────────────────────────────

def create_user(name: str, initial_balance: float = 10.0) -> dict:
    """Create a new user with a generated API key."""
    import secrets
    api_key = "kie-" + secrets.token_hex(16)
    users = _load_json(USERS_FILE)
    user = {
        "name": name,
        "api_key": api_key,
        "balance": initial_balance,
        "total_used": 0.0,
        "total_calls": 0,
        "enabled": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    users[api_key] = user
    _save_json(USERS_FILE, users)
    logger.info("User created: %s (key=%s, balance=%.1f)", name, api_key[:12]+"...", initial_balance)
    return user


def get_user_by_key(api_key: str) -> Optional[dict]:
    """Look up a user by their API key."""
    users = _load_json(USERS_FILE)
    return users.get(api_key)


def topup_user(api_key: str, amount: float) -> Optional[dict]:
    """Add balance to a user. Returns updated user or None if not found."""
    users = _load_json(USERS_FILE)
    user = users.get(api_key)
    if not user:
        return None
    user["balance"] += amount
    _save_json(USERS_FILE, users)
    logger.info("Topup: key=%s amount=%.1f new_balance=%.1f",
                api_key[:12]+"...", amount, user["balance"])
    return user


def deduct_balance(api_key: str, cost: float, model: str) -> bool:
    """Deduct cost from user balance. Returns False if insufficient funds."""
    users = _load_json(USERS_FILE)
    user = users.get(api_key)
    if not user or user.get("enabled") is False:
        return False
    if user["balance"] < cost:
        return False

    user["balance"] -= cost
    user["total_used"] += cost
    user["total_calls"] += 1
    user["last_used_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save_json(USERS_FILE, users)

    # Log usage
    log_usage(api_key, model, cost)
    return True


def list_users() -> list[dict]:
    """Return all users (without exposing full keys)."""
    users = _load_json(USERS_FILE)
    result = []
    for key, user in users.items():
        result.append({
            "name": user["name"],
            "key_prefix": key[:12] + "...",
            "balance": user["balance"],
            "total_used": user["total_used"],
            "total_calls": user["total_calls"],
            "enabled": user.get("enabled", True),
            "created_at": user.get("created_at", ""),
            "last_used_at": user.get("last_used_at", ""),
        })
    return result


# ── Usage logging ────────────────────────────────────────────

def log_usage(api_key: str, model: str, cost: float):
    """Append a usage record."""
    usage = _load_json(USAGE_FILE)
    records = usage.get("records", [])
    records.append({
        "key_prefix": api_key[:12] + "...",
        "model": model,
        "cost": cost,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    # Keep only last 10000 records
    if len(records) > 10000:
        records = records[-10000:]
    usage["records"] = records
    _save_json(USAGE_FILE, usage)


def get_usage_summary() -> dict:
    """Return usage statistics."""
    usage = _load_json(USAGE_FILE)
    records = usage.get("records", [])
    total_calls = len(records)
    total_cost = sum(r["cost"] for r in records)
    # Per-model breakdown
    by_model = {}
    for r in records:
        m = r["model"]
        by_model[m] = by_model.get(m, 0) + r["cost"]
    return {
        "total_calls": total_calls,
        "total_cost": round(total_cost, 1),
        "by_model": {k: round(v, 1) for k, v in sorted(by_model.items(), key=lambda x: -x[1])},
    }
