"""
SmartAPI — Angel One broker connection layer.

Exports:
    SmartAPIClient  — main client (auth + WebSocket + price cache)
    TOTPGenerator   — TOTP generation and validation
    RateLimiter     — single-endpoint token-bucket limiter
    MultiRateLimiter — all SmartAPI endpoints in one object

Example:
    from smartapi import SmartAPIClient

    with SmartAPIClient() as client:
        client.start_websocket()
        client.wait_for_websocket()
        client.subscribe_symbols([{"exchangeType": 1, "tokens": ["2885"]}])
        # your logic here
"""

from smartapi.totp_generator import TOTPGenerator, generate_totp_from_secret
from smartapi.rate_limiter import RateLimiter, MultiRateLimiter
from smartapi.smartapi_client import SmartAPIClient

__version__ = "2.0.0"

__all__ = [
    "SmartAPIClient",
    "TOTPGenerator",
    "generate_totp_from_secret",
    "RateLimiter",
    "MultiRateLimiter",
]