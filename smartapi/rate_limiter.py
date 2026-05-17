"""
Rate Limiter for Angel One SmartAPI

Token-bucket rate limiter. All numeric limits come from config.py.

SmartAPI limits:
  login      1 req/sec
  ltp       10 req/sec  (also 500/min)
  historical  3 req/sec
  orders      9 req/sec
"""

import time
import threading
import logging
from contextlib import contextmanager
from typing import Optional

from smartapi.config import (
    RATE_LIMIT_LOGIN,
    RATE_LIMIT_LTP,
    RATE_LIMIT_LTP_MINUTE,
    RATE_LIMIT_HISTORICAL,
    RATE_LIMIT_ORDERS,
    RATE_LIMIT_DEFAULT,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe token-bucket rate limiter for a single endpoint."""

    def __init__(self, max_requests: int, time_window: float = 1.0):
        """
        Args:
            max_requests: Tokens (requests) per time_window.
            time_window:  Window size in seconds.
        """
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if time_window <= 0:
            raise ValueError("time_window must be positive")

        self.max_requests = max_requests
        self.time_window  = time_window
        self.tokens       = float(max_requests)
        self.last_update  = time.time()
        self.refill_rate  = max_requests / time_window
        self._lock        = threading.RLock()

        logger.debug(f"RateLimiter: {max_requests} req / {time_window}s")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Refill tokens proportional to elapsed time. Call inside lock."""
        now     = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            float(self.max_requests),
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_update = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Block until a token is available, then consume it.

        Args:
            timeout: Give up after this many seconds (None = wait forever).

        Returns:
            True if a token was acquired, False if timeout.
        """
        start = time.time()
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
                wait = (1.0 - self.tokens) / self.refill_rate

            if timeout is not None and (time.time() - start + wait) > timeout:
                logger.warning("RateLimiter timed out")
                return False

            time.sleep(min(wait, 0.1))

    @contextmanager
    def limit(self, timeout: Optional[float] = None):
        """
        Context manager: acquire before the block, nothing to release.

        Raises:
            TimeoutError: if acquire times out.
        """
        if not self.acquire(timeout=timeout):
            raise TimeoutError(f"RateLimiter timeout after {timeout}s")
        yield

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        return False

    def get_available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self.tokens

    def wait_time_for_token(self) -> float:
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                return 0.0
            return (1.0 - self.tokens) / self.refill_rate

    def reset(self) -> None:
        with self._lock:
            self.tokens      = float(self.max_requests)
            self.last_update = time.time()


class MultiRateLimiter:
    """Manages per-endpoint rate limiters for the full SmartAPI surface."""

    def __init__(self):
        self.limiters: dict[str, RateLimiter] = {
            "login":      RateLimiter(RATE_LIMIT_LOGIN,      1.0),
            "ltp":        RateLimiter(RATE_LIMIT_LTP,        1.0),
            "ltp_minute": RateLimiter(RATE_LIMIT_LTP_MINUTE, 60.0),
            "historical": RateLimiter(RATE_LIMIT_HISTORICAL, 1.0),
            "orders":     RateLimiter(RATE_LIMIT_ORDERS,     1.0),
            "default":    RateLimiter(RATE_LIMIT_DEFAULT,    1.0),
        }
        logger.info("MultiRateLimiter initialised")

    def acquire(self, endpoint: str = "default", timeout: Optional[float] = None) -> bool:
        return self.limiters.get(endpoint, self.limiters["default"]).acquire(timeout=timeout)

    @contextmanager
    def limit(self, endpoint: str = "default", timeout: Optional[float] = None):
        if not self.acquire(endpoint=endpoint, timeout=timeout):
            raise TimeoutError(f"RateLimiter timeout for endpoint: {endpoint}")
        yield

    def get_status(self) -> dict:
        return {
            name: {
                "max_requests":      lim.max_requests,
                "time_window":       lim.time_window,
                "available_tokens":  lim.get_available_tokens(),
                "wait_time":         lim.wait_time_for_token(),
            }
            for name, lim in self.limiters.items()
        }