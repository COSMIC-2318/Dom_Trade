"""
SmartAPI Client — Angel One broker connection layer

Responsibilities:
  1. Login (TOTP → JWT + feed token)
  2. WebSocket streaming in a daemon thread (Mode 1/2/3)
  3. Rate-limit every API call

Does NOT do: order placement, strategy logic, P&L, DB writes.
Those belong in higher layers.

Fixed vs original:
  - ws.connect() now runs in a daemon thread (was blocking main thread)
  - subscribe_symbols() waits on threading.Event, not a bare flag
  - last_traded_price divided by PAISE_DIVISOR before caching
  - _price_cache protected by threading.RLock
  - reconnect() with exponential backoff on error/close
"""

import os
import time
import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from smartapi.config import (
    PAISE_DIVISOR,
    TOTP_MIN_REMAINING_SEC,
    WS_MODE_SNAP_QUOTE,
    WS_READY_TIMEOUT_SEC,
    WS_RECONNECT_BASE_DELAY,
    WS_RECONNECT_MAX_TRIES,
)
from smartapi.rate_limiter import MultiRateLimiter
from smartapi.totp_generator import TOTPGenerator

logger = logging.getLogger(__name__)


class SmartAPIClient:
    """
    Production-ready Angel One SmartAPI client.

    Credentials are read from environment variables or passed directly:
        ANGEL_API_KEY
        ANGEL_CLIENT_ID
        ANGEL_PASSWORD
        ANGEL_TOTP_SECRET

    Typical usage:
        with SmartAPIClient() as client:
            client.start_websocket()
            client.wait_for_websocket()
            client.subscribe_symbols([{"exchangeType": 1, "tokens": ["2885"]}])
            # trading logic here
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        client_id:   Optional[str] = None,
        password:    Optional[str] = None,
        totp_secret: Optional[str] = None,
    ):
        self.api_key     = api_key     or os.getenv("ANGEL_API_KEY")
        self.client_id   = client_id   or os.getenv("ANGEL_CLIENT_ID")
        self.password    = password    or os.getenv("ANGEL_PASSWORD")
        self.totp_secret = totp_secret or os.getenv("ANGEL_TOTP_SECRET")

        missing = [
            k for k, v in {
                "ANGEL_API_KEY":     self.api_key,
                "ANGEL_CLIENT_ID":   self.client_id,
                "ANGEL_PASSWORD":    self.password,
                "ANGEL_TOTP_SECRET": self.totp_secret,
            }.items() if not v
        ]
        if missing:
            raise ValueError(f"Missing credentials: {', '.join(missing)}")

        self.totp_generator = TOTPGenerator(self.totp_secret)
        self.rate_limiter   = MultiRateLimiter()

        # SmartAPI instances
        self.smart_api: Optional[SmartConnect]    = None
        self.ws:        Optional[SmartWebSocketV2] = None

        # Session tokens
        self.auth_token:    Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.feed_token:    Optional[str] = None

        # State
        self.is_logged_in = False
        self.ws_connected = False

        # FIX 1: threading.Event so subscribe_symbols() waits reliably
        self._ws_ready = threading.Event()

        # FIX 2: RLock protects _price_cache from concurrent reads/writes
        self._cache_lock  = threading.RLock()
        self._price_cache: Dict[str, Dict[str, Any]] = {}

        self._price_callbacks: List[Callable] = []
        self._reconnect_attempts = 0

        logger.info("SmartAPIClient initialised")

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self) -> bool:
        """
        Authenticate with Angel One.

        Steps:
          1. Generate TOTP (wait for fresh code if < TOTP_MIN_REMAINING_SEC left)
          2. generateSession() → jwtToken + refreshToken
          3. getfeedToken() → feed_token for WebSocket

        Returns:
            True on success.

        Raises:
            Exception on failure (so callers can decide whether to retry).
        """
        logger.info("Login: starting...")

        with self.rate_limiter.limit("login", timeout=5.0):

            remaining = self.totp_generator.get_remaining_seconds()
            if remaining < TOTP_MIN_REMAINING_SEC:
                logger.warning(f"TOTP expires in {remaining}s — waiting for next code")
                time.sleep(remaining + 1)

            totp = self.totp_generator.generate_totp()

            self.smart_api = SmartConnect(api_key=self.api_key)
            session = self.smart_api.generateSession(
                clientCode=self.client_id,
                password=self.password,
                totp=totp,
            )

            if not (session and session.get("status")):
                raise Exception(f"Login failed: {session.get('message', 'unknown error')}")

            data = session.get("data", {})
            self.auth_token    = data.get("jwtToken")
            self.refresh_token = data.get("refreshToken")

            if not self.auth_token or not self.refresh_token:
                raise Exception("Session response missing jwtToken or refreshToken")

            self._fetch_feed_token()
            self.is_logged_in = True
            logger.info("Login: success")
            return True

    def _fetch_feed_token(self) -> None:
        with self.rate_limiter.limit("default", timeout=5.0):
            token = self.smart_api.getfeedToken()
            if not token:
                raise Exception("getfeedToken() returned empty")
            self.feed_token = token
            logger.info("Feed token obtained")

    def refresh_session(self) -> bool:
        """
        Refresh the JWT using the stored refresh token.

        Call this proactively before token expiry rather than waiting
        for a 401. Exposed publicly so schedulers can call it.

        Returns:
            True if refresh succeeded.
        """
        if not self.refresh_token:
            logger.error("refresh_session: no refresh_token stored")
            return False
        try:
            result = self.smart_api.generateToken(self.refresh_token)
            if result and result.get("status"):
                self.auth_token = result["data"].get("jwtToken", self.auth_token)
                logger.info("Session refreshed")
                return True
            logger.warning(f"Session refresh failed: {result}")
            return False
        except Exception as exc:
            logger.error(f"refresh_session error: {exc}")
            return False

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    def start_websocket(self, on_tick: Optional[Callable] = None) -> bool:
        """
        Start WebSocket streaming in a background daemon thread.

        FIX: original code called ws.connect() on the main thread,
        which blocks forever. We now run it in a daemon thread so the
        caller can continue to subscribe_symbols() etc.

        Args:
            on_tick: Optional callback(tick_data: dict) for every message.

        Returns:
            True if the WebSocket object was created and thread started.
            Use wait_for_websocket() to block until actually connected.
        """
        if not self.is_logged_in:
            logger.error("start_websocket: not logged in")
            return False

        self._ws_ready.clear()

        self.ws = SmartWebSocketV2(
            auth_token=self.auth_token,
            api_key=self.api_key,
            client_code=self.client_id,
            feed_token=self.feed_token,
        )

        def on_open(ws):
            logger.info("WebSocket: connected")
            self.ws_connected          = True
            self._reconnect_attempts   = 0
            self._ws_ready.set()          # unblocks wait_for_websocket()

        def on_data(ws, message):
            self._update_price_cache(message)
            if on_tick:
                try:
                    on_tick(message)
                except Exception as exc:
                    logger.error(f"on_tick callback error: {exc}")
            for cb in list(self._price_callbacks):
                try:
                    cb(message)
                except Exception as exc:
                    logger.error(f"price callback error: {exc}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            self.ws_connected = False
            self._ws_ready.clear()
            self._schedule_reconnect()

        def on_close(ws):
            logger.info("WebSocket: disconnected")
            self.ws_connected = False
            self._ws_ready.clear()
            self._schedule_reconnect()

        self.ws.on_open  = on_open
        self.ws.on_data  = on_data
        self.ws.on_error = on_error
        self.ws.on_close = on_close

        # FIX: run in daemon thread — main thread stays free
        thread = threading.Thread(target=self.ws.connect, daemon=True)
        thread.start()
        logger.info("WebSocket thread started")
        return True

    def wait_for_websocket(self, timeout: float = WS_READY_TIMEOUT_SEC) -> bool:
        """
        Block until the WebSocket is open or timeout expires.

        Args:
            timeout: Seconds to wait.

        Returns:
            True if connected within timeout.
        """
        connected = self._ws_ready.wait(timeout=timeout)
        if not connected:
            logger.error(f"WebSocket not ready after {timeout}s")
        return connected

    def _schedule_reconnect(self) -> None:
        """Spawn a reconnect attempt in a background thread after a delay."""
        if self._reconnect_attempts >= WS_RECONNECT_MAX_TRIES:
            logger.error("WebSocket: max reconnect attempts reached")
            return

        delay = WS_RECONNECT_BASE_DELAY * (2 ** self._reconnect_attempts)
        self._reconnect_attempts += 1
        logger.info(
            f"WebSocket reconnect #{self._reconnect_attempts} in {delay}s"
        )

        def _reconnect():
            time.sleep(delay)
            if not self.ws_connected:
                logger.info("WebSocket: attempting reconnect...")
                self.start_websocket()

        threading.Thread(target=_reconnect, daemon=True).start()

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def subscribe_symbols(
        self,
        tokens: List[Dict[str, Any]],
        mode:   int = WS_MODE_SNAP_QUOTE,
    ) -> bool:
        """
        Subscribe to symbols for real-time ticks.

        FIX: original checked ws_connected directly, which is False until
        on_open fires. Now we wait on _ws_ready (a threading.Event set
        inside on_open) so this is safe to call right after start_websocket().

        Args:
            tokens: e.g. [{"exchangeType": 1, "tokens": ["2885", "3456"]}]
            mode:   WS_MODE_LTP=1, WS_MODE_QUOTE=2, WS_MODE_SNAP_QUOTE=3

        Returns:
            True on success.
        """
        if not self._ws_ready.wait(timeout=WS_READY_TIMEOUT_SEC):
            logger.error("subscribe_symbols: WebSocket not ready")
            return False

        try:
            self.ws.subscribe(
                correlation_id="sub",
                mode=mode,
                token_list=tokens,
            )
            logger.info(f"Subscribed: mode={mode}, groups={len(tokens)}")
            return True
        except Exception as exc:
            logger.error(f"subscribe_symbols failed: {exc}")
            return False

    def unsubscribe_symbols(
        self,
        tokens: List[Dict[str, Any]],
        mode:   int = WS_MODE_SNAP_QUOTE,
    ) -> bool:
        if not self.ws_connected:
            logger.error("unsubscribe_symbols: WebSocket not connected")
            return False
        try:
            self.ws.unsubscribe(
                correlation_id="unsub",
                mode=mode,
                token_list=tokens,
            )
            return True
        except Exception as exc:
            logger.error(f"unsubscribe_symbols failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Price cache
    # ------------------------------------------------------------------

    def _update_price_cache(self, tick: Dict) -> None:
        """
        Write one tick into _price_cache.

        FIX 1: divide last_traded_price by PAISE_DIVISOR (100) — WebSocket
                delivers prices in paise, we store and expose them in rupees.
        FIX 2: wrap all reads and writes in _cache_lock (RLock) to prevent
                data corruption when the WebSocket thread and a strategy
                thread read simultaneously.
        """
        try:
            token = tick.get("token")
            if not token:
                return

            raw_price = tick.get("last_traded_price")
            ltp = raw_price / PAISE_DIVISOR if raw_price is not None else None

            with self._cache_lock:
                self._price_cache[token] = {
                    "ltp":       ltp,
                    "volume":    tick.get("volume_trade_for_the_day"),
                    "timestamp": datetime.now(),
                    "raw":       tick,
                }
        except Exception as exc:
            logger.error(f"_update_price_cache error: {exc}")

    def get_cached_price(self, token: str) -> Optional[Dict]:
        """Thread-safe read from price cache."""
        with self._cache_lock:
            return self._price_cache.get(token)

    def register_price_callback(self, callback: Callable) -> None:
        if callback not in self._price_callbacks:
            self._price_callbacks.append(callback)

    def unregister_price_callback(self, callback: Callable) -> None:
        self._price_callbacks = [c for c in self._price_callbacks if c != callback]

    # ------------------------------------------------------------------
    # REST data helpers
    # ------------------------------------------------------------------

    def get_ltp_data(
        self,
        exchange:       str,
        trading_symbol: str,
        symbol_token:   str,
    ) -> Optional[Dict]:
        """LTP via REST (rate-limited). Use WebSocket for continuous feeds."""
        if not self.is_logged_in:
            return None
        try:
            with self.rate_limiter.limit("ltp"):
                with self.rate_limiter.limit("ltp_minute"):
                    return self.smart_api.ltpData(
                        exchange=exchange,
                        tradingsymbol=trading_symbol,
                        symboltoken=symbol_token,
                    )
        except Exception as exc:
            logger.error(f"get_ltp_data: {exc}")
            return None

    def get_profile(self) -> Optional[Dict]:
        if not self.is_logged_in:
            return None
        try:
            with self.rate_limiter.limit("default"):
                return self.smart_api.getProfile(self.refresh_token)
        except Exception as exc:
            logger.error(f"get_profile: {exc}")
            return None

    # ------------------------------------------------------------------
    # Session cleanup
    # ------------------------------------------------------------------

    def stop_websocket(self) -> None:
        if self.ws and self.ws_connected:
            try:
                self.ws.close_connection()
                self.ws_connected = False
                self._ws_ready.clear()
            except Exception as exc:
                logger.error(f"stop_websocket: {exc}")

    def logout(self) -> bool:
        """Terminate session with Angel One and clean up local state."""
        try:
            self.stop_websocket()

            if self.smart_api and self.client_id:
                try:
                    self.smart_api.terminateSession(self.client_id)
                    logger.info("Session terminated with Angel One")
                except Exception as exc:
                    logger.warning(f"terminateSession error (non-fatal): {exc}")

            self.auth_token    = None
            self.refresh_token = None
            self.feed_token    = None
            self.is_logged_in  = False

            with self._cache_lock:
                self._price_cache.clear()

            logger.info("Logged out")
            return True

        except Exception as exc:
            logger.error(f"logout error: {exc}")
            return False

    def get_rate_limiter_status(self) -> Dict:
        return self.rate_limiter.get_status()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, *_):
        self.logout()
        return False