"""
SmartAPI Configuration Constants

Single source of truth for all magic numbers.
No hardcoded values anywhere else in the codebase.
"""

# ---------------------------------------------------------------------------
# Rate limits (requests per second unless noted)
# ---------------------------------------------------------------------------
RATE_LIMIT_LOGIN        = 1       # 1 req/sec
RATE_LIMIT_LTP          = 10      # 10 req/sec
RATE_LIMIT_LTP_MINUTE   = 500     # 500 req/min
RATE_LIMIT_HISTORICAL   = 3       # 3 req/sec
RATE_LIMIT_ORDERS       = 9       # 9 req/sec
RATE_LIMIT_DEFAULT      = 5       # 5 req/sec fallback

# ---------------------------------------------------------------------------
# Exchange type integers (used in WebSocket token_list)
# ---------------------------------------------------------------------------
EXCHANGE_NSE   = 1
EXCHANGE_BSE   = 2
EXCHANGE_NFO   = 3
EXCHANGE_MCX   = 5

# ---------------------------------------------------------------------------
# WebSocket subscription modes
# ---------------------------------------------------------------------------
WS_MODE_LTP        = 1   # last traded price only
WS_MODE_QUOTE      = 2   # LTP + OHLC + volume + bid/ask
WS_MODE_SNAP_QUOTE = 3   # everything + 5-level order book depth

# ---------------------------------------------------------------------------
# Price conversion
# ---------------------------------------------------------------------------
PAISE_DIVISOR = 100   # SmartAPI WebSocket prices are in paise; divide by 100 for rupees

# ---------------------------------------------------------------------------
# WebSocket behaviour
# ---------------------------------------------------------------------------
WS_READY_TIMEOUT_SEC    = 10    # max seconds to wait for ws_ready event in subscribe_symbols
WS_RECONNECT_MAX_TRIES  = 3     # exponential backoff retry attempts
WS_RECONNECT_BASE_DELAY = 5     # seconds before first retry (doubles each attempt)

# ---------------------------------------------------------------------------
# TOTP
# ---------------------------------------------------------------------------
TOTP_MIN_REMAINING_SEC = 5   # if fewer seconds left, wait for next code before login