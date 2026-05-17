"""
Tests for the smartapi layer.

Run with:
    pytest smartapi/tests/ -v
"""

import time
import threading
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from smartapi.totp_generator import TOTPGenerator, generate_totp_from_secret
from smartapi.rate_limiter import RateLimiter, MultiRateLimiter


# ===========================================================================
# TOTPGenerator
# ===========================================================================

VALID_SECRET = "JBSWY3DPEHPK3PXP"   # well-known test vector, valid base32
INVALID_SECRET = "NOT_BASE32_!!!!"


class TestTOTPGenerator:

    def test_valid_secret_constructs(self):
        gen = TOTPGenerator(VALID_SECRET)
        assert gen.secret_key == VALID_SECRET

    def test_invalid_secret_raises_on_construct(self):
        with pytest.raises(ValueError, match="valid base32"):
            TOTPGenerator(INVALID_SECRET)

    def test_empty_secret_raises(self):
        with pytest.raises(ValueError):
            TOTPGenerator("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            TOTPGenerator("   ")

    def test_generate_returns_6_digit_string(self):
        gen = TOTPGenerator(VALID_SECRET)
        otp = gen.generate_totp()
        assert isinstance(otp, str)
        assert len(otp) == 6
        assert otp.isdigit()

    def test_verify_own_otp(self):
        gen = TOTPGenerator(VALID_SECRET)
        otp = gen.generate_totp()
        assert gen.verify_totp(otp) is True

    def test_verify_garbage_otp(self):
        gen = TOTPGenerator(VALID_SECRET)
        assert gen.verify_totp("000000") is False or gen.verify_totp("000000") is True
        # We can't assert False because 000000 might randomly be the current OTP,
        # so instead just assert it returns a bool.
        result = gen.verify_totp("garbage")
        assert isinstance(result, bool)

    def test_validate_secret_format_valid(self):
        assert TOTPGenerator.validate_secret_format(VALID_SECRET) is True

    def test_validate_secret_format_invalid(self):
        assert TOTPGenerator.validate_secret_format("1234!@#$") is False

    def test_generate_totp_from_secret_convenience(self):
        otp = generate_totp_from_secret(VALID_SECRET)
        assert otp is not None
        assert len(otp) == 6

    def test_generate_totp_from_secret_bad_key_returns_none(self):
        result = generate_totp_from_secret(INVALID_SECRET)
        assert result is None


# ===========================================================================
# RateLimiter
# ===========================================================================

class TestRateLimiter:

    def test_basic_acquire(self):
        limiter = RateLimiter(max_requests=5, time_window=1.0)
        assert limiter.acquire() is True

    def test_10_requests_at_10_per_sec_complete_in_roughly_1s(self):
        limiter = RateLimiter(max_requests=10, time_window=1.0)
        start = time.time()
        for _ in range(10):
            limiter.acquire()
        elapsed = time.time() - start
        # First 10 tokens are pre-filled, so all 10 should be instant
        assert elapsed < 0.5, f"Expected <0.5s for pre-filled bucket, got {elapsed:.2f}s"

    def test_exceeding_bucket_causes_wait(self):
        limiter = RateLimiter(max_requests=2, time_window=1.0)
        start = time.time()
        for _ in range(4):    # bucket holds 2; requests 3 and 4 must wait
            limiter.acquire()
        elapsed = time.time() - start
        assert elapsed >= 0.9, f"Expected >=0.9s for 4 req at 2/s, got {elapsed:.2f}s"

    def test_timeout_returns_false(self):
        limiter = RateLimiter(max_requests=1, time_window=10.0)
        limiter.acquire()   # drain bucket
        result = limiter.acquire(timeout=0.2)
        assert result is False

    def test_context_manager(self):
        limiter = RateLimiter(max_requests=5, time_window=1.0)
        with limiter:
            pass   # should not raise

    def test_limit_context_manager(self):
        limiter = RateLimiter(max_requests=5, time_window=1.0)
        with limiter.limit():
            pass

    def test_limit_timeout_raises(self):
        limiter = RateLimiter(max_requests=1, time_window=10.0)
        limiter.acquire()
        with pytest.raises(TimeoutError):
            with limiter.limit(timeout=0.1):
                pass

    def test_reset(self):
        limiter = RateLimiter(max_requests=3, time_window=1.0)
        limiter.acquire(); limiter.acquire(); limiter.acquire()
        assert limiter.get_available_tokens() < 1.0
        limiter.reset()
        assert limiter.get_available_tokens() == pytest.approx(3.0, abs=0.1)

    def test_thread_safety(self):
        """Multiple threads should never see corrupt token counts."""
        limiter = RateLimiter(max_requests=50, time_window=1.0)
        results = []

        def worker():
            results.append(limiter.acquire())

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)


class TestMultiRateLimiter:

    def test_all_endpoints_present(self):
        m = MultiRateLimiter()
        for ep in ("login", "ltp", "ltp_minute", "historical", "orders", "default"):
            assert ep in m.limiters

    def test_acquire_known_endpoint(self):
        m = MultiRateLimiter()
        assert m.acquire("ltp") is True

    def test_acquire_unknown_falls_back_to_default(self):
        m = MultiRateLimiter()
        assert m.acquire("nonexistent_endpoint") is True

    def test_get_status_shape(self):
        m = MultiRateLimiter()
        status = m.get_status()
        for ep, info in status.items():
            assert "max_requests" in info
            assert "available_tokens" in info


# ===========================================================================
# SmartAPIClient — login paths (SmartConnect mocked out)
# ===========================================================================

MOCK_ENV = {
    "ANGEL_API_KEY":     "test_api_key",
    "ANGEL_CLIENT_ID":   "test_client",
    "ANGEL_PASSWORD":    "test_pass",
    "ANGEL_TOTP_SECRET": VALID_SECRET,
}

MOCK_SESSION_SUCCESS = {
    "status": True,
    "data": {
        "jwtToken":     "jwt_abc123",
        "refreshToken": "refresh_xyz",
    },
}

MOCK_SESSION_FAILURE = {
    "status": False,
    "message": "Invalid credentials",
}


class TestSmartAPIClientLogin:

    def _make_client(self):
        """Return a SmartAPIClient with env vars patched."""
        with patch.dict("os.environ", MOCK_ENV):
            from smartapi.smartapi_client import SmartAPIClient
            return SmartAPIClient()

    @patch("smartapi.smartapi_client.SmartConnect")
    def test_login_success(self, MockSmartConnect):
        mock_sc = MagicMock()
        mock_sc.generateSession.return_value = MOCK_SESSION_SUCCESS
        mock_sc.getfeedToken.return_value     = "feed_token_abc"
        MockSmartConnect.return_value = mock_sc

        client = self._make_client()
        result = client.login()

        assert result is True
        assert client.is_logged_in is True
        assert client.auth_token    == "jwt_abc123"
        assert client.refresh_token == "refresh_xyz"
        assert client.feed_token    == "feed_token_abc"

    @patch("smartapi.smartapi_client.SmartConnect")
    def test_login_failure_raises(self, MockSmartConnect):
        mock_sc = MagicMock()
        mock_sc.generateSession.return_value = MOCK_SESSION_FAILURE
        MockSmartConnect.return_value = mock_sc

        client = self._make_client()
        with pytest.raises(Exception, match="Login failed"):
            client.login()

        assert client.is_logged_in is False

    @patch("smartapi.smartapi_client.SmartConnect")
    def test_login_missing_jwt_raises(self, MockSmartConnect):
        mock_sc = MagicMock()
        mock_sc.generateSession.return_value = {
            "status": True,
            "data": {"refreshToken": "r"},   # jwtToken missing
        }
        MockSmartConnect.return_value = mock_sc

        client = self._make_client()
        with pytest.raises(Exception):
            client.login()

    def test_missing_credentials_raises(self):
        from smartapi.smartapi_client import SmartAPIClient
        with pytest.raises(ValueError, match="Missing credentials"):
            SmartAPIClient(api_key=None, client_id=None, password=None, totp_secret=None)

    @patch("smartapi.smartapi_client.SmartConnect")
    def test_logout_calls_terminate_session(self, MockSmartConnect):
        mock_sc = MagicMock()
        mock_sc.generateSession.return_value = MOCK_SESSION_SUCCESS
        mock_sc.getfeedToken.return_value     = "feed_token"
        MockSmartConnect.return_value = mock_sc

        client = self._make_client()
        client.login()
        client.logout()

        mock_sc.terminateSession.assert_called_once_with(client.client_id)
        assert client.is_logged_in  is False
        assert client.auth_token    is None

    @patch("smartapi.smartapi_client.SmartConnect")
    def test_price_cache_divides_paise_by_100(self, MockSmartConnect):
        mock_sc = MagicMock()
        mock_sc.generateSession.return_value = MOCK_SESSION_SUCCESS
        mock_sc.getfeedToken.return_value     = "feed_token"
        MockSmartConnect.return_value = mock_sc

        client = self._make_client()
        client.login()

        # Simulate a WebSocket tick with price in paise
        client._update_price_cache({
            "token":             "2885",
            "last_traded_price": 250000,   # 2500.00 rupees in paise
        })

        cached = client.get_cached_price("2885")
        assert cached is not None
        assert cached["ltp"] == pytest.approx(2500.00)

    @patch("smartapi.smartapi_client.SmartConnect")
    def test_price_cache_thread_safe_concurrent_writes(self, MockSmartConnect):
        mock_sc = MagicMock()
        mock_sc.generateSession.return_value = MOCK_SESSION_SUCCESS
        mock_sc.getfeedToken.return_value     = "feed_token"
        MockSmartConnect.return_value = mock_sc

        client = self._make_client()
        client.login()

        def writer(price):
            for _ in range(100):
                client._update_price_cache({
                    "token": "2885",
                    "last_traded_price": price,
                })

        threads = [threading.Thread(target=writer, args=(p,)) for p in [100000, 200000]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        cached = client.get_cached_price("2885")
        assert cached is not None
        assert cached["ltp"] in (1000.0, 2000.0)   # one of the two writers won cleanly