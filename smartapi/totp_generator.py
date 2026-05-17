"""
TOTP Generator for Angel One SmartAPI Authentication

Generates the 6-digit time-based OTP required at login.
Secret must be a valid base32 string (from the Angel One QR code).
"""

import time
import base64
import logging
from typing import Optional

import pyotp

from smartapi.config import TOTP_MIN_REMAINING_SEC

logger = logging.getLogger(__name__)


class TOTPGenerator:
    """
    Generates Time-based One-Time Passwords for Angel One login.

    Validates the secret format immediately on construction so bad
    credentials surface at startup, not mid-login.
    """

    def __init__(self, secret_key: str):
        """
        Args:
            secret_key: Base32-encoded secret from the Angel One QR code.

        Raises:
            ValueError: If secret_key is empty or not valid base32.
        """
        if not secret_key or not secret_key.strip():
            raise ValueError("TOTP secret key cannot be empty")

        cleaned = secret_key.strip().upper()

        # Validate immediately — fail fast rather than failing at login time
        if not self.validate_secret_format(cleaned):
            raise ValueError(
                f"TOTP secret is not valid base32. "
                "Check the value of ANGEL_TOTP_SECRET."
            )

        self.secret_key = cleaned
        self.totp = pyotp.TOTP(self.secret_key)
        logger.debug("TOTP generator initialised")

    def generate_totp(self) -> str:
        """
        Return the current 6-digit OTP.

        Returns:
            str: 6-digit code valid for the current 30-second window.

        Raises:
            RuntimeError: If generation fails.
        """
        try:
            otp = self.totp.now()
            logger.debug("TOTP generated")
            return otp
        except Exception as exc:
            raise RuntimeError(f"TOTP generation failed: {exc}") from exc

    def verify_totp(self, otp: str, window: int = 1) -> bool:
        """
        Check whether otp is valid within ±window time steps.

        Args:
            otp: The code to verify.
            window: How many adjacent 30-second windows to accept.

        Returns:
            bool: True if valid.
        """
        try:
            return self.totp.verify(otp, valid_window=window)
        except Exception as exc:
            logger.error(f"TOTP verification error: {exc}")
            return False

    def get_remaining_seconds(self) -> int:
        """
        Seconds until the current OTP expires (0–30).
        """
        return TOTP_MIN_REMAINING_SEC - (int(time.time()) % 30) + (30 - TOTP_MIN_REMAINING_SEC)

    @staticmethod
    def validate_secret_format(secret_key: str) -> bool:
        """
        Return True if secret_key is a valid base32 string.
        """
        try:
            padded = secret_key + "=" * ((8 - len(secret_key) % 8) % 8)
            base64.b32decode(padded, casefold=True)
            return True
        except Exception:
            return False


def generate_totp_from_secret(secret_key: str) -> Optional[str]:
    """
    Convenience wrapper — returns OTP or None on failure.
    """
    try:
        return TOTPGenerator(secret_key).generate_totp()
    except Exception as exc:
        logger.error(f"generate_totp_from_secret failed: {exc}")
        return None