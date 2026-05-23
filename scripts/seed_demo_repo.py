"""Seed a GitLab repository with 10 realistic commits spread over 6 months.

This script demonstrates GhostEngineer's detection capabilities by creating
a controlled history with:
  - Hardcoded secrets (3 occurrences across commits 3, 8, 10)
  - Missing error handling (commits 3, 6, 10)
  - Large functions >80 lines (commits 3, 6, 10)
  - Copy-paste code (same large function in 3 files)
  - Race conditions (shared counter without lock — commits 6, 9, 10)
  - TODO bombs (planted in commit 6, will be 6+ weeks stale by commit 10)

Usage:
    python scripts/seed_demo_repo.py

Required env vars (set in .env or environment):
    GITLAB_URL          — e.g. https://gitlab.com
    GITLAB_TOKEN        — personal access token with api scope
    GITLAB_PROJECT_ID   — numeric project ID to seed
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import gitlab
from dotenv import load_dotenv

load_dotenv()

GITLAB_URL = os.environ.get("GITLAB_URL", "https://gitlab.com")
GITLAB_TOKEN = os.environ.get("GITLAB_TOKEN", "")
GITLAB_PROJECT_ID = int(os.environ.get("GITLAB_PROJECT_ID", "0"))

# ---------------------------------------------------------------------------
# Fake author roster — simulates a small team
# ---------------------------------------------------------------------------

AUTHORS = [
    {"name": "Alice Chen",   "email": "alice.chen@ghostengineerdemo.dev"},
    {"name": "Bob Martinez", "email": "bob.martinez@ghostengineerdemo.dev"},
    {"name": "Carol Okafor", "email": "carol.okafor@ghostengineerdemo.dev"},
]

# ---------------------------------------------------------------------------
# Timestamp generator — 10 commits spread across past 6 months
# ---------------------------------------------------------------------------

def _commit_timestamp(weeks_ago: float) -> str:
    """Return an ISO-8601 timestamp *weeks_ago* weeks in the past."""
    dt = datetime.now(timezone.utc) - timedelta(weeks=weeks_ago)
    return dt.isoformat()


# Commit timestamps (oldest → newest)
TIMESTAMPS = [
    _commit_timestamp(26),   # commit 1  — ~6 months ago
    _commit_timestamp(23),   # commit 2
    _commit_timestamp(20),   # commit 3
    _commit_timestamp(17),   # commit 4
    _commit_timestamp(15),   # commit 5
    _commit_timestamp(12),   # commit 6
    _commit_timestamp(10),   # commit 7
    _commit_timestamp(8),    # commit 8
    _commit_timestamp(6),    # commit 9
    _commit_timestamp(3),    # commit 10 — ~3 weeks ago
]

# ---------------------------------------------------------------------------
# File content definitions
# ---------------------------------------------------------------------------

PAYMENT_SERVICE_V1 = '''\
"""Payment processing service.

Handles Stripe payment intents with proper error handling,
retry logic, and structured logging.
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

STRIPE_BASE_URL = "https://api.stripe.com/v1"
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class PaymentError(Exception):
    """Raised when a payment operation fails after retries."""


class PaymentService:
    """Stripe payment integration with retry and error handling."""

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("api_key must be a valid Stripe secret key")
        self._api_key = api_key
        self._timeout = timeout
        self._session = requests.Session()
        self._session.auth = (api_key, "")

    def create_payment_intent(
        self,
        amount: int,
        currency: str = "usd",
        description: Optional[str] = None,
    ) -> dict:
        """Create a Stripe PaymentIntent.

        Args:
            amount: Amount in the smallest currency unit (e.g. cents).
            currency: ISO 4217 currency code.
            description: Optional human-readable description.

        Returns:
            The PaymentIntent object returned by Stripe.

        Raises:
            PaymentError: If all retries are exhausted.
            ValueError: If amount or currency are invalid.
        """
        if amount <= 0:
            raise ValueError(f"amount must be positive, got {amount}")
        if len(currency) != 3:
            raise ValueError(f"currency must be a 3-letter ISO code, got {currency!r}")

        payload: dict = {"amount": amount, "currency": currency.lower()}
        if description:
            payload["description"] = description

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._session.post(
                    f"{STRIPE_BASE_URL}/payment_intents",
                    data=payload,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()
                logger.info(
                    "PaymentIntent created | id=%s amount=%d currency=%s",
                    data.get("id"),
                    amount,
                    currency,
                )
                return data

            except requests.exceptions.Timeout:
                logger.warning("Stripe timeout (attempt %d/%d)", attempt, MAX_RETRIES)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
            except requests.exceptions.HTTPError as exc:
                logger.error("Stripe HTTP error: %s", exc)
                raise PaymentError(f"Stripe returned {exc.response.status_code}") from exc
            except requests.exceptions.RequestException as exc:
                logger.error("Stripe request failed: %s", exc)
                raise PaymentError("Network error contacting Stripe") from exc

        raise PaymentError("Stripe request timed out after all retries")

    def refund_payment(self, payment_intent_id: str, amount: Optional[int] = None) -> dict:
        """Issue a refund for an existing PaymentIntent.

        Args:
            payment_intent_id: The ID of the PaymentIntent to refund.
            amount: Partial refund amount in cents. None = full refund.

        Returns:
            The Refund object from Stripe.

        Raises:
            PaymentError: On API errors.
            ValueError: If payment_intent_id is blank.
        """
        if not payment_intent_id:
            raise ValueError("payment_intent_id is required")

        payload: dict = {"payment_intent": payment_intent_id}
        if amount is not None:
            if amount <= 0:
                raise ValueError("refund amount must be positive")
            payload["amount"] = amount

        try:
            response = self._session.post(
                f"{STRIPE_BASE_URL}/refunds",
                data=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            logger.info(
                "Refund issued | id=%s payment_intent=%s",
                data.get("id"),
                payment_intent_id,
            )
            return data

        except requests.exceptions.HTTPError as exc:
            logger.error("Refund HTTP error: %s", exc)
            raise PaymentError(f"Stripe refund failed: {exc.response.status_code}") from exc
        except requests.exceptions.RequestException as exc:
            raise PaymentError("Network error during refund") from exc
'''

AUTH_V1 = '''\
"""User authentication module.

Provides password hashing, verification, and JWT token generation.
All secrets are loaded from environment variables — never hardcoded.
"""

import logging
import os
import time
from typing import Optional

import bcrypt

logger = logging.getLogger(__name__)

JWT_SECRET = os.environ["JWT_SECRET"]           # mandatory — crash early if missing
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 3600


class AuthError(Exception):
    """Raised on authentication failures."""


def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt.

    Args:
        plain: The raw password string. Must be non-empty.

    Returns:
        A bcrypt hash string safe to store in the database.

    Raises:
        ValueError: If *plain* is empty.
    """
    if not plain:
        raise ValueError("Password must not be empty")
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plain.encode(), salt).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash.

    Args:
        plain: The raw password provided by the user.
        hashed: The stored bcrypt hash.

    Returns:
        True if the password matches, False otherwise.
    """
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        logger.exception("bcrypt.checkpw raised unexpectedly")
        return False


def generate_token(user_id: str, email: str) -> dict:
    """Generate a signed JWT for the given user.

    Args:
        user_id: The user\'s unique identifier.
        email: The user\'s email address (embedded as a claim).

    Returns:
        Dict with keys \'token\' and \'expires_at\' (Unix timestamp).

    Raises:
        AuthError: If token generation fails.
        ValueError: If user_id or email are blank.
    """
    if not user_id or not email:
        raise ValueError("user_id and email are required")

    try:
        import jwt  # PyJWT
        payload = {
            "sub": user_id,
            "email": email,
            "iat": int(time.time()),
            "exp": int(time.time()) + JWT_EXPIRY_SECONDS,
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        logger.info("JWT issued for user_id=%s", user_id)
        return {"token": token, "expires_at": payload["exp"]}
    except Exception as exc:
        logger.exception("Token generation failed")
        raise AuthError("Could not generate authentication token") from exc


def validate_token(token: str) -> Optional[dict]:
    """Validate and decode a JWT.

    Args:
        token: The encoded JWT string.

    Returns:
        The decoded payload dict, or None if invalid / expired.
    """
    if not token:
        return None
    try:
        import jwt
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        logger.debug("JWT validation failed")
        return None
'''

PAYMENT_SERVICE_V3 = '''\
"""Payment processing service — patched for timeout issue.

Quick fix: increased timeout and added retry for the new payment gateway.
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# TODO: move this to environment variable before next sprint
api_key = \'sk-prod-a8f3k2p9x1m7n4q6\'

STRIPE_BASE_URL = "https://api.stripe.com/v1"
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class PaymentError(Exception):
    pass


class PaymentService:

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        self._session.auth = (api_key, "")

    def create_payment_intent(self, amount: int, currency: str = "usd", description: Optional[str] = None) -> dict:
        payload = {"amount": amount, "currency": currency}
        if description:
            payload["description"] = description
        response = requests.post(
            f"{STRIPE_BASE_URL}/payment_intents",
            data=payload,
            timeout=self._timeout,
        )
        return response.json()

    def process_large_batch_payment(
        self,
        transactions: list,
        merchant_id: str,
        currency: str = "usd",
        notify: bool = True,
        dry_run: bool = False,
        max_amount: int = 100000,
    ) -> dict:
        """Process a batch of payment transactions for a merchant.

        This is doing too much — validation, processing, notification,
        reporting all in one function. Added quickly to meet deadline.
        """
        results = []
        errors = []
        total_processed = 0
        total_failed = 0
        summary = {}

        # Step 1: validate all transactions
        valid_transactions = []
        for txn in transactions:
            if not txn.get("amount"):
                errors.append({"txn": txn, "error": "missing amount"})
                total_failed += 1
                continue
            if txn["amount"] <= 0:
                errors.append({"txn": txn, "error": "amount must be positive"})
                total_failed += 1
                continue
            if txn["amount"] > max_amount:
                errors.append({"txn": txn, "error": f"amount exceeds max {max_amount}"})
                total_failed += 1
                continue
            if not txn.get("customer_id"):
                errors.append({"txn": txn, "error": "missing customer_id"})
                total_failed += 1
                continue
            valid_transactions.append(txn)

        logger.info("Batch validation complete: %d valid, %d invalid", len(valid_transactions), total_failed)

        # Step 2: process each valid transaction
        for txn in valid_transactions:
            if dry_run:
                results.append({"txn": txn, "status": "dry_run", "intent_id": None})
                continue

            payload = {
                "amount": txn["amount"],
                "currency": currency,
                "metadata[merchant_id]": merchant_id,
                "metadata[customer_id]": txn["customer_id"],
            }
            if txn.get("description"):
                payload["description"] = txn["description"]

            response = requests.post(
                f"{STRIPE_BASE_URL}/payment_intents",
                data=payload,
                timeout=self._timeout,
            )
            data = response.json()

            if response.status_code == 200:
                results.append({"txn": txn, "status": "success", "intent_id": data.get("id")})
                total_processed += txn["amount"]
            else:
                errors.append({"txn": txn, "error": data.get("error", {}).get("message", "unknown")})
                total_failed += 1

        # Step 3: send notification
        if notify and not dry_run:
            notification_payload = {
                "merchant_id": merchant_id,
                "total_processed": total_processed,
                "transaction_count": len(results),
                "failed_count": total_failed,
            }
            requests.post(
                "https://notifications.internal/batch-complete",
                json=notification_payload,
                timeout=5,
            )
            logger.info("Notification sent to merchant %s", merchant_id)

        # Step 4: build summary report
        summary = {
            "merchant_id": merchant_id,
            "currency": currency,
            "total_transactions": len(transactions),
            "successful": len(results),
            "failed": total_failed,
            "total_amount_processed": total_processed,
            "errors": errors,
            "results": results,
            "dry_run": dry_run,
        }
        logger.info("Batch complete: %s", summary)
        return summary

    def refund_payment(self, payment_intent_id: str, amount: Optional[int] = None) -> dict:
        if not payment_intent_id:
            raise ValueError("payment_intent_id is required")
        payload: dict = {"payment_intent": payment_intent_id}
        if amount is not None:
            payload["amount"] = amount
        try:
            response = self._session.post(
                f"{STRIPE_BASE_URL}/refunds",
                data=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            raise PaymentError(f"Refund failed: {exc.response.status_code}") from exc
        except requests.exceptions.RequestException as exc:
            raise PaymentError("Network error during refund") from exc
'''

MODELS_V1 = '''\
"""Database models for Ghost Engineer demo application.

Uses SQLAlchemy declarative models with proper typing and constraints.
All fields validated at the model level before hitting the database.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, Numeric, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    """Registered application user."""

    __tablename__ = "users"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: str = Column(String(254), nullable=False, unique=True, index=True)
    password_hash: str = Column(String(72), nullable=False)
    display_name: Optional[str] = Column(String(100), nullable=True)
    is_active: bool = Column(Boolean, nullable=False, default=True)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)

    orders = relationship("Order", back_populates="user", lazy="selectin")

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"


class Product(Base):
    """A purchasable product in the catalogue."""

    __tablename__ = "products"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    sku: str = Column(String(64), nullable=False, unique=True, index=True)
    name: str = Column(String(200), nullable=False)
    description: Optional[str] = Column(Text, nullable=True)
    price_cents: int = Column(Integer, nullable=False)
    currency: str = Column(String(3), nullable=False, default="usd")
    stock_quantity: int = Column(Integer, nullable=False, default=0)
    is_available: bool = Column(Boolean, nullable=False, default=True)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=_now)

    __table_args__ = (
        UniqueConstraint("sku", name="uq_products_sku"),
    )

    def __repr__(self) -> str:
        return f"<Product id={self.id} sku={self.sku!r} price={self.price_cents}>"


class Order(Base):
    """A customer order, linked to a payment intent."""

    __tablename__ = "orders"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    status: str = Column(String(20), nullable=False, default="pending")
    total_cents: int = Column(Integer, nullable=False)
    currency: str = Column(String(3), nullable=False, default="usd")
    stripe_payment_intent_id: Optional[str] = Column(String(100), nullable=True)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)

    user = relationship("User", back_populates="orders")
    line_items = relationship("OrderLineItem", back_populates="order", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Order id={self.id} status={self.status!r} total={self.total_cents}>"


class OrderLineItem(Base):
    """A single line item within an order."""

    __tablename__ = "order_line_items"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    order_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False, index=True)
    product_id: int = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity: int = Column(Integer, nullable=False)
    unit_price_cents: int = Column(Integer, nullable=False)
    subtotal_cents: int = Column(Integer, nullable=False)

    order = relationship("Order", back_populates="line_items")

    def __repr__(self) -> str:
        return f"<OrderLineItem order={self.order_id} product={self.product_id} qty={self.quantity}>"
'''

PAYMENT_SERVICE_V5 = '''\
"""Payment processing service — removed hardcoded API key.

Security review caught the hardcoded api_key. Reverted to env-var approach.
The batch processor function still needs refactoring (tracked in ISSUE-42).
"""

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

STRIPE_BASE_URL = "https://api.stripe.com/v1"
MAX_RETRIES = 3
RETRY_DELAY = 1.0

_api_key = os.environ.get("STRIPE_API_KEY", "")


class PaymentError(Exception):
    pass


class PaymentService:

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        self._session.auth = (_api_key, "")

    def create_payment_intent(self, amount: int, currency: str = "usd", description: Optional[str] = None) -> dict:
        payload = {"amount": amount, "currency": currency}
        if description:
            payload["description"] = description
        response = requests.post(
            f"{STRIPE_BASE_URL}/payment_intents",
            data=payload,
            timeout=self._timeout,
        )
        return response.json()

    def process_large_batch_payment(
        self,
        transactions: list,
        merchant_id: str,
        currency: str = "usd",
        notify: bool = True,
        dry_run: bool = False,
        max_amount: int = 100000,
    ) -> dict:
        """Process a batch of payment transactions for a merchant.

        FIXME: This function is too large — tracked in ISSUE-42.
        """
        results = []
        errors = []
        total_processed = 0
        total_failed = 0

        valid_transactions = []
        for txn in transactions:
            if not txn.get("amount"):
                errors.append({"txn": txn, "error": "missing amount"})
                total_failed += 1
                continue
            if txn["amount"] <= 0:
                errors.append({"txn": txn, "error": "amount must be positive"})
                total_failed += 1
                continue
            if txn["amount"] > max_amount:
                errors.append({"txn": txn, "error": f"amount exceeds max {max_amount}"})
                total_failed += 1
                continue
            if not txn.get("customer_id"):
                errors.append({"txn": txn, "error": "missing customer_id"})
                total_failed += 1
                continue
            valid_transactions.append(txn)

        logger.info("Batch validation: %d valid, %d invalid", len(valid_transactions), total_failed)

        for txn in valid_transactions:
            if dry_run:
                results.append({"txn": txn, "status": "dry_run", "intent_id": None})
                continue

            payload = {
                "amount": txn["amount"],
                "currency": currency,
                "metadata[merchant_id]": merchant_id,
                "metadata[customer_id]": txn["customer_id"],
            }
            if txn.get("description"):
                payload["description"] = txn["description"]

            response = requests.post(
                f"{STRIPE_BASE_URL}/payment_intents",
                data=payload,
                timeout=self._timeout,
            )
            data = response.json()

            if response.status_code == 200:
                results.append({"txn": txn, "status": "success", "intent_id": data.get("id")})
                total_processed += txn["amount"]
            else:
                errors.append({"txn": txn, "error": data.get("error", {}).get("message", "unknown")})
                total_failed += 1

        if notify and not dry_run:
            requests.post(
                "https://notifications.internal/batch-complete",
                json={"merchant_id": merchant_id, "total": total_processed},
                timeout=5,
            )

        return {
            "merchant_id": merchant_id,
            "currency": currency,
            "total_transactions": len(transactions),
            "successful": len(results),
            "failed": total_failed,
            "total_amount_processed": total_processed,
            "errors": errors,
            "results": results,
            "dry_run": dry_run,
        }

    def refund_payment(self, payment_intent_id: str, amount: Optional[int] = None) -> dict:
        if not payment_intent_id:
            raise ValueError("payment_intent_id is required")
        payload: dict = {"payment_intent": payment_intent_id}
        if amount is not None:
            payload["amount"] = amount
        try:
            response = self._session.post(
                f"{STRIPE_BASE_URL}/refunds",
                data=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            raise PaymentError(f"Refund failed: {exc.response.status_code}") from exc
        except requests.exceptions.RequestException as exc:
            raise PaymentError("Network error during refund") from exc
'''

ORDER_SERVICE_V6 = '''\
"""Order processing service.

Handles order creation, fulfilment, and status updates.
"""

import asyncio
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

STRIPE_BASE_URL = "https://api.stripe.com/v1"

# Shared order counter — accessed from multiple async handlers
_pending_order_count = 0


async def process_order_batch(
    orders: list,
    merchant_id: str,
    currency: str = "usd",
    notify: bool = True,
    dry_run: bool = False,
    max_amount: int = 100000,
) -> dict:
    """Process a batch of orders for fulfilment.

    TODO: validate input here — currently accepts any dict structure
    """
    global _pending_order_count
    results = []
    errors = []
    total_processed = 0
    total_failed = 0

    # Race condition: check-then-act on shared counter without lock
    if _pending_order_count > 100:
        logger.warning("Too many pending orders: %d", _pending_order_count)
        return {"status": "rejected", "reason": "queue full"}

    _pending_order_count += 1   # another coroutine can interleave here

    valid_orders = []
    for order in orders:
        if not order.get("amount"):
            errors.append({"order": order, "error": "missing amount"})
            total_failed += 1
            continue
        if order["amount"] <= 0:
            errors.append({"order": order, "error": "amount must be positive"})
            total_failed += 1
            continue
        if order["amount"] > max_amount:
            errors.append({"order": order, "error": f"amount exceeds max {max_amount}"})
            total_failed += 1
            continue
        if not order.get("customer_id"):
            errors.append({"order": order, "error": "missing customer_id"})
            total_failed += 1
            continue
        valid_orders.append(order)

    logger.info("Order validation: %d valid, %d invalid", len(valid_orders), total_failed)

    for order in valid_orders:
        if dry_run:
            results.append({"order": order, "status": "dry_run", "intent_id": None})
            continue

        payload = {
            "amount": order["amount"],
            "currency": currency,
            "metadata[merchant_id]": merchant_id,
            "metadata[customer_id]": order["customer_id"],
        }
        if order.get("description"):
            payload["description"] = order["description"]

        response = requests.post(
            f"{STRIPE_BASE_URL}/payment_intents",
            data=payload,
            timeout=30,
        )
        data = response.json()

        if response.status_code == 200:
            results.append({"order": order, "status": "success", "intent_id": data.get("id")})
            total_processed += order["amount"]
        else:
            errors.append({"order": order, "error": data.get("error", {}).get("message", "unknown")})
            total_failed += 1

        await asyncio.sleep(0.1)   # yield control — shared state window

    if notify and not dry_run:
        requests.post(
            "https://notifications.internal/orders-complete",
            json={"merchant_id": merchant_id, "total": total_processed},
            timeout=5,
        )

    _pending_order_count -= 1

    return {
        "merchant_id": merchant_id,
        "total_orders": len(orders),
        "successful": len(results),
        "failed": total_failed,
        "total_amount": total_processed,
        "errors": errors,
        "results": results,
    }
'''

INVENTORY_V7 = '''\
"""Inventory management module.

Tracks stock levels, handles reservations, and triggers reorder alerts.
All state changes are atomic and properly logged.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_LOW_STOCK_THRESHOLD = 10
_REORDER_QUANTITY = 100


class InsufficientStockError(Exception):
    """Raised when reservation exceeds available stock."""


class InventoryManager:
    """Thread-safe inventory manager using asyncio.Lock for state protection."""

    def __init__(self) -> None:
        self._stock: dict[str, int] = {}
        self._reserved: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def set_stock(self, sku: str, quantity: int) -> None:
        """Set absolute stock level for a SKU.

        Args:
            sku: The product SKU.
            quantity: The new stock quantity (must be non-negative).

        Raises:
            ValueError: If quantity is negative.
        """
        if quantity < 0:
            raise ValueError(f"Stock quantity cannot be negative: {quantity}")

        async with self._lock:
            old = self._stock.get(sku, 0)
            self._stock[sku] = quantity
            logger.info("Stock updated | sku=%s old=%d new=%d", sku, old, quantity)

            if quantity <= _LOW_STOCK_THRESHOLD:
                logger.warning("Low stock alert | sku=%s quantity=%d", sku, quantity)

    async def reserve(self, sku: str, quantity: int, order_id: str) -> None:
        """Reserve *quantity* units for an order.

        Args:
            sku: The product SKU.
            quantity: Number of units to reserve.
            order_id: The order making the reservation.

        Raises:
            InsufficientStockError: If available stock is less than *quantity*.
            ValueError: If quantity is not positive.
        """
        if quantity <= 0:
            raise ValueError(f"Reserve quantity must be positive: {quantity}")

        async with self._lock:
            available = self._stock.get(sku, 0) - self._reserved.get(sku, 0)
            if available < quantity:
                raise InsufficientStockError(
                    f"Cannot reserve {quantity} units of {sku!r}: "
                    f"only {available} available"
                )
            self._reserved[sku] = self._reserved.get(sku, 0) + quantity
            logger.info(
                "Reserved | sku=%s qty=%d order=%s remaining=%d",
                sku, quantity, order_id, available - quantity,
            )

    async def release(self, sku: str, quantity: int, order_id: str) -> None:
        """Release a previously reserved quantity back to available stock."""
        if quantity <= 0:
            raise ValueError(f"Release quantity must be positive: {quantity}")

        async with self._lock:
            current = self._reserved.get(sku, 0)
            self._reserved[sku] = max(0, current - quantity)
            logger.info("Released reservation | sku=%s qty=%d order=%s", sku, quantity, order_id)

    async def get_available(self, sku: str) -> int:
        """Return the number of units available to reserve."""
        async with self._lock:
            return self._stock.get(sku, 0) - self._reserved.get(sku, 0)
'''

ORDER_SERVICE_V8 = '''\
"""Order processing service — urgent hotfix for fulfilment bug.

Hotfix: DB connection was failing silently. Added direct connection string
as a stopgap until the secrets manager is configured.
"""

import asyncio
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

STRIPE_BASE_URL = "https://api.stripe.com/v1"

# URGENT HOTFIX — remove before next sprint review
db_password = \'prod_password_123\'
DB_URL = f"postgresql://app_user:{db_password}@db.internal:5432/orders_prod"

_pending_order_count = 0


async def process_order_batch(
    orders: list,
    merchant_id: str,
    currency: str = "usd",
    notify: bool = True,
    dry_run: bool = False,
    max_amount: int = 100000,
) -> dict:
    """Process a batch of orders for fulfilment.

    TODO: validate input here — currently accepts any dict structure
    """
    global _pending_order_count
    results = []
    errors = []
    total_processed = 0
    total_failed = 0

    if _pending_order_count > 100:
        logger.warning("Too many pending orders: %d", _pending_order_count)
        return {"status": "rejected", "reason": "queue full"}

    _pending_order_count += 1

    valid_orders = []
    for order in orders:
        if not order.get("amount"):
            errors.append({"order": order, "error": "missing amount"})
            total_failed += 1
            continue
        if order["amount"] <= 0:
            errors.append({"order": order, "error": "amount must be positive"})
            total_failed += 1
            continue
        if order["amount"] > max_amount:
            errors.append({"order": order, "error": f"amount exceeds max {max_amount}"})
            total_failed += 1
            continue
        if not order.get("customer_id"):
            errors.append({"order": order, "error": "missing customer_id"})
            total_failed += 1
            continue
        valid_orders.append(order)

    logger.info("Order validation: %d valid, %d invalid", len(valid_orders), total_failed)

    for order in valid_orders:
        if dry_run:
            results.append({"order": order, "status": "dry_run", "intent_id": None})
            continue

        payload = {
            "amount": order["amount"],
            "currency": currency,
            "metadata[merchant_id]": merchant_id,
            "metadata[customer_id]": order["customer_id"],
        }

        response = requests.post(
            f"{STRIPE_BASE_URL}/payment_intents",
            data=payload,
            timeout=30,
        )
        data = response.json()

        if response.status_code == 200:
            results.append({"order": order, "status": "success", "intent_id": data.get("id")})
            total_processed += order["amount"]
        else:
            errors.append({"order": order, "error": data.get("error", {}).get("message", "unknown")})
            total_failed += 1

        await asyncio.sleep(0.1)

    _pending_order_count -= 1

    return {
        "merchant_id": merchant_id,
        "successful": len(results),
        "failed": total_failed,
        "total_amount": total_processed,
        "errors": errors,
    }
'''

ORDER_SERVICE_V9 = '''\
"""Order processing service — cleanup pass.

Removed hardcoded db_password (ISSUE-67). Race condition on _pending_order_count
still present — needs proper fix with asyncio.Lock (ISSUE-68, low priority).
"""

import asyncio
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

STRIPE_BASE_URL = "https://api.stripe.com/v1"
DB_URL = os.environ.get("DATABASE_URL", "")

_pending_order_count = 0


async def process_order_batch(
    orders: list,
    merchant_id: str,
    currency: str = "usd",
    notify: bool = True,
    dry_run: bool = False,
    max_amount: int = 100000,
) -> dict:
    """Process a batch of orders.

    TODO: validate input here — currently accepts any dict structure
    """
    global _pending_order_count
    results = []
    errors = []
    total_processed = 0
    total_failed = 0

    if _pending_order_count > 100:
        return {"status": "rejected", "reason": "queue full"}

    _pending_order_count += 1

    valid_orders = []
    for order in orders:
        if not order.get("amount"):
            errors.append({"order": order, "error": "missing amount"})
            total_failed += 1
            continue
        if order["amount"] <= 0:
            errors.append({"order": order, "error": "amount must be positive"})
            total_failed += 1
            continue
        if order["amount"] > max_amount:
            errors.append({"order": order, "error": f"exceeds max {max_amount}"})
            total_failed += 1
            continue
        if not order.get("customer_id"):
            errors.append({"order": order, "error": "missing customer_id"})
            total_failed += 1
            continue
        valid_orders.append(order)

    for order in valid_orders:
        if dry_run:
            results.append({"order": order, "status": "dry_run"})
            continue

        payload = {
            "amount": order["amount"],
            "currency": currency,
            "metadata[merchant_id]": merchant_id,
            "metadata[customer_id]": order["customer_id"],
        }
        response = requests.post(f"{STRIPE_BASE_URL}/payment_intents", data=payload, timeout=30)
        data = response.json()

        if response.status_code == 200:
            results.append({"order": order, "status": "success", "intent_id": data.get("id")})
            total_processed += order["amount"]
        else:
            errors.append({"order": order, "error": "payment failed"})
            total_failed += 1

        await asyncio.sleep(0.1)

    _pending_order_count -= 1
    return {
        "merchant_id": merchant_id,
        "successful": len(results),
        "failed": total_failed,
        "total_amount": total_processed,
        "errors": errors,
    }


async def process_shipping_batch(
    shipments: list,
    carrier_id: str,
    currency: str = "usd",
    notify: bool = True,
    dry_run: bool = False,
    max_amount: int = 50000,
) -> dict:
    """Process a batch of shipping charges — copy of order processor, adapted."""
    global _pending_order_count
    results = []
    errors = []
    total_processed = 0
    total_failed = 0

    if _pending_order_count > 100:
        return {"status": "rejected", "reason": "queue full"}

    _pending_order_count += 1   # same unprotected counter reused

    valid_shipments = []
    for shipment in shipments:
        if not shipment.get("amount"):
            errors.append({"shipment": shipment, "error": "missing amount"})
            total_failed += 1
            continue
        if shipment["amount"] <= 0:
            errors.append({"shipment": shipment, "error": "amount must be positive"})
            total_failed += 1
            continue
        if shipment["amount"] > max_amount:
            errors.append({"shipment": shipment, "error": f"exceeds max {max_amount}"})
            total_failed += 1
            continue
        if not shipment.get("tracking_id"):
            errors.append({"shipment": shipment, "error": "missing tracking_id"})
            total_failed += 1
            continue
        valid_shipments.append(shipment)

    for shipment in valid_shipments:
        if dry_run:
            results.append({"shipment": shipment, "status": "dry_run"})
            continue

        payload = {"amount": shipment["amount"], "currency": currency}
        response = requests.post(f"{STRIPE_BASE_URL}/payment_intents", data=payload, timeout=30)
        data = response.json()

        if response.status_code == 200:
            results.append({"shipment": shipment, "status": "success"})
            total_processed += shipment["amount"]
        else:
            total_failed += 1

        await asyncio.sleep(0.1)

    _pending_order_count -= 1
    return {
        "carrier_id": carrier_id,
        "successful": len(results),
        "failed": total_failed,
        "total_amount": total_processed,
    }
'''

SHIPPING_V10 = '''\
"""Shipping service.

Handles carrier integrations and shipment charge processing.
Adapted from order_service.py for shipping-specific workflows.
"""

import asyncio
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# TODO: validate input here — add schema validation before production rollout
shipping_api_key = \'sh-live-x7p2k9m4n8q1r3t6v0w5y\'

CARRIER_API_BASE = "https://api.carrier.internal/v2"
STRIPE_BASE_URL = "https://api.stripe.com/v1"

_pending_order_count = 0   # shared with order_service (same pattern)


async def process_shipping_batch(
    shipments: list,
    carrier_id: str,
    currency: str = "usd",
    notify: bool = True,
    dry_run: bool = False,
    max_amount: int = 50000,
) -> dict:
    """Process a batch of shipment charges via the carrier API.

    Note: Input validation not yet implemented — see TODO above.
    """
    global _pending_order_count
    results = []
    errors = []
    total_processed = 0
    total_failed = 0

    if _pending_order_count > 100:
        logger.warning("Shipping queue full: %d", _pending_order_count)
        return {"status": "rejected", "reason": "queue full"}

    _pending_order_count += 1

    valid_shipments = []
    for shipment in shipments:
        if not shipment.get("amount"):
            errors.append({"shipment": shipment, "error": "missing amount"})
            total_failed += 1
            continue
        if shipment["amount"] <= 0:
            errors.append({"shipment": shipment, "error": "amount must be positive"})
            total_failed += 1
            continue
        if shipment["amount"] > max_amount:
            errors.append({"shipment": shipment, "error": f"amount exceeds max {max_amount}"})
            total_failed += 1
            continue
        if not shipment.get("tracking_id"):
            errors.append({"shipment": shipment, "error": "missing tracking_id"})
            total_failed += 1
            continue
        valid_shipments.append(shipment)

    logger.info("Shipping validation: %d valid, %d invalid", len(valid_shipments), total_failed)

    for shipment in valid_shipments:
        if dry_run:
            results.append({"shipment": shipment, "status": "dry_run"})
            continue

        charge_payload = {"amount": shipment["amount"], "currency": currency}
        response = requests.post(
            f"{STRIPE_BASE_URL}/payment_intents",
            data=charge_payload,
            timeout=30,
        )
        data = response.json()

        if response.status_code == 200:
            # Book with carrier
            carrier_response = requests.post(
                f"{CARRIER_API_BASE}/shipments",
                json={
                    "carrier_id": carrier_id,
                    "tracking_id": shipment["tracking_id"],
                    "charge_id": data.get("id"),
                },
                headers={"Authorization": f"Bearer {shipping_api_key}"},
                timeout=15,
            )
            results.append({
                "shipment": shipment,
                "status": "success",
                "carrier_status": carrier_response.status_code,
            })
            total_processed += shipment["amount"]
        else:
            errors.append({"shipment": shipment, "error": data.get("error", {}).get("message", "unknown")})
            total_failed += 1

        await asyncio.sleep(0.1)   # yield — shared counter still unprotected

    if notify and not dry_run:
        requests.post(
            "https://notifications.internal/shipping-complete",
            json={"carrier_id": carrier_id, "total": total_processed},
            timeout=5,
        )

    _pending_order_count -= 1

    return {
        "carrier_id": carrier_id,
        "total_shipments": len(shipments),
        "successful": len(results),
        "failed": total_failed,
        "total_amount_processed": total_processed,
        "errors": errors,
        "results": results,
    }
'''

# ---------------------------------------------------------------------------
# Commit definitions
# ---------------------------------------------------------------------------

COMMITS = [
    {
        "message": "Initial payment service setup",
        "author": AUTHORS[0],
        "timestamp_index": 0,
        "actions": [
            {"action": "create", "file_path": "payment_service.py", "content": PAYMENT_SERVICE_V1},
        ],
    },
    {
        "message": "Add user authentication",
        "author": AUTHORS[1],
        "timestamp_index": 1,
        "actions": [
            {"action": "create", "file_path": "auth.py", "content": AUTH_V1},
        ],
    },
    {
        "message": "Quick fix for payment timeout",
        "author": AUTHORS[0],
        "timestamp_index": 2,
        "actions": [
            {"action": "update", "file_path": "payment_service.py", "content": PAYMENT_SERVICE_V3},
        ],
    },
    {
        "message": "Add database models",
        "author": AUTHORS[2],
        "timestamp_index": 3,
        "actions": [
            {"action": "create", "file_path": "models.py", "content": MODELS_V1},
        ],
    },
    {
        "message": "Fix the payment bug — remove hardcoded API key",
        "author": AUTHORS[1],
        "timestamp_index": 4,
        "actions": [
            {"action": "update", "file_path": "payment_service.py", "content": PAYMENT_SERVICE_V5},
        ],
    },
    {
        "message": "Add order processing",
        "author": AUTHORS[0],
        "timestamp_index": 5,
        "actions": [
            {"action": "create", "file_path": "order_service.py", "content": ORDER_SERVICE_V6},
        ],
    },
    {
        "message": "Add inventory management",
        "author": AUTHORS[2],
        "timestamp_index": 6,
        "actions": [
            {"action": "create", "file_path": "inventory.py", "content": INVENTORY_V7},
        ],
    },
    {
        "message": "Hotfix order processing bug — urgent fix",
        "author": AUTHORS[0],
        "timestamp_index": 7,
        "actions": [
            {"action": "update", "file_path": "order_service.py", "content": ORDER_SERVICE_V8},
        ],
    },
    {
        "message": "Cleanup and refactor — remove hardcoded password",
        "author": AUTHORS[1],
        "timestamp_index": 8,
        "actions": [
            {"action": "update", "file_path": "order_service.py", "content": ORDER_SERVICE_V9},
            {"action": "create", "file_path": "shipping.py", "content": SHIPPING_V10},
        ],
    },
    {
        "message": "Add shipping service",
        "author": AUTHORS[0],
        "timestamp_index": 9,
        "actions": [
            {"action": "update", "file_path": "shipping.py", "content": SHIPPING_V10},
        ],
    },
]


# ---------------------------------------------------------------------------
# GitLab commit helper
# ---------------------------------------------------------------------------

def _push_commit(project, commit_def: dict, branch: str = "main") -> None:
    """Create a single commit on GitLab via the Commits API."""
    idx = commit_def["timestamp_index"]
    timestamp = TIMESTAMPS[idx]
    author = commit_def["author"]

    data = {
        "branch": branch,
        "commit_message": commit_def["message"],
        "author_name": author["name"],
        "author_email": author["email"],
        "actions": commit_def["actions"],
    }

    commit = project.commits.create(data)
    print(
        f"  ✓  [{idx + 1:02d}/10] {commit_def['message'][:60]}"
        f"  —  {author['name']}  —  sha={commit.id[:8]}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not GITLAB_TOKEN:
        print("ERROR: GITLAB_TOKEN is not set. Add it to your .env file.")
        sys.exit(1)

    if GITLAB_PROJECT_ID == 0:
        print("ERROR: GITLAB_PROJECT_ID is not set. Add it to your .env file.")
        sys.exit(1)

    print(f"\nConnecting to {GITLAB_URL} …")
    gl = gitlab.Gitlab(url=GITLAB_URL, private_token=GITLAB_TOKEN)

    try:
        gl.auth()
        print(f"Authenticated as: {gl.users.get(gl.auth()).username}")
    except Exception as exc:
        print(f"ERROR: Authentication failed — {exc}")
        sys.exit(1)

    try:
        project = gl.projects.get(GITLAB_PROJECT_ID)
        print(f"Project: {project.name_with_namespace}  (id={GITLAB_PROJECT_ID})")
    except Exception as exc:
        print(f"ERROR: Could not access project {GITLAB_PROJECT_ID} — {exc}")
        sys.exit(1)

    # Determine default branch
    branch = project.default_branch or "main"
    print(f"Default branch: {branch}")
    print(f"\nCreating {len(COMMITS)} commits …\n")

    for commit_def in COMMITS:
        try:
            _push_commit(project, commit_def, branch=branch)
        except Exception as exc:
            print(f"  ✗  FAILED — {commit_def['message'][:60]}  —  {exc}")

    print("\n✓  Seeding complete!\n")
    print("GhostEngineer should now detect:")
    print("  • Hardcoded secrets   — 3 occurrences (commits 3, 8, 10)")
    print("  • Race conditions     — same unprotected counter in 3 files")
    print("  • Copy-paste code     — large batch function duplicated 3×")
    print("  • Missing error handling — bare requests.post calls")
    print("  • TODO bomb           — planted in commit 6, now 12+ weeks stale")


if __name__ == "__main__":
    main()
