"""Subscription management tools, input validation, permissions and error handling.

Includes transaction import, subscription management, expense calculation,
alternative search and subscription deletion. Tool execution returns structured
results and applies bounded retries to idempotent operations.
"""

from __future__ import annotations

import csv
import json
import math
import signal
import threading
from contextlib import contextmanager
from datetime import date
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.state import PersistentMemoryStore, SubscriptionRecord

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RETRIES_DEFAULT = 2
DEFAULT_TIMEOUT_SEC = 5.0


# ---------------------------------------------------------------------------
# Custom Error Hierarchy (Transient vs. Permanent)
# ---------------------------------------------------------------------------

class TransientError(Exception):
    """Temporary error eligible for retry on idempotent tools (timeout, 503, lock)."""

    def __init__(self, message: str, error_type: str = "service_unavailable") -> None:
        super().__init__(message)
        self.error_type = error_type


class PermanentError(Exception):
    """Non-retryable error (bad input, missing resource, corrupted file, forbidden)."""

    def __init__(self, message: str, error_type: str = "permanent_error") -> None:
        super().__init__(message)
        self.error_type = error_type


class _DeadlineExceeded(BaseException):
    """Internal interruption that is not swallowed by a broad Exception handler."""


@contextmanager
def tool_deadline(seconds):
    """Enforce a deadline in the Linux main thread (Google Colab)."""
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        raise PermanentError("Timeout requires the Linux main thread.", "unsupported_runtime")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise PermanentError("Another timer is active.", "unsupported_runtime")
    previous = signal.getsignal(signal.SIGALRM)
    def interrupt(signum, frame):
        raise _DeadlineExceeded()
    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------------
# Structured JSON Envelopes
# ---------------------------------------------------------------------------

def ok(data: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "data": data}


def err(kind: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {
        "status": "error",
        "error": {
            "type": kind,
            "message": message,
            "retryable": retryable,
        },
    }


# ---------------------------------------------------------------------------
# Pydantic Input Schemas (extra="forbid" prevents unexpected/injected fields)
# ---------------------------------------------------------------------------

class ImportTransactionsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(
        min_length=1,
        max_length=300,
        description="Relative or absolute path to the synthetic CSV transactions file.",
    )
    min_occurrences: int = Field(
        default=2,
        ge=2,
        le=12,
        description="Minimum number of recurring charges needed to flag a potential subscription.",
    )

    @field_validator("file_path")
    @classmethod
    def must_be_csv(cls, v: str) -> str:
        if not v.lower().endswith(".csv"):
            raise ValueError("file_path must point to a .csv file.")
        return v


class ManageSubscriptionArgs(BaseModel):
    """Note: user_id is intentionally EXCLUDED here; it is injected from the session."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["add", "update"] = Field(
        description="Whether to add a new subscription or update an existing one."
    )
    subscription_id: str = Field(
        min_length=2,
        max_length=64,
        description="Unique identifier for the subscription (e.g., 'sub_netflix').",
    )
    service_name: str = Field(
        min_length=1,
        max_length=120,
        description="Display name of the service (e.g., 'Netflix').",
    )
    amount: float = Field(
        gt=0,
        le=50000,
        description="Recurring charge amount in ILS (must be positive).",
    )
    billing_cycle: Literal["monthly", "yearly"] = Field(
        default="monthly",
        description="Billing frequency: 'monthly' or 'yearly'.",
    )
    category: Literal["streaming", "fitness", "music", "cloud", "internet", "mobile", "other"] = Field(
        default="other",
        description="Normalized subscription category.",
    )
    next_billing_date: str = Field(
        default="2026-12-01",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Next billing date in YYYY-MM-DD format.",
    )


class CalculateExpensesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_prices: list[float] = Field(
        min_length=1,
        max_length=100,
        description="List of current monthly subscription prices in ILS (all > 0).",
    )
    alternative_prices: list[float] | None = Field(
        default=None,
        description="Optional matching list of alternative monthly prices to compute annual savings.",
    )

    @field_validator("current_prices")
    @classmethod
    def validate_current_prices(cls, prices: list[float]) -> list[float]:
        for p in prices:
            if not math.isfinite(p) or p <= 0 or p > 100000:
                raise ValueError("Each price in current_prices must be > 0 and <= 100000.")
        return prices

    @field_validator("alternative_prices")
    @classmethod
    def validate_alternative_prices(cls, prices: list[float] | None) -> list[float] | None:
        if prices is None:
            return None
        for p in prices:
            if not math.isfinite(p) or p < 0 or p > 100000:
                raise ValueError("Each price in alternative_prices must be >= 0 and <= 100000.")
        return prices


class SearchAlternativesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["streaming", "fitness", "music", "cloud", "internet", "mobile"] = Field(
        description="Category to search in the local synthetic alternatives catalog."
    )
    current_price: float = Field(
        gt=0,
        le=50000,
        description="User's current monthly price in ILS; returns alternatives cheaper than this price.",
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of cheaper alternatives to return.",
    )


class DeleteSubscriptionArgs(BaseModel):
    """Note: user_id is intentionally EXCLUDED here; it is injected from the session."""

    model_config = ConfigDict(extra="forbid")

    subscription_id: str = Field(
        min_length=2,
        max_length=64,
        description="Exact subscription_id of a single subscription to delete.",
    )

    @field_validator("subscription_id")
    @classmethod
    def forbid_bulk_delete(cls, v: str) -> str:
        if v.strip().lower() in {"*", "all", "everything", "all_subscriptions"}:
            raise ValueError("Bulk deletion is forbidden. Specify a single subscription_id.")
        return v


# ---------------------------------------------------------------------------
# Explicit Tool Contracts Registry
# ---------------------------------------------------------------------------

TOOL_CONTRACTS: dict[str, dict[str, Any]] = {
    "import_transactions": {
        "name": "import_transactions",
        "description": (
            "Reads a user-uploaded synthetic CSV transactions file, cleans rows, and identifies "
            "recurring charges (potential subscriptions). Use when the user uploads or references "
            "a CSV file. Do NOT use to save subscriptions to the database."
        ),
        "tool_type": "read/compute",
        "requires_approval": False,
        "idempotent": True,
        "args_model": ImportTransactionsArgs,
        "input_schema": ImportTransactionsArgs.model_json_schema(),
        "output_schema": {
            "type": "object",
            "properties": {
                "total_rows": {"type": "integer"},
                "potential_subscriptions": {"type": "array"},
                "ambiguous_merchants": {"type": "array"},
            },
            "required": ["total_rows", "potential_subscriptions", "ambiguous_merchants"],
        },
        "error_types": ["invalid_input", "file_not_found", "file_corrupted", "service_unavailable"],
    },
    "manage_subscriptions": {
        "name": "manage_subscriptions",
        "description": (
            "Adds or updates a single approved subscription in the user's persistent database. "
            "Use ONLY after explicit human approval. Do NOT supply user_id (injected from session)."
        ),
        "tool_type": "write",
        "requires_approval": True,
        "idempotent": False,
        "args_model": ManageSubscriptionArgs,
        "input_schema": ManageSubscriptionArgs.model_json_schema(),
        "output_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "subscription": {"type": "object"},
                "saved": {"type": "boolean"},
            },
            "required": ["action", "subscription", "saved"],
        },
        "error_types": ["invalid_input", "approval_required", "sensitive_data_rejected", "db_transient_error"],
    },
    "calculate_expenses": {
        "name": "calculate_expenses",
        "description": (
            "Performs exact Python arithmetic to compute total monthly and annual subscription "
            "expenses, and optional monthly/annual savings compared to alternative prices. "
            "Always use this tool instead of mental LLM math."
        ),
        "tool_type": "compute",
        "requires_approval": False,
        "idempotent": True,
        "args_model": CalculateExpensesArgs,
        "input_schema": CalculateExpensesArgs.model_json_schema(),
        "output_schema": {
            "type": "object",
            "properties": {
                "monthly_total": {"type": "number"},
                "annual_total": {"type": "number"},
                "monthly_savings": {"type": "number"},
                "annual_savings": {"type": "number"},
            },
            "required": ["monthly_total", "annual_total"],
        },
        "error_types": ["invalid_input", "length_mismatch", "service_unavailable"],
    },
    "search_alternatives": {
        "name": "search_alternatives",
        "description": (
            "Searches the local synthetic alternatives catalog (JSON) for cheaper subscription "
            "alternatives in a given category. Returns alternatives strictly cheaper than current_price "
            "along with the catalog source. Do NOT invent prices outside the catalog."
        ),
        "tool_type": "read",
        "requires_approval": False,
        "idempotent": True,
        "args_model": SearchAlternativesArgs,
        "input_schema": SearchAlternativesArgs.model_json_schema(),
        "output_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "category": {"type": "string"},
                "current_price": {"type": "number"},
                "alternatives": {"type": "array"},
            },
            "required": ["source", "category", "current_price", "alternatives"],
        },
        "error_types": ["invalid_input", "catalog_not_found", "service_unavailable"],
    },
    "delete_subscription": {
        "name": "delete_subscription",
        "description": (
            "Deletes a single existing subscription from the user's database. "
            "REQUIRES explicit human approval per subscription. Bulk deletion is strictly forbidden. "
            "Do NOT supply user_id (injected from session)."
        ),
        "tool_type": "write",
        "requires_approval": True,
        "idempotent": False,
        "args_model": DeleteSubscriptionArgs,
        "input_schema": DeleteSubscriptionArgs.model_json_schema(),
        "output_schema": {
            "type": "object",
            "properties": {
                "deleted": {"type": "boolean"},
                "subscription_id": {"type": "string"},
            },
            "required": ["deleted", "subscription_id"],
        },
        "error_types": ["invalid_input", "approval_required", "not_found", "db_transient_error"],
    },
}

ALLOWED_TOOLS = set(TOOL_CONTRACTS.keys())

KNOWN_CATEGORIES: dict[str, str] = {
    "netflix": "streaming",
    "disney": "streaming",
    "amazon prime": "streaming",
    "spotify": "music",
    "holmes place": "fitness",
    "gym": "fitness",
    "profit": "fitness",
    "icloud": "cloud",
    "google one": "cloud",
    "partner": "internet",
}

AMBIGUOUS_MERCHANTS = {"apple", "mysubs", "google"}


def _infer_category(merchant_name: str) -> tuple[str | None, bool]:
    lower = merchant_name.strip().lower()
    if lower in AMBIGUOUS_MERCHANTS:
        return None, True
    for keyword, cat in KNOWN_CATEGORIES.items():
        if keyword in lower:
            return cat, False
    return None, True


# ---------------------------------------------------------------------------
# Tool Executor with Validation, Permissions, Timeout, Retries & Logging
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Executes approved tools with contract validation, permissions, retries, and logging."""

    def __init__(
        self,
        memory_store: PersistentMemoryStore,
        catalog_path: Path | str | None = None,
        fault_injector: Callable[[str, int], None] | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.catalog_path = Path(catalog_path) if catalog_path else DEFAULT_DATA_DIR / "alternatives_catalog.json"
        # Optional mock hook for testing transient failures without external services
        self.fault_injector = fault_injector
        self.execution_logs: list[dict[str, Any]] = []

    def _impl_import_transactions(self, args: ImportTransactionsArgs) -> dict[str, Any]:
        path = Path(args.file_path)
        if not path.is_absolute():
            candidate = DEFAULT_DATA_DIR / path.name
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise PermanentError(f"CSV file not found: {args.file_path}", "file_not_found")

        grouped: dict[str, list[tuple[date, float, str]]] = defaultdict(list)
        total_rows = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                required_cols = {"date", "merchant", "amount", "currency"}
                if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
                    raise PermanentError(
                        "CSV file missing required columns: date, merchant, amount, currency.",
                        "file_corrupted",
                    )
                for row in reader:
                    total_rows += 1
                    merchant = (row.get("merchant") or "").strip()
                    raw_amount = (row.get("amount") or "").strip()
                    date_str = (row.get("date") or "").strip()
                    if not merchant or not raw_amount:
                        continue
                    try:
                        amount = round(float(raw_amount), 2)
                    except ValueError as exc:
                        raise PermanentError(
                            f"Non-numeric amount '{raw_amount}' at row {total_rows}.",
                            "file_corrupted",
                        ) from exc
                    if not math.isfinite(amount):
                        raise PermanentError("Non-finite amount.", "file_corrupted")
                    if amount <= 0:
                        continue
                    if (row.get("currency") or "").strip().upper() != "ILS":
                        raise PermanentError("Only ILS transactions are supported.", "file_corrupted")
                    try:
                        charge_date = date.fromisoformat(date_str)
                    except ValueError as exc:
                        raise PermanentError("Invalid transaction date.", "file_corrupted") from exc
                    grouped[merchant.casefold()].append((charge_date, amount, merchant))
        except PermanentError:
            raise
        except Exception as exc:
            raise PermanentError(f"Failed to parse CSV file: {exc}", "file_corrupted") from exc

        potential: list[dict[str, Any]] = []
        ambiguous: list[dict[str, Any]] = []

        # זיהוי רצף חודשי לפי בית עסק, גם כאשר המחיר משתנה.
        for charges in grouped.values():
            charges.sort()
            streak = []
            for charge in charges:
                if streak and not 25 <= (charge[0] - streak[-1][0]).days <= 35:
                    streak = []
                streak.append(charge)
            if len(streak) >= args.min_occurrences:
                _, amount, merchant = streak[-1]
                category, is_ambiguous = _infer_category(merchant)
                item = {
                    "service_name": merchant,
                    "amount": amount,
                    "occurrences": len(streak),
                    "price_changed": len({c[1] for c in streak}) > 1,
                    "previous_amount": streak[-2][1],
                    "billing_cycle": "monthly",
                    "inferred_category": category,
                    "needs_clarification": is_ambiguous,
                }
                potential.append(item)
                if is_ambiguous:
                    ambiguous.append(item)

        return {
            "total_rows": total_rows,
            "potential_subscriptions": potential,
            "ambiguous_merchants": ambiguous,
        }

    def _impl_manage_subscriptions(
        self, args: ManageSubscriptionArgs, session_user_id: str
    ) -> dict[str, Any]:
        try:
            record = SubscriptionRecord(
                subscription_id=args.subscription_id,
                service_name=args.service_name,
                amount=args.amount,
                billing_cycle=args.billing_cycle,
                category=args.category,
                next_billing_date=args.next_billing_date,
            )
            saved = self.memory_store.upsert_subscription(session_user_id, record)
        except ValueError as exc:
            raise PermanentError(str(exc), "sensitive_data_rejected") from exc

        return {
            "action": args.action,
            "subscription": saved.model_dump(),
            "saved": True,
        }

    def _impl_calculate_expenses(self, args: CalculateExpensesArgs) -> dict[str, Any]:
        monthly_total = round(sum(args.current_prices), 2)
        annual_total = round(monthly_total * 12, 2)
        result: dict[str, Any] = {
            "monthly_total": monthly_total,
            "annual_total": annual_total,
        }
        if args.alternative_prices is not None:
            if len(args.alternative_prices) != len(args.current_prices):
                raise PermanentError(
                    "alternative_prices must have the same length as current_prices.",
                    "length_mismatch",
                )
            alt_monthly = round(sum(args.alternative_prices), 2)
            monthly_savings = round(monthly_total - alt_monthly, 2)
            annual_savings = round(monthly_savings * 12, 2)
            result["alternative_monthly_total"] = alt_monthly
            result["monthly_savings"] = monthly_savings
            result["annual_savings"] = annual_savings
        return result

    def _impl_search_alternatives(self, args: SearchAlternativesArgs) -> dict[str, Any]:
        if not self.catalog_path.exists():
            raise PermanentError(
                f"Alternatives catalog not found at {self.catalog_path}.",
                "catalog_not_found",
            )
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception as exc:
            raise PermanentError(f"Catalog JSON is corrupted: {exc}", "catalog_not_found") from exc

        items = catalog.get("categories", {}).get(args.category, [])
        cheaper = [
            {
                **item,
                "monthly_savings": round(args.current_price - item["monthly_price"], 2),
                "annual_savings": round((args.current_price - item["monthly_price"]) * 12, 2),
            }
            for item in items
            if item["monthly_price"] < args.current_price
        ]
        cheaper.sort(key=lambda x: x["monthly_price"])

        return {
            "source": catalog.get("source", "internal_synthetic_catalog"),
            "updated_at": catalog.get("updated_at", "unknown"),
            "category": args.category,
            "current_price": args.current_price,
            "alternatives": cheaper[: args.max_results],
        }

    def _impl_delete_subscription(
        self, args: DeleteSubscriptionArgs, session_user_id: str
    ) -> dict[str, Any]:
        deleted = self.memory_store.delete_subscription(session_user_id, args.subscription_id)
        if not deleted:
            raise PermanentError(
                f"Subscription '{args.subscription_id}' not found for user '{session_user_id}'.",
                "not_found",
            )
        return {
            "deleted": True,
            "subscription_id": args.subscription_id,
        }

    def run_tool(
        self,
        name: str,
        raw_args: dict[str, Any],
        session_user_id: str,
        approved: bool = False,
        retries: int = MAX_RETRIES_DEFAULT,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        backoff_base: float = 0.05,
        step: int = 0,
    ) -> dict[str, Any]:
        """Validates, authorizes, executes (with retry/timeout), and logs a tool call."""
        t0 = time.perf_counter()

        def _record(decision: str, status: str, attempts: int, reason: str = "") -> None:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            self.execution_logs.append(
                {
                    "step": step,
                    "tool": name,
                    "args": raw_args,
                    "session_user_id": session_user_id,
                    "decision": decision,
                    "reason": reason,
                    "attempts": attempts,
                    "status": status,
                    "latency_ms": latency_ms,
                }
            )

        # 1. Allowlist check
        if name not in ALLOWED_TOOLS:
            _record("rejected", "error", 0, "permission_denied: tool not in allowlist")
            return err("permission_denied", f"Tool '{name}' is not allowed.", retryable=False)

        contract = TOOL_CONTRACTS[name]

        # 2. Input schema validation (Pydantic re-validation)
        if not isinstance(raw_args, dict):
            _record("rejected", "error", 0, "invalid_input: args must be dict")
            return err("invalid_input", "Tool arguments must be a JSON object.", retryable=False)

        args_model = contract["args_model"]
        try:
            validated_args = args_model(**raw_args)
        except (ValidationError, TypeError) as exc:
            _record("rejected", "error", 0, f"invalid_input: {exc}")
            return err("invalid_input", f"Invalid arguments for '{name}': {exc}", retryable=False)

        # 3. Human-in-the-loop approval check (enforced in code!)
        if contract["requires_approval"] and not approved:
            _record("rejected", "error", 0, "approval_required: missing explicit user approval")
            return err(
                "approval_required",
                f"Tool '{name}' modifies user data and requires explicit human approval before execution.",
                retryable=False,
            )

        # 4. Execution with timeout & bounded retries (strictly on idempotent tools!)
        max_attempts = (min(max(retries, 0), MAX_RETRIES_DEFAULT) + 1) if contract["idempotent"] else 1
        last_transient: TransientError | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                if not math.isfinite(timeout) or timeout <= 0:
                    raise PermanentError("Timeout must be positive and finite.", "invalid_input")
                with tool_deadline(timeout):
                    if self.fault_injector is not None:
                        self.fault_injector(name, attempt)

                    if name == "import_transactions":
                        data = self._impl_import_transactions(validated_args)
                    elif name == "manage_subscriptions":
                        data = self._impl_manage_subscriptions(validated_args, session_user_id)
                    elif name == "calculate_expenses":
                        data = self._impl_calculate_expenses(validated_args)
                    elif name == "search_alternatives":
                        data = self._impl_search_alternatives(validated_args)
                    elif name == "delete_subscription":
                        data = self._impl_delete_subscription(validated_args, session_user_id)
                    else:
                        raise PermanentError(f"Unhandled tool '{name}'.", "permission_denied")

                _record("executed", "ok", attempt, "success")
                return ok(data)

            except PermanentError as exc:
                _record("executed", "error", attempt, f"permanent:{exc.error_type}")
                return err(exc.error_type, str(exc), retryable=False)

            except _DeadlineExceeded:
                if not contract["idempotent"]:
                    _record("executed", "error", attempt, "write_timeout_outcome_unknown")
                    return err("timeout", "Write outcome may be unknown; check state before retrying.", False)
                if attempt < max_attempts:
                    time.sleep(backoff_base * (2 ** (attempt - 1)))
                    continue
                _record("executed", "error", attempt, "timeout")
                return err("timeout", "Tool exceeded the time limit.", True)

            except TransientError as exc:
                last_transient = exc
                if attempt < max_attempts:
                    time.sleep(backoff_base * (2 ** (attempt - 1)))
                else:
                    _record(
                        "executed",
                        "error",
                        attempt,
                        f"transient_exhausted:{exc.error_type}",
                    )
                    return err(
                        exc.error_type,
                        f"Transient error in '{name}' after {attempt} attempt(s): {exc}",
                        retryable=True,
                    )

        # Fallback (unreachable)
        _record("executed", "error", max_attempts, "unknown_failure")
        return err("service_unavailable", str(last_transient or "Unknown error"), retryable=True)
