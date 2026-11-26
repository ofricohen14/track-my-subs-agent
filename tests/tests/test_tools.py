"""Unit tests for all 5 approved tools in Track My Subs.

Requirement (Milestone 1, Component 3):
At least 3 tests per tool: valid input, invalid input, and transient failure.
All tests run in a single command (pytest) with zero external services.
"""

from src.state import PersistentMemoryStore, SubscriptionRecord
from src.tools import TOOL_CONTRACTS, ToolExecutor, TransientError


def _make_flaky_injector(target_tool: str, fail_times: int):
    """Mock fault injector (explicitly marked as mock) simulating transient errors."""
    state = {"calls": 0}

    def injector(tool_name: str, attempt: int) -> None:
        if tool_name == target_tool:
            state["calls"] += 1
            if state["calls"] <= fail_times:
                raise TransientError(f"Mock 503 temporary failure #{state['calls']} in {tool_name}")

    return injector, state


# ===========================================================================
# 1. import_transactions (3+ tests: valid, invalid, transient)
# ===========================================================================

def test_import_transactions_valid():
    mem = PersistentMemoryStore()
    ex = ToolExecutor(mem)
    res = ex.run_tool("import_transactions", {"file_path": "sample_transactions.csv"}, session_user_id="u1")
    assert res["status"] == "ok"
    names = {s["service_name"] for s in res["data"]["potential_subscriptions"]}
    assert "Netflix" in names
    assert "Gym" in names
    assert "Apple" in names
    assert "Supermarket" not in names  # one-time charge filtered out


def test_import_transactions_invalid_input():
    mem = PersistentMemoryStore()
    ex = ToolExecutor(mem)
    # Wrong extension + invalid min_occurrences
    res = ex.run_tool("import_transactions", {"file_path": "data.txt", "min_occurrences": 1}, session_user_id="u1")
    assert res["status"] == "error"
    assert res["error"]["type"] == "invalid_input"
    assert res["error"]["retryable"] is False


def test_import_transactions_transient_failure_recovers():
    mem = PersistentMemoryStore()
    injector, calls = _make_flaky_injector("import_transactions", fail_times=2)
    ex = ToolExecutor(mem, fault_injector=injector)
    res = ex.run_tool(
        "import_transactions",
        {"file_path": "sample_transactions.csv"},
        session_user_id="u1",
        retries=2,
        backoff_base=0.001,
    )
    assert res["status"] == "ok"
    assert calls["calls"] == 3  # 2 transient failures + 1 success


# ===========================================================================
# 2. manage_subscriptions (4 tests: valid, invalid, approval gate, transient)
# ===========================================================================

def test_manage_subscriptions_valid_with_approval():
    mem = PersistentMemoryStore()
    ex = ToolExecutor(mem)
    res = ex.run_tool(
        "manage_subscriptions",
        {
            "action": "add",
            "subscription_id": "sub_netflix",
            "service_name": "Netflix",
            "amount": 69.90,
            "billing_cycle": "monthly",
            "category": "streaming",
        },
        session_user_id="u1",
        approved=True,
    )
    assert res["status"] == "ok"
    assert res["data"]["saved"] is True
    assert mem.get_subscription("u1", "sub_netflix") is not None


def test_manage_subscriptions_invalid_input_and_forbids_model_user_id():
    mem = PersistentMemoryStore()
    ex = ToolExecutor(mem)
    # Negative amount + forbidden user_id supplied by model
    res = ex.run_tool(
        "manage_subscriptions",
        {
            "action": "add",
            "subscription_id": "sub_bad",
            "service_name": "Netflix",
            "amount": -10,
            "user_id": "victim_user",
        },
        session_user_id="u1",
        approved=True,
    )
    assert res["status"] == "error"
    assert res["error"]["type"] == "invalid_input"
    assert mem.get_subscriptions("victim_user") == []


def test_manage_subscriptions_blocked_without_approval():
    mem = PersistentMemoryStore()
    ex = ToolExecutor(mem)
    res = ex.run_tool(
        "manage_subscriptions",
        {
            "action": "add",
            "subscription_id": "sub_netflix",
            "service_name": "Netflix",
            "amount": 69.90,
            "category": "streaming",
        },
        session_user_id="u1",
        approved=False,
    )
    assert res["status"] == "error"
    assert res["error"]["type"] == "approval_required"
    assert mem.get_subscriptions("u1") == []  # Zero writes without approval!


def test_manage_subscriptions_transient_failure_no_blind_retry():
    """Write tools are non-idempotent: must NOT blindly retry on transient DB error."""
    mem = PersistentMemoryStore()
    injector, calls = _make_flaky_injector("manage_subscriptions", fail_times=1)
    ex = ToolExecutor(mem, fault_injector=injector)
    res = ex.run_tool(
        "manage_subscriptions",
        {
            "action": "add",
            "subscription_id": "sub_netflix",
            "service_name": "Netflix",
            "amount": 69.90,
            "category": "streaming",
        },
        session_user_id="u1",
        approved=True,
        retries=2,
        backoff_base=0.001,
    )
    assert res["status"] == "error"
    assert res["error"]["retryable"] is True
    assert calls["calls"] == 1  # Executed only once, no duplicate-write retry


# ===========================================================================
# 3. calculate_expenses (3 tests: valid, invalid, transient)
# ===========================================================================

def test_calculate_expenses_valid():
    mem = PersistentMemoryStore()
    ex = ToolExecutor(mem)
    res = ex.run_tool(
        "calculate_expenses",
        {"current_prices": [250.0], "alternative_prices": [150.0]},
        session_user_id="u1",
    )
    assert res["status"] == "ok"
    assert res["data"]["monthly_savings"] == 100.0
    assert res["data"]["annual_savings"] == 1200.0


def test_calculate_expenses_invalid_input():
    mem = PersistentMemoryStore()
    ex = ToolExecutor(mem)
    res = ex.run_tool(
        "calculate_expenses",
        {"current_prices": ["not_a_number"]},
        session_user_id="u1",
    )
    assert res["status"] == "error"
    assert res["error"]["type"] == "invalid_input"


def test_calculate_expenses_transient_failure_recovers():
    mem = PersistentMemoryStore()
    injector, calls = _make_flaky_injector("calculate_expenses", fail_times=1)
    ex = ToolExecutor(mem, fault_injector=injector)
    res = ex.run_tool(
        "calculate_expenses",
        {"current_prices": [69.90], "alternative_prices": [20.00]},
        session_user_id="u1",
        retries=2,
        backoff_base=0.001,
    )
    assert res["status"] == "ok"
    assert res["data"]["annual_savings"] == 598.80
    assert calls["calls"] == 2


# ===========================================================================
# 4. search_alternatives (3 tests: valid, invalid, transient)
# ===========================================================================

def test_search_alternatives_valid():
    mem = PersistentMemoryStore()
    ex = ToolExecutor(mem)
    res = ex.run_tool(
        "search_alternatives",
        {"category": "streaming", "current_price": 69.90, "max_results": 3},
        session_user_id="u1",
    )
    assert res["status"] == "ok"
    assert res["data"]["source"] == "internal_synthetic_catalog_v1"
    assert len(res["data"]["alternatives"]) >= 1
    assert res["data"]["alternatives"][0]["service_name"] == "Amazon Prime Video"


def test_search_alternatives_invalid_input():
    mem = PersistentMemoryStore()
    ex = ToolExecutor(mem)
    res = ex.run_tool(
        "search_alternatives",
        {"category": "unknown_category", "current_price": -5},
        session_user_id="u1",
    )
    assert res["status"] == "error"
    assert res["error"]["type"] == "invalid_input"


def test_search_alternatives_transient_failure_exhausted():
    mem = PersistentMemoryStore()
    injector, calls = _make_flaky_injector("search_alternatives", fail_times=5)
    ex = ToolExecutor(mem, fault_injector=injector)
    res = ex.run_tool(
        "search_alternatives",
        {"category": "fitness", "current_price": 250.0},
        session_user_id="u1",
        retries=2,
        backoff_base=0.001,
    )
    assert res["status"] == "error"
    assert res["error"]["type"] == "service_unavailable"
    assert res["error"]["retryable"] is True
    assert calls["calls"] == 3  # Initial call + 2 retries


# ===========================================================================
# 5. delete_subscription (4 tests: valid, invalid/bulk, approval gate, transient)
# ===========================================================================

def test_delete_subscription_valid_with_approval():
    mem = PersistentMemoryStore()
    mem.upsert_subscription(
        "u1",
        SubscriptionRecord(
            subscription_id="sub_netflix",
            service_name="Netflix",
            amount=69.90,
            category="streaming",
        ),
    )
    ex = ToolExecutor(mem)
    res = ex.run_tool(
        "delete_subscription",
        {"subscription_id": "sub_netflix"},
        session_user_id="u1",
        approved=True,
    )
    assert res["status"] == "ok"
    assert res["data"]["deleted"] is True
    assert mem.get_subscription("u1", "sub_netflix") is None


def test_delete_subscription_invalid_bulk_delete_forbidden():
    mem = PersistentMemoryStore()
    ex = ToolExecutor(mem)
    res = ex.run_tool(
        "delete_subscription",
        {"subscription_id": "all"},
        session_user_id="u1",
        approved=True,
    )
    assert res["status"] == "error"
    assert res["error"]["type"] == "invalid_input"


def test_delete_subscription_blocked_without_approval():
    mem = PersistentMemoryStore()
    mem.upsert_subscription(
        "u1",
        SubscriptionRecord(
            subscription_id="sub_netflix",
            service_name="Netflix",
            amount=69.90,
            category="streaming",
        ),
    )
    ex = ToolExecutor(mem)
    res = ex.run_tool(
        "delete_subscription",
        {"subscription_id": "sub_netflix"},
        session_user_id="u1",
        approved=False,
    )
    assert res["status"] == "error"
    assert res["error"]["type"] == "approval_required"
    assert mem.get_subscription("u1", "sub_netflix") is not None


def test_delete_subscription_transient_failure_no_blind_retry():
    mem = PersistentMemoryStore()
    injector, calls = _make_flaky_injector("delete_subscription", fail_times=1)
    ex = ToolExecutor(mem, fault_injector=injector)
    res = ex.run_tool(
        "delete_subscription",
        {"subscription_id": "sub_netflix"},
        session_user_id="u1",
        approved=True,
        retries=2,
        backoff_base=0.001,
    )
    assert res["status"] == "error"
    assert res["error"]["retryable"] is True
    assert calls["calls"] == 1


# ===========================================================================
# General Contract & Allowlist Tests
# ===========================================================================

def test_unknown_tool_permission_denied():
    mem = PersistentMemoryStore()
    ex = ToolExecutor(mem)
    res = ex.run_tool("drop_database", {}, session_user_id="u1")
    assert res["status"] == "error"
    assert res["error"]["type"] == "permission_denied"


def test_all_five_proposal_tools_have_explicit_contracts():
    expected = {
        "import_transactions",
        "manage_subscriptions",
        "calculate_expenses",
        "search_alternatives",
        "delete_subscription",
    }
    assert set(TOOL_CONTRACTS.keys()) == expected
    for contract in TOOL_CONTRACTS.values():
        assert contract["name"]
        assert contract["description"]
        assert "properties" in contract["input_schema"]
        assert "properties" in contract["output_schema"]


def test_tool_timeout_triggers_transient_error():
    """Verifies that exceeding the tool timeout raises a retryable transient timeout error."""
    import time
    mem = PersistentMemoryStore()

    def slow_injector(tool_name: str, attempt: int) -> None:
        if tool_name == "search_alternatives":
            time.sleep(0.05)

    ex = ToolExecutor(mem, fault_injector=slow_injector)
    res = ex.run_tool(
        "search_alternatives",
        {"category": "streaming", "current_price": 69.90},
        session_user_id="u1",
        retries=1,
        timeout=0.01,
        backoff_base=0.001,
    )
    assert res["status"] == "error"
    assert res["error"]["type"] == "timeout"
    assert res["error"]["retryable"] is True


def test_import_detects_price_change():
    ex = ToolExecutor(PersistentMemoryStore())
    result = ex.run_tool("import_transactions", {"file_path": "sample_transactions.csv"}, session_user_id="u1")
    assert result["status"] == "ok"
    candidates = {s["service_name"]: s for s in result["data"]["potential_subscriptions"]}
    assert set(candidates) == {"Netflix", "Spotify", "Gym", "Apple", "Google One"}
    assert candidates["Netflix"]["price_changed"] is True
    assert candidates["Netflix"]["amount"] == 79.90
    assert candidates["Netflix"]["previous_amount"] == 69.90
    assert candidates["Apple"]["needs_clarification"] is True
