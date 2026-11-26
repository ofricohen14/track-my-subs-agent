"""Unit tests for State Schema, Session Isolation, Persistent Memory, and Context Builder."""

from src.state import (
    AgentState,
    PersistentMemoryStore,
    SessionStore,
    SubscriptionRecord,
    apply_event,
    build_context,
    run_context_comparison,
)


def test_two_conversations_never_mix():
    """Milestone 1 Requirement 4: Prove two separate conversations do not mix state."""
    sessions = SessionStore()
    s_user1 = sessions.create_thread("thread_001_1", user_id="user_001")
    s_user2 = sessions.create_thread("thread_002_1", user_id="user_002")

    s_user1 = apply_event(s_user1, "user", "אני רוצה לבדוק חלופות ל-Netflix")
    s_user1 = apply_event(s_user1, "assistant", "בודק חלופות ל-Netflix עבורך.")
    sessions.save_thread(s_user1)

    s_user2 = apply_event(s_user2, "user", "אני רוצה לבדוק חלופות לחדר הכושר")
    sessions.save_thread(s_user2)

    loaded_user1 = sessions.get_thread("thread_001_1", user_id="user_001")
    loaded_user2 = sessions.get_thread("thread_002_1", user_id="user_002")

    assert len(loaded_user1.messages) == 2
    assert len(loaded_user2.messages) == 1
    assert "Netflix" in loaded_user1.messages[0].content
    assert "Netflix" not in loaded_user2.messages[0].content
    assert loaded_user1.step == 1
    assert loaded_user2.step == 0


def test_cross_user_thread_access_denied():
    sessions = SessionStore()
    sessions.create_thread("thread_private", user_id="user_001")
    try:
        sessions.get_thread("thread_private", user_id="user_002")
        assert False, "Expected PermissionError for cross-user thread access"
    except PermissionError:
        pass


def test_persistent_memory_isolated_by_user_id():
    mem = PersistentMemoryStore()
    mem.upsert_subscription(
        "user_001",
        SubscriptionRecord(
            subscription_id="sub_netflix",
            service_name="Netflix",
            amount=69.90,
            category="streaming",
        ),
    )
    mem.add_preference("user_001", "מעדיפה חלופות ללא התחייבות")

    assert len(mem.get_subscriptions("user_001")) == 1
    assert len(mem.get_subscriptions("user_002")) == 0
    assert mem.get_preferences("user_002") == []


def test_sensitive_financial_data_rejected():
    mem = PersistentMemoryStore()
    try:
        mem.add_preference("user_001", "כרטיס האשראי שלי הוא 4242-4242-4242-4242")
        assert False, "Expected ValueError when storing credit card number"
    except ValueError:
        pass


def test_immutable_state_and_step_budget():
    s0 = AgentState(thread_id="t1", user_id="u1", max_steps=2)
    s1 = apply_event(s0, "assistant", "step 1")
    assert s0.step == 0  # Original state untouched
    assert s1.step == 1
    s2 = apply_event(s1, "assistant", "step 2")
    assert s2.step == 2
    try:
        apply_event(s2, "assistant", "step 3")
        assert False, "Expected RuntimeError when step budget is exhausted"
    except RuntimeError:
        pass


def test_context_comparison_full_vs_reduced():
    results = run_context_comparison()
    by_id = {r["strategy_id"]: r for r in results}

    assert by_id["full"]["required_info_present"] is True
    assert by_id["last_n"]["required_info_present"] is False
    assert by_id["notes"]["required_info_present"] is True
    assert by_id["notes"]["estimated_tokens"] < by_id["full"]["estimated_tokens"]


def test_same_user_threads_are_separate():
    sessions = SessionStore()
    first = sessions.create_thread("t1", "user_001")
    sessions.create_thread("t2", "user_001")
    sessions.save_thread(apply_event(first, "user", "Netflix"))
    assert sessions.get_thread("t2", "user_001").messages == []


def test_identity_cannot_be_changed_by_event():
    import pytest
    state = AgentState(thread_id="t1", user_id="user_001")
    with pytest.raises(ValueError):
        apply_event(state, "user", "hello", extra_updates={"user_id": "user_002"})
