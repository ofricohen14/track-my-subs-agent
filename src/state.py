"""Application state, session isolation, user memory and context management."""


from __future__ import annotations

import re
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

MAX_STEPS_DEFAULT = 7
SYSTEM_PROMPT = (
    "אתה סוכן בינה מלאכותית לניהול והוזלת מנויים (Track My Subs). "
    "השתמש אך ורק בנתונים החוזרים מהכלים, ציין תמיד את מקור המחיר והחישוב, "
    "ואל תבצע הוספה, עדכון או מחיקה של מנוי ללא אישור מפורש מהמשתמש."
)

SENSITIVE_PATTERNS = [
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),  # Credit card numbers
    re.compile(r"\bcvv\s*[:=]?\s*\d{3,4}\b", re.IGNORECASE),
    re.compile(r"\b(?:חשבון בנק|מספר חשבון|סיסמה|password)\s*[:=]?\s*\S+", re.IGNORECASE),
]


class Message(BaseModel):
    """Single message in conversation history."""

    role: Literal["user", "assistant", "tool"]
    content: str


class SubscriptionRecord(BaseModel):
    """Approved subscription stored in persistent memory (non-sensitive fields only)."""

    model_config = ConfigDict(extra="forbid")

    subscription_id: str
    service_name: str
    amount: float = Field(gt=0)
    billing_cycle: Literal["monthly", "yearly"] = "monthly"
    category: str = "other"
    next_billing_date: str = "2026-12-01"


class AgentState(BaseModel):
    """Explicit application state for a single conversation thread."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    user_id: str
    step: int = Field(default=0, ge=0)
    max_steps: int = Field(default=MAX_STEPS_DEFAULT, ge=1)
    messages: list[Message] = Field(default_factory=list)
    summary: str = ""
    uploaded_transactions: list[dict[str, Any]] = Field(default_factory=list)
    pending_subscriptions: list[dict[str, Any]] = Field(default_factory=list)
    last_alternatives: list[dict[str, Any]] = Field(default_factory=list)
    pending_action: dict[str, Any] | None = None


def apply_event(
    state: AgentState,
    role: Literal["user", "assistant", "tool"],
    content: str,
    extra_updates: dict[str, Any] | None = None,
) -> AgentState:
    """Single immutable state transition function. Enforces step budget on assistant turns."""
    if role == "assistant" and state.step >= state.max_steps:
        raise RuntimeError(
            f"Step budget exhausted ({state.step}/{state.max_steps}) for thread '{state.thread_id}'."
        )

    new_messages = state.messages + [Message(role=role, content=content)]
    updates: dict[str, Any] = {
        "messages": new_messages,
        "step": state.step + (1 if role == "assistant" else 0),
    }
    if extra_updates:
        allowed = {"summary", "uploaded_transactions", "pending_subscriptions", "last_alternatives", "pending_action"}
        if not set(extra_updates).issubset(allowed):
            raise ValueError("Invalid state update fields")
        updates.update(extra_updates)

    return AgentState.model_validate({**state.model_dump(), **updates}).model_copy(deep=True)


class SessionStore:
    """Manages isolated conversation states keyed by thread_id with strict user_id ownership."""

    def __init__(self) -> None:
        self._threads: dict[str, AgentState] = {}

    def create_thread(
        self, thread_id: str, user_id: str, max_steps: int = MAX_STEPS_DEFAULT
    ) -> AgentState:
        if thread_id in self._threads:
            existing = self._threads[thread_id]
            if existing.user_id != user_id:
                raise PermissionError(
                    f"Thread '{thread_id}' belongs to another user and cannot be shared."
                )
            return existing
        state = AgentState(thread_id=thread_id, user_id=user_id, max_steps=max_steps)
        self._threads[thread_id] = state
        return state

    def get_thread(self, thread_id: str, user_id: str) -> AgentState:
        if thread_id not in self._threads:
            raise KeyError(f"Thread '{thread_id}' not found.")
        state = self._threads[thread_id]
        if state.user_id != user_id:
            raise PermissionError(
                f"Access denied: user '{user_id}' cannot access thread '{thread_id}'."
            )
        return state

    def save_thread(self, state: AgentState) -> None:
        if state.thread_id in self._threads:
            if self._threads[state.thread_id].user_id != state.user_id:
                raise PermissionError(
                    f"Ownership mismatch on thread '{state.thread_id}'."
                )
        self._threads[state.thread_id] = state


class PersistentMemoryStore:
    """User-scoped in-memory store shared across conversations in the same process."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, dict[str, SubscriptionRecord]] = {}
        self._preferences: dict[str, list[str]] = {}

    @staticmethod
    def _check_not_sensitive(text: str) -> None:
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(text):
                raise ValueError(
                    "Sensitive financial information (e.g., credit card, CVV, password) "
                    "is strictly prohibited from being stored."
                )

    def upsert_subscription(self, user_id: str, record: SubscriptionRecord) -> SubscriptionRecord:
        self._check_not_sensitive(record.service_name)
        user_subs = self._subscriptions.setdefault(user_id, {})
        user_subs[record.subscription_id] = record
        return record

    def get_subscriptions(self, user_id: str) -> list[SubscriptionRecord]:
        return list(self._subscriptions.get(user_id, {}).values())

    def get_subscription(self, user_id: str, subscription_id: str) -> SubscriptionRecord | None:
        return self._subscriptions.get(user_id, {}).get(subscription_id)

    def delete_subscription(self, user_id: str, subscription_id: str) -> bool:
        user_subs = self._subscriptions.get(user_id, {})
        if subscription_id in user_subs:
            del user_subs[subscription_id]
            return True
        return False

    def add_preference(self, user_id: str, note: str) -> None:
        self._check_not_sensitive(note)
        notes = self._preferences.setdefault(user_id, [])
        if note not in notes:
            notes.append(note)

    def get_preferences(self, user_id: str, query: str = "", limit: int = 3) -> list[str]:
        notes = self._preferences.get(user_id, [])
        if not query:
            return notes[:limit]
        q_words = set(query.lower().split())
        scored = sorted(
            notes,
            key=lambda n: sum(1 for w in q_words if w in n.lower()),
            reverse=True,
        )
        return scored[:limit]

    def clear_user_data(self, user_id: str) -> None:
        """Right to be forgotten: deletes all stored data for a given user."""
        self._subscriptions.pop(user_id, None)
        self._preferences.pop(user_id, None)


def estimate_tokens(texts: list[str]) -> int:
    """Rough token estimator (~4 characters per token, as taught in Lesson 3)."""
    total_chars = sum(len(t) for t in texts)
    return max(1, (total_chars + 3) // 4)


def keep_newest(lines: list[str], token_budget: int) -> list[str]:
    """Keeps the newest messages that fit within the remaining token budget."""
    if token_budget <= 0:
        return []
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = estimate_tokens([line])
        if used + cost > token_budget:
            break
        kept.append(line)
        used += cost
    return list(reversed(kept))


def build_context(
    state: AgentState,
    memory_store: PersistentMemoryStore,
    strategy: Literal["full", "last_n", "notes", "summary_and_last_n"] = "full",
    last_n: int = 4,
    budget: int | None = None,
    query: str = "",
) -> list[str]:
    """Single context-builder function deciding what gets sent to the model."""
    if last_n < 0:
        raise ValueError("last_n must be non-negative")
    ctx: list[str] = [f"[system] {SYSTEM_PROMPT}"]

    if strategy == "full":
        chosen = state.messages
    elif strategy == "last_n":
        chosen = state.messages[-last_n:] if last_n else []
    elif strategy == "notes":
        prefs = memory_store.get_preferences(state.user_id, query=query, limit=3)
        for pref in prefs:
            ctx.append(f"[memory_note] {pref}")
        subs = memory_store.get_subscriptions(state.user_id)
        if subs:
            sub_summary = ", ".join(f"{s.service_name} ({s.amount} ILS)" for s in subs)
            ctx.append(f"[active_subs] {sub_summary}")
        chosen = state.messages[-last_n:] if last_n else []
    elif strategy == "summary_and_last_n":
        if state.summary:
            ctx.append(f"[summary] {state.summary}")
        chosen = state.messages[-last_n:] if last_n else []
    else:
        raise ValueError(f"Unknown context strategy: {strategy}")

    lines = [f"{m.role}: {m.content}" for m in chosen]

    if budget is not None:
        header_tokens = estimate_tokens(ctx)
        remaining = budget - header_tokens
        lines = keep_newest(lines, remaining)

    return ctx + lines


def answer_question_from_context(context_lines: list[str]) -> str:
    """Deterministic mock responder that answers strictly from the provided context lines.

    Demonstrates objectively what the model can vs. cannot answer under each context strategy.
    """
    joined = "\n".join(context_lines)
    budget_match = re.search(r"תקציב[^\d]{0,30}(\d+)\s*ש״ח", joined)
    has_no_lockin = "ללא התחייבות שנתית" in joined

    if budget_match and has_no_lockin:
        return (
            f"לפי ההקשר שבידי, תקציב המקסימום שלך הוא {budget_match.group(1)} ש״ח בחודש, "
            "והתנאי החשוב לך הוא חלופות ללא התחייבות שנתית."
        )
    if budget_match and not has_no_lockin:
        return (
            f"לפי ההקשר שבידי, תקציב המקסימום שלך הוא {budget_match.group(1)} ש״ח בחודש, "
            "אך לא מופיע בהקשר תנאי ההתחייבות שביקשת."
        )
    return "המידע על תקציב המקסימום החודשי ותנאי ההתחייבות אינו מופיע בהקשר הנוכחי — אנא צייני אותם שוב."


def run_context_comparison() -> list[dict[str, Any]]:
    """Runs the Milestone 1 empirical comparison: same question executed under full vs. reduced context."""
    memory = PersistentMemoryStore()
    user_id = "user_001"
    memory.add_preference(
        user_id, "תקציב מנויים מקסימלי: 250 ש״ח בחודש; העדפה לחלופות ללא התחייבות שנתית."
    )
    memory.upsert_subscription(
        user_id,
        SubscriptionRecord(
            subscription_id="sub_netflix",
            service_name="Netflix",
            amount=69.90,
            billing_cycle="monthly",
            category="streaming",
        ),
    )

    state = AgentState(
        thread_id="thread_compare_1",
        user_id=user_id,
        max_steps=20,
        summary=(
            "המשתמשת ציינה תקציב מנויים מקסימלי של 250 ש״ח בחודש והעדפה לחלופות ללא התחייבות שנתית. "
            "נבדקו מנויי Netflix (69.90 ש״ח), Spotify (21.90 ש״ח), Apple (40 ש״ח) וחדר כושר (250 ש״ח)."
        ),
    )

    conversation = [
        ("user", "היי, חשוב לי שתזכור: תקציב המנויים המקסימלי שלי הוא 250 ש״ח בחודש ואני מעדיפה חלופות ללא התחייבות שנתית."),
        ("assistant", "שמרתי את ההעדפות שלך: תקציב מקסימלי של 250 ש״ח בחודש וללא התחייבות שנתית. איך אוכל לעזור?"),
        ("user", "העליתי קובץ עסקאות עם החיובים שלי מהרבעון האחרון."),
        ("assistant", "זיהיתי 4 חיובים חוזרים: Netflix (69.90 ש״ח), Holmes Place (250 ש״ח), Apple (40 ש״ח) ו-Spotify (21.90 ש״ח)."),
        ("user", "החיוב של Apple הוא עבור iCloud+. אפשר לשמור את המנויים שזיהית."),
        ("assistant", "מצוין, המנויים המאושרים נשמרו בפרופיל שלך. סך ההוצאה החודשית הוא 381.80 ש״ח."),
        ("user", "תראה לי חלופות למנוי Netflix שלי."),
        ("assistant", "בקטלוג המקומי מצאתי Amazon Prime Video ב-20 ש״ח בחודש (חיסכון של 598.80 ש״ח בשנה)."),
        ("user", "ומה לגבי חדר הכושר Holmes Place?"),
        ("assistant", "בקטלוג המקומי מצאתי Profit Gym ב-150 ש״ח בחודש ללא התחייבות (חיסכון של 1,200 ש״ח בשנה)."),
        ("user", "מעולה, כמה אחסוך בסך הכול אם אעבור לשניהם?"),
        ("user", "רגע, לפני שנחליט — לפי מה שביקשתי בהתחלה, מה תקציב המקסימום שלי בחודש ומה התנאי החשוב לי בחלופות?"),
    ]

    for role, text in conversation:
        state = apply_event(state, role, text)  # type: ignore[arg-type]

    question = conversation[-1][1]

    strategies: list[tuple[str, Literal["full", "last_n", "notes", "summary_and_last_n"]]] = [
        ("היסטוריה מלאה (full)", "full"),
        ("4 הודעות אחרונות (last_n=4)", "last_n"),
        ("זיכרון מובנה + 4 אחרונות (notes)", "notes"),
        ("סיכום + 4 אחרונות (summary_and_last_n)", "summary_and_last_n"),
    ]

    results: list[dict[str, Any]] = []
    full_tokens = None

    for label, strat in strategies:
        ctx = build_context(state, memory, strategy=strat, last_n=4, query=question)
        actual_answer = answer_question_from_context(ctx)
        joined = "\n".join(ctx)
        chars = len(joined)
        tokens = estimate_tokens(ctx)
        if full_tokens is None:
            full_tokens = tokens

        is_correct = ("250" in actual_answer) and ("ללא התחייבות שנתית" in actual_answer)

        if strat == "full":
            lost = "לא אבד מידע (כל ההיסטוריה נשלחה, אך עלות הטוקנים גדלה בכל תור)"
        elif strat == "last_n":
            lost = "אבדו ההודעות הראשונות: תקציב המקסימום (250 ש״ח) והעדפת ׳ללא התחייבות שנתית׳"
        elif strat == "notes":
            lost = "אבדו פירוטי הביניים המילוליים של אמצע השיחה (למשל שאלת ההבהרה על Apple)"
        else:
            lost = "אבד הנוסח המדויק של הודעות הביניים (נשמר רק סיכום דחוס; דורש קריאת מודל לסיכום)"

        results.append(
            {
                "strategy_id": strat,
                "strategy_label": label,
                "context_messages": len(ctx),
                "chars": chars,
                "estimated_tokens": tokens,
                "pct_of_full": round((tokens / full_tokens) * 100, 1),
                "required_info_present": is_correct,
                "answer_correctness": "נכונה ומלאה" if is_correct else "שגויה / חסרה (המידע נפל מהחלון)",
                "actual_answer": actual_answer,
                "what_is_lost": lost,
            }
        )

    return results
