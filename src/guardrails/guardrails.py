"""API-boundary guardrails.

Input guardrail: rejects malformed / oversized / clearly off-topic queries
before they ever reach the LLM (saves cost, reduces attack surface).

Output guardrail: this is the differentiating piece. Instead of trusting the
LLM's fluent answer, it checks the *retrieval* evidence backing that answer.
If the best-matching source chunk scores below MIN_RETRIEVAL_SCORE, the system
refuses to answer rather than let the LLM generate an unsupported (and in a
clinical context, potentially dangerous) response. This is a cheap,
deterministic proxy for faithfulness that runs on every request; the LLM-based
faithfulness score in src/evaluation is the expensive, offline version used to
validate that this cheap proxy is actually catching the right cases.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

from src import config

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),          # SSN-like
    re.compile(r"\b\d{10}\b"),                       # bare 10-digit phone/ID
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),     # email address
]

_CLINICAL_HINT_WORDS = {
    "guideline", "treatment", "diagnosis", "dose", "dosage", "therapy",
    "patient", "symptom", "recommend", "management", "screening", "risk",
    "disease", "condition", "medication", "drug", "clinical", "chronic",
    "hypertension", "diabetes", "cancer", "infection", "prevention",
}


@dataclass
class GuardrailResult:
    allowed: bool
    reason: Optional[str] = None
    sanitized_text: Optional[str] = None


def check_input(query: str) -> GuardrailResult:
    if not query or not query.strip():
        return GuardrailResult(allowed=False, reason="Empty query.")

    if len(query) > config.MAX_QUERY_CHARS:
        return GuardrailResult(
            allowed=False,
            reason=f"Query exceeds {config.MAX_QUERY_CHARS} characters.",
        )

    for pattern in _PII_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(
                allowed=False,
                reason="Query appears to contain personal identifying information. "
                "Please remove names, emails, phone numbers, or ID numbers.",
            )

    # Soft scope check: this is a heuristic, not a hard block — it flags
    # obviously off-topic queries without needing an extra LLM call.
    lowered = query.lower()
    if not any(word in lowered for word in _CLINICAL_HINT_WORDS):
        return GuardrailResult(
            allowed=True,
            reason="off_topic_warning",
            sanitized_text=query.strip(),
        )

    return GuardrailResult(allowed=True, sanitized_text=query.strip())


def redact_pii(text: str) -> str:
    for pattern in _PII_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


def check_output(top_score: float, source_nodes: List) -> GuardrailResult:
    if not source_nodes or top_score < config.MIN_RETRIEVAL_SCORE:
        return GuardrailResult(
            allowed=False,
            reason=(
                f"top_retrieval_score={top_score:.3f} below threshold "
                f"{config.MIN_RETRIEVAL_SCORE}"
            ),
        )
    return GuardrailResult(allowed=True)
