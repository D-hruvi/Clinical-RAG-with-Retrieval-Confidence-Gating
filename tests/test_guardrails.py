from src.guardrails.guardrails import check_input, check_output, redact_pii


def test_empty_query_rejected():
    result = check_input("")
    assert not result.allowed


def test_oversized_query_rejected():
    result = check_input("a" * 10_000)
    assert not result.allowed


def test_pii_query_rejected():
    result = check_input("What treatment does patient john@example.com need?")
    assert not result.allowed
    assert "identifying information" in result.reason


def test_clinical_query_allowed():
    result = check_input("What is the recommended treatment for hypertension?")
    assert result.allowed
    assert result.reason is None


def test_off_topic_query_flagged_not_blocked():
    result = check_input("What's the weather like today in Jaipur?")
    assert result.allowed
    assert result.reason == "off_topic_warning"


def test_output_blocked_below_threshold():
    result = check_output(top_score=0.2, source_nodes=[object()])
    assert not result.allowed


def test_output_blocked_when_no_sources():
    result = check_output(top_score=0.99, source_nodes=[])
    assert not result.allowed


def test_output_allowed_above_threshold():
    result = check_output(top_score=0.9, source_nodes=[object()])
    assert result.allowed


def test_redact_pii_email():
    redacted = redact_pii("Contact patient at john@example.com for follow-up.")
    assert "example.com" not in redacted
    assert "[redacted]" in redacted
