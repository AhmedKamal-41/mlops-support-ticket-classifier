"""
Tests for the IT service desk workflow layer (src/service_desk.py).

These tests are pure logic - they don't touch the ML model - so they run fast
and check that every category routes correctly and the triage output is complete.
"""

import pytest

from src.config import LABELS
from src.service_desk import (
    SERVICE_DESK_RULES,
    LOW_CONFIDENCE_THRESHOLD,
    get_rule,
    triage_ticket,
    format_triage,
)


def test_every_label_has_a_rule():
    """Every ML category in config.LABELS must have a service desk rule."""
    for label in LABELS:
        assert label in SERVICE_DESK_RULES, f"No service desk rule for '{label}'"


def test_rules_have_all_required_fields():
    """Each rule must define priority, team, KB, first steps, and escalation."""
    required = {"priority", "routing_team", "kb_article", "first_steps", "escalation"}
    for label, rule in SERVICE_DESK_RULES.items():
        assert required <= set(rule), f"Rule '{label}' is missing fields"
        assert isinstance(rule["first_steps"], list) and rule["first_steps"]


def test_priorities_are_from_the_expected_set():
    valid = {"Low", "Medium", "High", "Critical"}
    for rule in SERVICE_DESK_RULES.values():
        assert rule["priority"] in valid


def test_triage_output_is_complete():
    """A triage result exposes the ML prediction plus every routing field."""
    triage = triage_ticket(
        "My Outlook is not syncing and I cannot receive emails.",
        "Microsoft 365 / Outlook",
        0.87,
    )
    for key in (
        "text", "predicted_category", "confidence", "priority",
        "routing_team", "suggested_kb", "first_steps", "escalation", "needs_review",
    ):
        assert key in triage


def test_triage_routes_outlook_to_m365():
    triage = triage_ticket("Outlook broken", "Microsoft 365 / Outlook", 0.9)
    assert triage["routing_team"] == "Microsoft 365 Support"
    assert triage["suggested_kb"] == "kb-outlook-sync.md"
    assert triage["priority"] == "Medium"
    assert triage["needs_review"] is False


def test_security_ticket_is_critical():
    triage = triage_ticket("phishing email", "Security / Phishing", 0.8)
    assert triage["priority"] == "Critical"
    assert triage["routing_team"] == "Security / SOC"


def test_low_confidence_flags_for_review():
    low = LOW_CONFIDENCE_THRESHOLD - 0.05
    triage = triage_ticket("something vague", "Hardware", low)
    assert triage["needs_review"] is True
    assert "confidence" in triage["escalation"].lower()


def test_high_confidence_does_not_flag_for_review():
    triage = triage_ticket("clear ticket", "Hardware", 0.95)
    assert triage["needs_review"] is False


def test_unknown_category_falls_back_to_default():
    rule = get_rule("Not A Real Category")
    assert rule["routing_team"] == "Service Desk (Tier 1)"


def test_format_triage_is_readable_text():
    text = format_triage(triage_ticket("VPN down", "VPN", 0.75))
    assert "ML Predicted Category" in text
    assert "Routing Team" in text
    assert "Network Operations" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
