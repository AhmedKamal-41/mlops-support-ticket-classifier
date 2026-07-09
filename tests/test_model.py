"""
Tests for the ML classifier + end-to-end triage (src/model.py).

These train a small model once and check that clear, keyword-obvious tickets are
classified into the right IT category and get sensible service desk routing.
"""

import pytest

from src.config import LABELS
from src.model import train_and_save, classify, classify_and_triage


@pytest.fixture(scope="module", autouse=True)
def trained_model():
    """Train the model once for this test module."""
    train_and_save(verbose=False)


def test_classify_returns_label_and_confidence():
    results = classify(["I can't connect to the VPN from home"])
    assert len(results) == 1
    label, confidence = results[0]
    assert label in LABELS
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize(
    "ticket, expected_category",
    [
        ("I forgot my password and need it reset", "Password Reset"),
        ("My account is locked after too many login attempts", "Account Lockout"),
        ("My Outlook is not syncing and I cannot receive emails", "Microsoft 365 / Outlook"),
        ("I can't connect to the VPN from home", "VPN"),
        ("The printer is jammed and won't print my documents", "Printer"),
        ("I received a suspicious phishing email asking for my password", "Security / Phishing"),
    ],
)
def test_clear_tickets_classify_correctly(ticket, expected_category):
    label, _ = classify([ticket])[0]
    assert label == expected_category


def test_classify_and_triage_is_end_to_end():
    """A single call should return the ML prediction plus routing fields."""
    result = classify_and_triage("My Outlook won't sync and I can't get emails")
    assert result["predicted_category"] == "Microsoft 365 / Outlook"
    assert result["routing_team"] == "Microsoft 365 Support"
    assert result["priority"] in {"Low", "Medium", "High", "Critical"}
    assert isinstance(result["first_steps"], list) and result["first_steps"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_batch_classification():
    tickets = ["reset my password", "the wifi keeps dropping", "install Zoom for me"]
    results = classify(tickets)
    assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
