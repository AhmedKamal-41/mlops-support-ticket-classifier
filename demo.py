"""
CLI demo for the IT Service Desk Ticket Classification & Routing System.

Runs the full pipeline on support tickets:
    1. ML model (TF-IDF + LogisticRegression) predicts the IT category
    2. Service desk layer applies priority, routing, KB, first steps, escalation

Usage:
    python demo.py                       # run on a set of built-in example tickets
    python demo.py "My Outlook won't sync and I can't get emails"
    python demo.py --interactive         # type tickets and get triage live
    python demo.py --retrain             # retrain the local model first

The first run trains the model automatically and caches it in models/.
"""

import argparse
import sys

from src.model import train_and_save, classify_and_triage
from src.service_desk import format_triage


# A few representative tickets covering different teams and priorities.
EXAMPLE_TICKETS = [
    "My Outlook is not syncing and I cannot receive emails.",
    "I forgot my password and can't log in.",
    "I can't connect to the VPN from home.",
    "The printer on the third floor is not printing.",
    "I received a suspicious email asking me to confirm my password.",
    "A new hire starts Monday and needs a laptop and accounts set up.",
    "The entire office has lost internet, nobody can work.",
]


def run_examples() -> None:
    print("=" * 60)
    print("IT Service Desk Ticket Classification & Routing System")
    print("ML category prediction + service desk triage")
    print("=" * 60)
    for ticket in EXAMPLE_TICKETS:
        triage = classify_and_triage(ticket)
        print()
        print(format_triage(triage))
        print()


def run_single(text: str) -> None:
    print(format_triage(classify_and_triage(text)))


def run_interactive() -> None:
    print("Interactive mode - type a support ticket and press Enter.")
    print("Type 'quit' or press Ctrl+C to exit.\n")
    try:
        while True:
            text = input("Ticket> ").strip()
            if text.lower() in {"quit", "exit", "q"}:
                break
            if not text:
                continue
            print()
            print(format_triage(classify_and_triage(text)))
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticket", nargs="?", help="A single ticket text to triage")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model first")
    args = parser.parse_args()

    if args.retrain:
        train_and_save()
        print()

    if args.interactive:
        run_interactive()
    elif args.ticket:
        run_single(args.ticket)
    else:
        run_examples()


if __name__ == "__main__":
    sys.exit(main())
