"""
IT service desk workflow layer.

The machine learning model (see src/model.py and src/train.py) predicts *what
kind* of IT issue a ticket describes. This module is the service-desk layer that
sits on top of that prediction and decides *what to do about it*:

    ML model      -> predicted category  (e.g. "Microsoft 365 / Outlook")
    service desk  -> priority, routing team, KB article, first troubleshooting
                     steps, and an escalation recommendation

The mapping below is intentionally rule-based and easy to read. Real service
desks encode this kind of triage logic in their ITSM tool (ServiceNow, Jira
Service Management, Freshservice); here it lives in one plain dictionary so the
routing decisions stay transparent and explainable.
"""

from typing import Optional

# Confidence threshold below which we don't fully trust the ML prediction and
# ask a human analyst to confirm the category before auto-routing.
LOW_CONFIDENCE_THRESHOLD = 0.35

# One entry per category in src/config.py LABELS. Each rule describes how the
# service desk should handle a ticket the model assigned to that category.
SERVICE_DESK_RULES = {
    "Password Reset": {
        "priority": "Low",
        "routing_team": "Service Desk (Tier 1)",
        "kb_article": "kb-password-reset.md",
        "first_steps": [
            "Verify the user's identity per the ID-check policy",
            "Have the user try the self-service password reset portal",
            "Reset the password in Active Directory and set 'change at next logon'",
            "Confirm the user can sign in with the temporary password",
        ],
        "escalation": "Escalate to Identity & Access Management if the reset "
                      "fails repeatedly or the account may be compromised.",
    },
    "Account Lockout": {
        "priority": "Medium",
        "routing_team": "Service Desk (Tier 1)",
        "kb_article": "kb-account-lockout.md",
        "first_steps": [
            "Verify the user's identity",
            "Unlock the account in Active Directory",
            "Check for the lockout source (cached password on phone, mapped drive)",
            "Confirm the user can log in again",
        ],
        "escalation": "Escalate to Identity & Access Management if the account "
                      "keeps re-locking or lockouts come from an unknown source.",
    },
    "Microsoft 365 / Outlook": {
        "priority": "Medium",
        "routing_team": "Microsoft 365 Support",
        "kb_article": "kb-outlook-sync.md",
        "first_steps": [
            "Check the user's internet connection",
            "Restart Outlook and test send/receive",
            "Confirm mailbox access via Outlook on the Web (OWA)",
            "Check the Microsoft 365 service health dashboard",
        ],
        "escalation": "Escalate to Messaging/Exchange if mailbox access fails in "
                      "OWA or multiple users are affected.",
    },
    "Teams / OneDrive": {
        "priority": "Medium",
        "routing_team": "Microsoft 365 Support",
        "kb_article": "kb-teams-onedrive.md",
        "first_steps": [
            "Confirm the user is signed in with the correct work account",
            "Clear the Teams cache / restart the OneDrive sync client",
            "Check available OneDrive storage and pending sync errors",
            "Test in the Teams web client to isolate the desktop app",
        ],
        "escalation": "Escalate to Microsoft 365 engineering if sync errors "
                      "persist after a resync or a whole team is affected.",
    },
    "Network / Wi-Fi": {
        "priority": "High",
        "routing_team": "Network Operations",
        "kb_article": "kb-network-wifi.md",
        "first_steps": [
            "Confirm whether the issue affects one user or the whole area",
            "Check cabling / reconnect to the corporate Wi-Fi SSID",
            "Renew the IP address (ipconfig /release && /renew)",
            "Test connectivity to an internal and an external site",
        ],
        "escalation": "Escalate to Network Operations immediately if multiple "
                      "users or a whole floor/site have lost connectivity.",
    },
    "VPN": {
        "priority": "High",
        "routing_team": "Network Operations",
        "kb_article": "kb-vpn-connectivity.md",
        "first_steps": [
            "Confirm the user has a working internet connection first",
            "Verify credentials and that MFA is approved",
            "Reinstall / update the VPN client if needed",
            "Test connecting from a different network",
        ],
        "escalation": "Escalate to Network Operations if the VPN gateway is "
                      "unreachable or many remote users cannot connect.",
    },
    "Printer": {
        "priority": "Low",
        "routing_team": "Service Desk (Tier 1)",
        "kb_article": "kb-printing.md",
        "first_steps": [
            "Check the printer is powered on and shows ready (no jam/toner alert)",
            "Clear the stuck jobs in the print queue",
            "Reinstall the printer / update the driver on the user's PC",
            "Send a test print",
        ],
        "escalation": "Escalate to Desktop Support / vendor if there is a "
                      "hardware fault or the device stays offline after a restart.",
    },
    "Hardware": {
        "priority": "Medium",
        "routing_team": "Desktop Support",
        "kb_article": "kb-hardware-troubleshooting.md",
        "first_steps": [
            "Confirm the symptom and when it started",
            "Reseat cables / peripherals and try a different port or outlet",
            "Restart the device and test with a known-good accessory",
            "Check warranty status for a possible repair/replacement",
        ],
        "escalation": "Escalate to Desktop Support for a repair, loaner, or "
                      "asset replacement if the fault is confirmed hardware.",
    },
    "Software Installation": {
        "priority": "Low",
        "routing_team": "Desktop Support",
        "kb_article": "kb-software-install.md",
        "first_steps": [
            "Confirm the exact application and version required",
            "Check licensing and whether manager/software approval is needed",
            "Deploy via the software portal (SCCM/Intune) or manual install",
            "Launch the app to confirm it works",
        ],
        "escalation": "Escalate to Application Support if the install fails "
                      "repeatedly or a license/approval is missing.",
    },
    "Access Request": {
        "priority": "Medium",
        "routing_team": "Identity & Access Management",
        "kb_article": "kb-access-request.md",
        "first_steps": [
            "Confirm the exact system/role the user needs",
            "Obtain manager or data-owner approval",
            "Grant access via the correct security group",
            "Notify the user and confirm they can access the resource",
        ],
        "escalation": "Escalate to the application/data owner for approval on "
                      "privileged or sensitive access requests.",
    },
    "Shared Folder": {
        "priority": "Medium",
        "routing_team": "Identity & Access Management",
        "kb_article": "kb-shared-folder-access.md",
        "first_steps": [
            "Confirm the exact share/path and the access level needed",
            "Verify or re-map the network drive for the user",
            "Check and update NTFS/share permissions (with owner approval)",
            "Confirm the user can open and edit the files as required",
        ],
        "escalation": "Escalate to the file-share owner for approval, or to "
                      "Storage/Server team if the file server is unreachable.",
    },
    "Security / Phishing": {
        "priority": "Critical",
        "routing_team": "Security / SOC",
        "kb_article": "kb-phishing-response.md",
        "first_steps": [
            "Tell the user NOT to click links or enter credentials further",
            "Isolate the device from the network if malware is suspected",
            "Reset the user's password if credentials were entered",
            "Report the message via the Phish Report button / to the SOC",
        ],
        "escalation": "Escalate to the Security / SOC team immediately; treat "
                      "credential entry or malware as a security incident.",
    },
    "New Hire Setup": {
        "priority": "Medium",
        "routing_team": "IT Onboarding",
        "kb_article": "kb-new-hire-setup.md",
        "first_steps": [
            "Confirm start date, department, and required access from HR",
            "Create the user account, mailbox, and standard group memberships",
            "Provision and image the laptop plus any peripherals/phone",
            "Prepare a first-day login and welcome/setup instructions",
        ],
        "escalation": "Escalate to the onboarding coordinator if the start date "
                      "is at risk or approvals for special access are missing.",
    },
    "Offboarding": {
        "priority": "High",
        "routing_team": "IT Onboarding",
        "kb_article": "kb-offboarding.md",
        "first_steps": [
            "Confirm the leaver's name and effective date/time with HR",
            "Disable the account and revoke VPN, email, and app access",
            "Archive the mailbox and back up/transfer needed files",
            "Arrange return and wipe of the laptop and devices",
        ],
        "escalation": "Escalate to Security for immediate access revocation on "
                      "urgent or for-cause terminations.",
    },
    "Escalation Required": {
        "priority": "Critical",
        "routing_team": "Major Incident / On-call",
        "kb_article": "kb-major-incident.md",
        "first_steps": [
            "Confirm scope: how many users/sites/services are affected",
            "Raise a major incident and open a bridge call",
            "Notify the on-call engineer and incident manager",
            "Post a status update to stakeholders and start the timeline",
        ],
        "escalation": "Engage the Major Incident process now and page the "
                      "on-call team; this is a business-impacting outage.",
    },
}

# Fallback rule used when the model returns a category we don't have a rule for
# (should not normally happen) or when we can't confidently route.
DEFAULT_RULE = {
    "priority": "Medium",
    "routing_team": "Service Desk (Tier 1)",
    "kb_article": "kb-general-triage.md",
    "first_steps": [
        "Review the ticket and gather any missing details from the user",
        "Categorise the request manually",
        "Route to the appropriate support team",
    ],
    "escalation": "Escalate to a senior analyst if the correct category or "
                  "owning team is unclear.",
}


def get_rule(category: str) -> dict:
    """Return the service desk rule for a predicted category (or the default)."""
    return SERVICE_DESK_RULES.get(category, DEFAULT_RULE)


def triage_ticket(
    text: str,
    predicted_category: str,
    confidence: Optional[float] = None,
) -> dict:
    """
    Combine an ML prediction with the service desk rules into a full triage.

    Args:
        text: The original ticket text.
        predicted_category: The category predicted by the ML model.
        confidence: Optional model confidence (0-1) for the predicted category.

    Returns:
        A dictionary with the ML prediction plus the service-desk decision:
        predicted_category, confidence, priority, routing_team, suggested_kb,
        first_steps, escalation, and needs_review.

    Example:
        >>> triage = triage_ticket(
        ...     "My Outlook is not syncing and I cannot receive emails.",
        ...     "Microsoft 365 / Outlook",
        ...     0.87,
        ... )
        >>> triage["routing_team"]
        'Microsoft 365 Support'
    """
    rule = get_rule(predicted_category)

    # Tie the ML confidence into the workflow: if the model isn't confident, we
    # flag the ticket for a human to confirm the category before auto-routing.
    needs_review = confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD

    escalation = rule["escalation"]
    if needs_review:
        escalation = (
            f"Low model confidence ({confidence:.0%}) - have a service desk "
            f"analyst confirm the category before routing. " + escalation
        )

    return {
        "text": text,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "priority": rule["priority"],
        "routing_team": rule["routing_team"],
        "suggested_kb": rule["kb_article"],
        "first_steps": list(rule["first_steps"]),
        "escalation": escalation,
        "needs_review": needs_review,
    }


def format_triage(triage: dict) -> str:
    """Render a triage dictionary as a readable, service-desk-style summary."""
    conf = triage.get("confidence")
    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"

    lines = [
        f"Ticket: {triage['text']}",
        "-" * 60,
        f"ML Predicted Category : {triage['predicted_category']}",
        f"Confidence            : {conf_str}",
        f"Priority              : {triage['priority']}",
        f"Routing Team          : {triage['routing_team']}",
        f"Suggested KB          : {triage['suggested_kb']}",
        "First Steps           :",
    ]
    for i, step in enumerate(triage["first_steps"], start=1):
        lines.append(f"    {i}. {step}")
    lines.append(f"Escalation            : {triage['escalation']}")
    if triage.get("needs_review"):
        lines.append("Review Flag           : LOW CONFIDENCE - confirm category manually")
    return "\n".join(lines)
