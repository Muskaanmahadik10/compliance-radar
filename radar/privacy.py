"""Outbound-text redaction: one policy for every piece of text that is sent to a hosted model.

Scope, stated exactly: e-mail addresses and phone numbers. Nothing else (names, postal addresses, national ids) is
detected. Dates are matched first and left alone, so "2026-11-13" or "13/11/2026" is never mistaken for a phone
number, and a phone match must contain at least 9 digits, so regulation numbers such as "2023/1542" or CAS numbers
such as "117-81-7" are not masked either.

Where it is applied: at the boundary, on the final text of every outbound message, not at each place a prompt is
assembled. `redact_messages` runs inside `call_structured` (decomposition, drafting, critic), in the classifier's LLM
tier and on the final chat messages, so event JSON, evidence headings, drafts and feedback are all covered. Originals
stay in the database for the human reviewer.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# At each position a date (ISO, or day/month-first with a 4-digit year) wins over a phone number.
_DATE_OR_PHONE = re.compile(
    r"(?P<date>(?<!\d)(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})(?!\d))"
    r"|(?P<phone>(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w))"
)
_MIN_PHONE_DIGITS = 9


def redact_pii(text: str) -> tuple[str, int]:
    """Mask e-mail addresses and phone numbers. Returns (text, number_redacted)."""
    if not text:
        return text, 0
    n = 0

    def _email(m: re.Match) -> str:
        nonlocal n
        n += 1
        return "[email]"

    def _num(m: re.Match) -> str:
        nonlocal n
        if m.group("date") or sum(ch.isdigit() for ch in m.group(0)) < _MIN_PHONE_DIGITS:
            return m.group(0)
        n += 1
        return "[phone]"

    return _DATE_OR_PHONE.sub(_num, _EMAIL.sub(_email, text)), n


def redact_messages(messages: list) -> tuple[list, int]:
    """Mask the text of every outbound message: LangChain messages (copied, not mutated) or plain strings."""
    out, total = [], 0
    for m in messages:
        if isinstance(m, str):
            text, n = redact_pii(m)
            out.append(text)
        elif isinstance(getattr(m, "content", None), str):
            text, n = redact_pii(m.content)
            out.append(m.model_copy(update={"content": text}) if n else m)
        else:  # multi-part content: mask each text part
            n, parts = 0, []
            for part in getattr(m, "content", None) or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    t, k = redact_pii(part["text"])
                    parts.append(part | {"text": t})
                    n += k
                else:
                    parts.append(part)
            out.append(m.model_copy(update={"content": parts}) if n else m)
        total += n
    return out, total
