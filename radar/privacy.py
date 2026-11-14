"""Outbound-text redaction: one policy for every piece of text that is sent to a hosted model.

Scope, stated exactly: e-mail addresses and phone numbers. Nothing else (names, postal addresses, national ids) is
detected. Dates are set aside before phones are matched, so "2026-11-13" or "13/11/2026" is never mistaken for (or
swallowed by) a phone number; a phone must contain at least 9 digits and cannot run into a "/", so regulation numbers
such as "2023/1542" and CAS numbers such as "117-81-7" survive, also right after a phone. Supported phone shapes:
digit groups joined by single spaces, dots, hyphens or a line break, an optional "+" and bracketed area code.

Where it is applied: at the boundary, on the final text of every outbound message, not at each place a prompt is
assembled. `redact_messages` runs inside `call_structured` (decomposition, drafting, critic), in the classifier's LLM
tier and on the final chat messages, so event JSON, evidence headings, drafts and feedback are all covered. Originals
stay in the database for the human reviewer.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Dates (ISO, or day/month-first with a 4-digit year) are set aside before phones are looked for, so a phone number
# can never run into a following date.
_DATE = re.compile(r"(?<!\d)(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})(?!\d)")
# A phone: optional "+", an optional bracketed area code, then digit groups joined by at most one space, dot, hyphen
# or line break. It may not touch a Latin letter or digit on either side (a Han character may: "電話0912345678"), and
# it may not run into a "/", so "... 678 2023/1542" stops before the regulation number.
_PHONE = re.compile(r"(?<![A-Za-z0-9+])\+?(?:\(\d{1,4}\)[ ]?)?\d{1,5}(?:(?:[ .\-]|\r?\n)?\(?\d{1,5}\)?){0,6}(?![A-Za-z0-9/])")
_MIN_PHONE_DIGITS = 9
_HOLD = re.compile("\x00([-])\x00")


def redact_pii(text: str) -> tuple[str, int]:
    """Mask e-mail addresses and phone numbers (at least 9 digits). Returns (text, number_redacted)."""
    if not text:
        return text, 0
    text, n = _EMAIL.subn("[email]", text)
    held: list[str] = []

    def _hold(m: re.Match) -> str:
        held.append(m.group(0))
        return f"\x00{chr(0xE000 + len(held) - 1)}\x00"  # placeholder with no digits, so no phone can absorb it

    def _num(m: re.Match) -> str:
        nonlocal n
        if sum(ch.isdigit() for ch in m.group(0)) < _MIN_PHONE_DIGITS:
            return m.group(0)
        n += 1
        return "[phone]"

    text = _PHONE.sub(_num, _DATE.sub(_hold, text))
    return _HOLD.sub(lambda m: held[ord(m.group(1)) - 0xE000], text), n


def redact_messages(messages: list) -> tuple[list, int]:
    """Mask the text of every outbound message: LangChain messages (copied, not mutated) or plain strings."""
    out, total = [], 0
    for m in messages:
        if isinstance(m, str):
            text, n = redact_pii(m)
            out.append(text)
        elif isinstance(m, tuple) and len(m) == 2 and isinstance(m[1], str):  # LangChain shorthand: ("human", "text")
            text, n = redact_pii(m[1])
            out.append((m[0], text))
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
