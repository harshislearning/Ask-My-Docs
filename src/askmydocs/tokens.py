"""Token counting shared across layers.

Ingestion sizes chunks with the embedding model's real tokenizer; the
generation layer only needs a cheap estimate to keep the prompt inside the
context window, and must not pull in a tokenizer download to do it.
"""

from __future__ import annotations

#: English prose averages close to four characters per token across the
#: tokenizers used here. Good enough for budgeting, never for hard limits.
_CHARS_PER_TOKEN = 4


def heuristic_token_count(text: str) -> int:
    """Approximate token count from character length."""
    return max(1, len(text) // _CHARS_PER_TOKEN) if text else 0
