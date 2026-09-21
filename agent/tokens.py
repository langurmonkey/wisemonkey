"""Token counting for chat memory accounting.

Uses tiktoken (already a dependency of the vector store) for exact counts
with the OpenAI o200k encoding, which is a good approximation for
OpenAI-compatible endpoints and Claude-family models alike (within a few
percent for English and code). Falls back to a chars/4 estimate when
tiktoken is unavailable, so the agent still works in minimal installs.
"""

from __future__ import annotations

# Chars-per-token used by the fallback estimate and for legacy config
# conversion (agent.max_chat_history was defined in characters).
CHARS_PER_TOKEN = 4

_encoding = None
_encoding_loaded = False


def _get_encoding():
    """Load and cache the tiktoken encoding (None if unavailable)."""
    global _encoding, _encoding_loaded
    if not _encoding_loaded:
        _encoding_loaded = True
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("o200k_base")
        except Exception:
            _encoding = None
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens in text, exactly if tiktoken is available.

    Falls back to the chars/4 estimate otherwise. Empty/None input costs 0.
    """
    if not text:
        return 0
    enc = _get_encoding()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // CHARS_PER_TOKEN)
