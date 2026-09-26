from __future__ import annotations


class SessionCache(dict):
    """Per-request cache for foundation-model prompts/KV state/prosody lookahead."""
