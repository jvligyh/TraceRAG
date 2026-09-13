from __future__ import annotations

import hashlib


class MockLLMClient:
    """Deterministic client for local development and regression tests."""

    def complete(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        return f"Mock answer generated from the supplied context. trace={digest}"

