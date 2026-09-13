from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Small provider boundary kept stable across mock and MaaS clients."""

    def complete(self, prompt: str) -> str:
        """Return one text completion for the supplied prompt."""

