from __future__ import annotations

from typing import Any

from openai import APIError, OpenAI


class DeepSeekLLMClient:
    """DeepSeek OpenAI-compatible Chat Completions 客户端。"""

    def __init__(
        self,
        api_key: str,
        *,
        api_base: str = "https://api.deepseek.com",
        model: str = "deepseek-flash",
        timeout_seconds: float = 60.0,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("DeepSeek API key must not be empty")

        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base.rstrip("/"),
            timeout=timeout_seconds,
        )

    def complete(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "只根据用户提供的资料简洁作答。"},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )
        except APIError as exc:
            raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc

        return extract_chat_content(response)


def extract_chat_content(response: Any) -> str:
    """从 OpenAI SDK 返回的 ChatCompletion 对象中取出回答。"""

    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("DeepSeek response does not contain choices")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("DeepSeek response does not contain a message")

    for key in ("content", "reasoning_content"):
        value = getattr(message, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        value = model_extra.get("reasoning_content")
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise RuntimeError("DeepSeek response message does not contain text")
