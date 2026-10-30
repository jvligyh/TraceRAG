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
        max_retries: int = 2,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("DeepSeek API key must not be empty")

        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base.rstrip("/"),
            timeout=timeout_seconds,
            max_retries=max_retries,
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
                extra_body={"thinking": {"type": "disabled"}},
            )
        except APIError as exc:
            raise RuntimeError("DeepSeek API request failed; check connectivity, model and account") from exc

        return extract_chat_content(response)


def extract_chat_content(response: Any) -> str:
    """从 OpenAI SDK 返回的 ChatCompletion 对象中取出回答。"""

    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("DeepSeek response does not contain choices")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("DeepSeek response does not contain a message")

    value = getattr(message, "content", None)
    if isinstance(value, str) and value.strip():
        return value.strip()

    raise RuntimeError("DeepSeek response message does not contain text")
