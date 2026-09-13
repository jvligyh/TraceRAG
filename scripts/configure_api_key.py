from __future__ import annotations

import getpass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    api_key = getpass.getpass("DeepSeek API key: ").strip()
    if not api_key:
        raise SystemExit("API key must not be empty")

    content = (
        "DEEPSEEK_API_BASE=https://api.deepseek.com\n"
        "DEEPSEEK_MODEL=deepseek-flash\n"
        f"DEEPSEEK_API_KEY={api_key}\n"
    )
    (ROOT / ".env").write_text(content, encoding="utf-8")
    print("DeepSeek API key saved to the Git-ignored .env file.")


if __name__ == "__main__":
    main()
