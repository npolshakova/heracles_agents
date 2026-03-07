from typing import Literal, Optional

import openai
from pydantic import Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings


class OpenaiCompletionsClientConfig(BaseSettings):
    """OpenAI-compatible client that uses the Chat Completions API.

    This works with any provider routed through agentgateway (OpenAI, Anthropic,
    Gemini, Ollama, etc.) since /v1/chat/completions is universally supported.
    """

    client_type: Literal["openai_completions"]
    timeout: int
    auth_key: SecretStr = Field(alias="HERACLES_OPENAI_API_KEY", exclude=True)
    base_url: Optional[str] = None
    default_headers: Optional[dict[str, str]] = None
    _client: object = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        kwargs = {
            "api_key": self.auth_key.get_secret_value(),
            "timeout": self.timeout,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        self._client = openai.OpenAI(**kwargs)

    def call(self, model_info, tools, response_format, messages):
        converted = _convert_developer_to_system(messages)

        kwargs = {
            "model": model_info.model,
            "messages": converted,
            "temperature": model_info.temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = False

        return self._client.chat.completions.create(**kwargs)


def _convert_developer_to_system(messages):
    """The heracles prompt builder uses 'developer' role (OpenAI Responses API).

    Chat Completions expects 'system' for broad provider compatibility.
    """
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "developer":
            out.append({**m, "role": "system"})
        else:
            out.append(m)
    return out
