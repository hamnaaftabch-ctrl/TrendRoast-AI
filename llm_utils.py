"""llm_utils.py - CrewAI LLM backed directly by the official Groq SDK.

Uses ONLY the model `openai/gpt-oss-120b`. No LiteLLM involved, which avoids
LiteLLM import problems and the `cache_breakpoint` field Groq rejects.
"""
from typing import Any

from crewai import BaseLLM
from groq import Groq

MODEL = "openai/gpt-oss-120b"       # the only model this app uses
REASONING_EFFORT = "medium"          # "low" = faster, "high" = deeper


class GroqSDKLLM(BaseLLM):
    llm_type: str = "groq-sdk"

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Keep only what Groq accepts (drops CrewAI-internal fields).
        clean = [{"role": m["role"], "content": m.get("content") or ""} for m in messages]

        kwargs: dict[str, Any] = dict(
            model=MODEL,
            messages=clean,
            temperature=self.temperature if self.temperature is not None else 1,
            max_completion_tokens=int(self.max_tokens or 2048),
            top_p=1,
            reasoning_effort=REASONING_EFFORT,
            stream=False,
        )
        if self.stop:
            kwargs["stop"] = self.stop[:4]

        client = Groq(api_key=self.api_key)
        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""

    def supports_function_calling(self) -> bool:
        return False  # our agents use plain text; no tool-call JSON to get wrong

    def get_context_window_size(self) -> int:
        return 131072


def build_llm(api_key: str, max_tokens: int = 2048) -> GroqSDKLLM:
    return GroqSDKLLM(model=MODEL, api_key=api_key, temperature=0.8, max_tokens=max_tokens)
