"""llm_utils.py - Groq LLM wrapper that fixes the 'cache_breakpoint' 400 error.

CrewAI 1.15+ tags system messages with a `cache_breakpoint` flag (for prompt
caching). Its native providers strip it, but the LiteLLM route used for Groq
does not, so Groq rejects the request. We strip it before sending.
"""
from crewai import LLM

DEFAULT_MODEL = "groq/openai/gpt-oss-120b"
_UNSUPPORTED_KEYS = ("cache_breakpoint",)


class GroqLLM(LLM):
    def _format_messages_for_provider(self, messages):
        formatted = super()._format_messages_for_provider(messages)
        cleaned = []
        for m in formatted:
            m = dict(m)
            for key in _UNSUPPORTED_KEYS:
                m.pop(key, None)
            cleaned.append(m)
        return cleaned


def build_llm(api_key: str, model: str = DEFAULT_MODEL, max_tokens: int = 1500) -> LLM:
    return GroqLLM(model=model, api_key=api_key, temperature=0.7, max_tokens=max_tokens)
