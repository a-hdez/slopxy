"""Anthropic backend (official `anthropic` SDK: pip install slopxy[anthropic]) for scene
descriptions with Claude.

Profile settings:
  model        default claude-opus-5-5
  api_key_env  env var holding the key (default: the SDK's own lookup, ANTHROPIC_API_KEY first)
  effort       low (default) | medium | high | xhigh | max
  fallbacks    true (default): if the model declines, the API retries on a fallback model
  max_tokens   output limit including thinking (default 16000)
  concurrency  parallel requests (default 4)"""

from __future__ import annotations

from . import ArtDeclined, ArtError, ArtUnavailable, api_key, require_sdk

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def text_backend(settings: dict) -> "AnthropicTextBackend":
    return AnthropicTextBackend(settings)


class AnthropicTextBackend:
    name = "anthropic"
    paid = True

    def __init__(self, settings: dict):
        self.sdk = require_sdk("anthropic", "anthropic")
        key = api_key(settings) if settings.get("api_key_env") else None
        if settings.get("api_key_env") and not key:
            raise ArtUnavailable("the anthropic backend's key isn't set: the variable the "
                                 "profile's api_key_env names is empty or missing")
        # Without api_key_env the SDK finds the credentials itself (ANTHROPIC_API_KEY, ...).
        self.client = self.sdk.Anthropic(**({"api_key": key} if key else {}),
                                         max_retries=int(settings.get("max_retries", 3)),
                                         timeout=float(settings.get("timeout", 300)))
        if not (self.client.api_key or self.client.auth_token
                or getattr(self.client, "credentials", None)):
            raise ArtUnavailable("the anthropic backend has no API key: set ANTHROPIC_API_KEY "
                                 "(or the variable the profile's api_key_env names)")
        self.model = settings.get("model") or DEFAULT_MODEL
        self.effort = settings.get("effort", "low")
        self.fallbacks = bool(settings.get("fallbacks", True))
        self.max_tokens = int(settings.get("max_tokens", 16000))
        self.concurrency = int(settings.get("concurrency", 4))

    def complete(self, prompt: str, seed: int) -> str:
        # No seed in this API: descriptions vary per run, and the PNG keeps the text.
        kwargs = {"model": self.model, "max_tokens": self.max_tokens,
                  "messages": [{"role": "user", "content": prompt}]}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            if self.fallbacks:
                response = self.client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
            else:
                response = self.client.messages.create(**kwargs)
        except self.sdk.APIError as e:
            raise ArtError(f"Anthropic request failed: {e}") from None
        except TypeError as e:  # the SDK found no credentials at all
            if "authentication" not in str(e).lower():
                raise
            raise ArtError("the anthropic backend has no API key: set ANTHROPIC_API_KEY (or "
                           "the variable the profile's api_key_env names)") from None
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            why = getattr(details, "category", None) if details else None
            raise ArtDeclined("Claude declined to describe this card"
                              + (f" ({why})" if why else ""))
        return "".join(b.text for b in response.content if b.type == "text")
