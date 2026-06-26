"""
Qwen LLM wrapper using Hugging Face Inference API.
No local model download — the model runs on HF servers.
"""
import os
import time
import logging

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

logger = logging.getLogger(__name__)

# Load .env when running locally; silently ignored on Streamlit Cloud.
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


def _resolve_token() -> str:
    """
    Resolve the Hugging Face access token.

    Resolution order:
      1. Streamlit Cloud Secrets  (st.secrets["huggingface_token"])
      2. Environment variable     (huggingface_token / HF_TOKEN)

    Raises ValueError if no token is found.
    """
    # 1. Streamlit Cloud secrets (only available inside a Streamlit process)
    try:
        import streamlit as st
        token = (
            st.secrets.get("huggingface_token")
            or st.secrets.get("HF_TOKEN")
        )
        if token:
            return token
    except Exception:
        pass

    # 2. Environment variable fallback
    token = os.getenv("huggingface_token") or os.getenv("HF_TOKEN")
    if token:
        return token

    raise ValueError(
        "Hugging Face token not found.\n"
        "• Local:           set 'huggingface_token' in your .env file.\n"
        "• Streamlit Cloud: add 'huggingface_token' in App Settings → Secrets."
    )


class QwenLLM:
    """
    Lightweight wrapper around the Hugging Face Inference API for chat models.

    Features:
    - Zero local model download — inference runs on HF servers.
    - Automatic retry with exponential back-off for transient errors
      (rate-limit 429, server-busy 503).
    - Configurable request timeout to prevent indefinite hangs.
    """

    #: HTTP status codes that are safe to retry.
    _RETRYABLE_CODES = {429, 503}
    #: Maximum number of retry attempts.
    _MAX_RETRIES = 3
    #: Initial back-off delay in seconds (doubles on each retry).
    _BACKOFF_BASE = 2.0

    def __init__(
        self,
        model_name: str | None = None,
        max_tokens: int = 512,
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )
        self.max_tokens = max_tokens
        self.timeout = timeout

        token = _resolve_token()
        self.client = InferenceClient(
            model=self.model_name,
            token=token,
            timeout=self.timeout,
        )

    def invoke(self, prompt: str) -> str:
        """
        Send a prompt to the model and return the generated text.

        Retries automatically on transient HTTP errors (429, 503).
        Raises RuntimeError on authentication failures or permanent errors.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise RAG assistant. "
                    "Answer strictly from the provided context. "
                    "If the answer is not present, say: "
                    "'I don't know based on the document.'"
                ),
            },
            {"role": "user", "content": prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                )
                content = response.choices[0].message.content or ""
                return content.strip()

            except HfHubHTTPError as exc:
                status = getattr(exc.response, "status_code", None)

                if status == 401:
                    raise RuntimeError(
                        "Authentication failed (401). Your Hugging Face token is "
                        "invalid or has been revoked.\n"
                        "Generate a new token at https://huggingface.co/settings/tokens "
                        "and update your Secrets."
                    ) from exc

                if status in self._RETRYABLE_CODES:
                    wait = self._BACKOFF_BASE ** attempt
                    logger.warning(
                        "HF API returned %s on attempt %d/%d — retrying in %.1fs",
                        status, attempt + 1, self._MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    last_error = exc
                    continue

                # Non-retryable HTTP error
                raise RuntimeError(
                    f"Hugging Face API error ({status}): {exc}"
                ) from exc

            except TimeoutError as exc:
                wait = self._BACKOFF_BASE ** attempt
                logger.warning(
                    "Request timed out on attempt %d/%d — retrying in %.1fs",
                    attempt + 1, self._MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_error = exc

        raise RuntimeError(
            f"HF Inference API failed after {self._MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )
