"""
Qwen LLM wrapper using Hugging Face Inference API.
No local model download required — runs on HF servers.
"""
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv(override=True)



class QwenLLM:
    """
    Lightweight wrapper around Hugging Face Inference API for Qwen models.
    Uses your HF token to call the model remotely — no GPU or large RAM needed.
    """

    def __init__(self, model_name: str | None = None, max_tokens: int = 500):
        self.model_name = model_name or os.getenv(
            "QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )
        self.max_tokens = max_tokens

        token = os.getenv("huggingface_token") or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "Hugging Face token not found. "
                "Set 'huggingface_token' or 'HF_TOKEN' in your .env file, "
                "or configure it in your Streamlit Cloud Secrets."
            )


        self.client = InferenceClient(model=self.model_name, token=token)

    def invoke(self, prompt: str) -> str:
        """Send a prompt to the Qwen model via HF Inference API."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful RAG assistant. "
                    "Answer only from the provided context. "
                    "If the answer is missing, say you do not know."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

