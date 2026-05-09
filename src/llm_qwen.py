import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class QwenLLM:
    """
    Minimal wrapper around Hugging Face Qwen Instruct models.
    Recommended for Colab: Qwen/Qwen2.5-0.5B-Instruct or 1.5B on GPU.
    """
    def __init__(self, model_name: str | None = None, max_new_tokens: int = 350):
        self.model_name = model_name or os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )

    def invoke(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful RAG assistant. Answer only from the provided context. If the answer is missing, say you do not know."},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        out = self.pipe(formatted)[0]["generated_text"]
        return out[len(formatted):].strip() if out.startswith(formatted) else out.strip()
