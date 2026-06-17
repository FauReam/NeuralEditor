"""Local LLM inference with optional LoRA."""

import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel


class LLMEngine:
    """Wrapper for local quantized LLM with optional LoRA adapter."""

    def __init__(
        self,
        model_path: str,
        context_length: int = 4096,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        lora_path: str | None = None,
    ):
        self.model_path = model_path
        self.context_length = context_length
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.lora_path = lora_path

        self.model: Any = None
        self.tokenizer: Any = None
        self._load()

    def _load(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Run: python scripts/download_model.py"
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
        )

        if self.lora_path and Path(self.lora_path).exists():
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            print(f"Loaded LoRA adapter: {self.lora_path}")

        self.model.eval()

    def _truncate_messages(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> list[dict[str, str]]:
        """Truncate messages to fit within max_tokens, keeping system first."""
        # Tokenize to count
        test = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if len(test) <= max_tokens:
            return messages

        # Keep system prompt if present
        system_msg = None
        chat_msgs = []
        for m in messages:
            if m["role"] == "system" and system_msg is None:
                system_msg = m
            else:
                chat_msgs.append(m)

        # Binary search for max messages that fit
        left, right = 1, len(chat_msgs)
        best = []
        while left <= right:
            mid = (left + right) // 2
            candidate = ([system_msg] if system_msg else []) + chat_msgs[-mid:]
            test_ids = self.tokenizer.apply_chat_template(
                candidate, tokenize=True, add_generation_prompt=True
            )
            if len(test_ids) <= max_tokens:
                best = candidate
                left = mid + 1
            else:
                right = mid - 1

        return best if best else ([system_msg] if system_msg else [])

    def chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
    ) -> str:
        """Generate a response from a message list.

        Messages should already include system prompt if needed.
        """
        max_input_tokens = self.context_length - self.max_tokens
        msgs = self._truncate_messages(messages, max_input_tokens)
        if len(msgs) < len(messages):
            print(f"Warning: truncated {len(messages)} -> {len(msgs)} messages to fit context")

        inputs = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repeat_penalty,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs["streamer"] = streamer
            # Would need threading for real streaming

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        response_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response.strip()

    def unload_lora(self) -> None:
        """Remove LoRA adapter, revert to base model."""
        if isinstance(self.model, PeftModel):
            self.model = self.model.unload()
            print("LoRA unloaded.")
