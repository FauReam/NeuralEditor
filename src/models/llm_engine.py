"""Local LLM inference with 4-bit quantization, CUDA-first."""

import gc
import os
import threading
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


def _detect_device() -> tuple[str, torch.dtype | None]:
    """Detect best available device and dtype."""
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception:
            major = 0
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return "cuda", dtype
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", None
    return "cpu", None


def _pick_attention():
    """Pick the best available attention implementation for speed."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        pass
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return "sdpa"
        except Exception:
            pass
    return "eager"


class LLMEngine:
    """4-bit quantised LLM wrapper — CUDA > MPS > CPU.

    Typical VRAM on RTX 4070 (12 GB):
        base model (4-bit)   ~3.5 GB
        KV cache (4K ctx)    ~1.5 GB
        total                 ~5.0 GB  (well under 12 GB)
    """

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
        self.device_type, self._compute_dtype = _detect_device()
        self._attention = _pick_attention()
        self._inference_lock = threading.Lock()

        self._load()

    def _load(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Download: python scripts/download_model.py\n"
                "Or link from ModelScope cache to models/Qwen2.5-7B-Instruct"
            )

        quant_type = "nf4"
        compute = self._compute_dtype or torch.float32

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=quant_type,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = dict(
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype="auto",
            use_cache=True,
        )

        # CUDA path — use device_map="auto" for multi-GPU or single-GPU
        if self.device_type == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"CUDA: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
            print(f"    attention: {self._attention}   compute dtype: {compute}")

            load_kwargs.update(
                device_map="auto",
                attn_implementation=self._attention,
            )

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, **load_kwargs,
                )
            except Exception as e:
                if self._attention == "flash_attention_2":
                    print(f"Flash Attention failed ({e}), falling back to sdpa")
                    load_kwargs["attn_implementation"] = "sdpa"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path, **load_kwargs,
                    )
                else:
                    raise
        else:
            print(f"Device: {self.device_type}  (4-bit CPU fallback)")
            load_kwargs.update(device_map={"": "cpu"})
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **load_kwargs,
            )

        # Attach LoRA if requested
        if self.lora_path and Path(self.lora_path).exists():
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model = self.model.merge_and_unload()
            print(f"Loaded + merged LoRA adapter: {self.lora_path}")

        self.model.eval()

        # Log VRAM usage on CUDA
        if self.device_type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"    VRAM: allocated={allocated:.1f} GB  reserved={reserved:.1f} GB", flush=True)

    def _truncate_messages(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> list[dict[str, str]]:
        """Truncate chat history to fit within context budget, keeping system prompt."""
        test = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if len(test) <= max_tokens:
            return messages

        system_msg = None
        chat_msgs = []
        for m in messages:
            if m["role"] == "system" and system_msg is None:
                system_msg = m
            else:
                chat_msgs.append(m)

        left, right = 1, len(chat_msgs)
        best: list[dict[str, str]] = []
        while left <= right:
            mid = (left + right) // 2
            candidate = ([system_msg] if system_msg else []) + chat_msgs[-mid:]
            ids = self.tokenizer.apply_chat_template(
                candidate, tokenize=True, add_generation_prompt=True
            )
            if len(ids) <= max_tokens:
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
        """Generate a response from a message list."""
        max_input_tokens = self.context_length - self.max_tokens
        msgs = self._truncate_messages(messages, max_input_tokens)
        if len(msgs) < len(messages):
            print(f"Warning: truncated {len(messages)} -> {len(msgs)} messages")

        inputs = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repeat_penalty,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs["streamer"] = streamer
            # Streaming requires a reader thread — not implemented in this synchronous path.
            # For real streaming use, call chat() inside a threading wrapper.

        with self._inference_lock:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)

        response_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response.strip()

    def unload(self) -> None:
        """Free VRAM by unloading the model."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Model unloaded, VRAM freed.")

    def vram_info(self) -> dict[str, float]:
        """Return current CUDA VRAM usage (GB)."""
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1),
        }
