"""Configuration-driven LoRA trainer with sensible defaults for 4070."""

import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer


class LossHistoryCallback(TrainerCallback):
    """Callback that records training loss history."""

    def __init__(self, on_step=None):
        self.history: list[dict] = []
        self.on_step = on_step  # optional fn(step, loss)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            entry = {"step": state.global_step, "loss": round(logs["loss"], 6)}
            self.history.append(entry)
            print(f"Step {state.global_step}: loss={logs['loss']:.4f}")
            if self.on_step:
                self.on_step(state.global_step, logs["loss"])

    def get_final_loss(self) -> float | None:
        if not self.history:
            return None
        return self.history[-1]["loss"]

    def get_loss_trend(self) -> list[float]:
        return [h["loss"] for h in self.history]


def load_jsonl(path: str | Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_chat_to_text(example: dict, tokenizer: AutoTokenizer) -> str:
    """Convert chat messages to a single text string using chat template."""
    messages = example.get("messages", example.get("text"))
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    raise ValueError(f"Unknown data format: {example}")


class LoRATrainer:
    """Opinionated LoRA trainer optimized for single-GPU (4070)."""

    DEFAULT_CONFIG = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "lora_output",
        "dataset_path": "data/romance_chat_sample.jsonl",
        "lora": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.0,
            "target_modules": "all-linear",
            "bias": "none",
        },
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        },
        "training": {
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "max_grad_norm": 0.3,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_strategy": "epoch",
            "bf16": True,
            "group_by_length": False,
            "optim": "paged_adamw_8bit",
        },
        "max_seq_length": 512,
        "early_stopping_patience": None,
    }

    def __init__(self, config: dict[str, Any] | None = None, progress_callback=None):
        self.cfg = self._merge_config(config or {})
        self.model: Any = None
        self.tokenizer: Any = None
        self.trainer: SFTTrainer | None = None
        self.loss_history: list[dict] = []
        self.progress_callback = progress_callback  # fn(step, loss) or None

    def _merge_config(self, user_cfg: dict[str, Any]) -> dict[str, Any]:
        """Deep merge user config over defaults."""
        import copy
        cfg = copy.deepcopy(self.DEFAULT_CONFIG)
        for key, val in user_cfg.items():
            if isinstance(val, dict) and key in cfg:
                cfg[key].update(val)
            else:
                cfg[key] = val
        return cfg

    def load_base_model(self) -> None:
        """Load and quantize the base model."""
        q_cfg = self.cfg["quantization"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=q_cfg["load_in_4bit"],
            bnb_4bit_compute_dtype=q_cfg["bnb_4bit_compute_dtype"],
            bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        )

        print(f"Loading base model: {self.cfg['model_name']}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["model_name"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_name"],
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = prepare_model_for_kbit_training(self.model)

    def apply_lora(self) -> None:
        """Attach LoRA adapters."""
        l_cfg = self.cfg["lora"]
        lora_config = LoraConfig(
            r=l_cfg["r"],
            lora_alpha=l_cfg["alpha"],
            target_modules=l_cfg["target_modules"],
            lora_dropout=l_cfg["dropout"],
            bias=l_cfg["bias"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def load_dataset(self) -> Dataset:
        """Load and preprocess dataset."""
        path = self.cfg["dataset_path"]
        print(f"Loading dataset from {path}")
        raw_data = load_jsonl(path)

        # Convert to text using chat template
        texts = []
        for ex in raw_data:
            try:
                text = format_chat_to_text(ex, self.tokenizer)
                texts.append({"text": text})
            except Exception as e:
                print(f"Warning: skipping malformed example: {e}")

        return Dataset.from_list(texts)

    def train(self) -> None:
        """Run training."""
        if self.model is None:
            self.load_base_model()
            self.apply_lora()

        dataset = self.load_dataset()
        t_cfg = self.cfg["training"]

        output_dir = self.cfg["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=t_cfg["num_train_epochs"],
            per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
            learning_rate=t_cfg["learning_rate"],
            max_grad_norm=t_cfg["max_grad_norm"],
            warmup_ratio=t_cfg["warmup_ratio"],
            lr_scheduler_type=t_cfg["lr_scheduler_type"],
            logging_steps=t_cfg["logging_steps"],
            save_strategy=t_cfg["save_strategy"],
            bf16=t_cfg["bf16"],
            group_by_length=t_cfg["group_by_length"],
            optim=t_cfg["optim"],
            report_to="none",
        )

        loss_callback = LossHistoryCallback(on_step=self.progress_callback)
        callbacks = [loss_callback]
        patience = self.cfg.get("early_stopping_patience")
        if patience:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.cfg["max_seq_length"],
            args=training_args,
            callbacks=callbacks,
            packing=False,
        )

        print("Starting training...")
        self.trainer.train()

        # Capture loss history
        self.loss_history = loss_callback.history

        # Save
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"LoRA adapter saved to: {output_dir}")

    def save_config(self) -> None:
        """Save merged config alongside adapter."""
        path = Path(self.cfg["output_dir"]) / "training_config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LoRATrainer":
        """Load trainer from YAML config."""
        from src.utils.config_loader import load_yaml
        cfg = load_yaml(path)
        return cls(cfg)

    @classmethod
    def from_json(cls, path: str | Path) -> "LoRATrainer":
        """Load trainer from JSON config."""
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(cfg)