"""Apply batch MEMIT edits.

Usage:
    python scripts/editing/memit_edit.py --config config/training/memit.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_editor import EditRequest, MEMITEditor
from src.utils.config_loader import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training/memit.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )

    editor = MEMITEditor(model, tokenizer)

    requests = [
        EditRequest(
            subject=e["subject"],
            relation=e.get("relation"),
            target=e["target"],
            layer_idx=e.get("layer_idx", 15),
            lam=e.get("lam", 5.0),
        )
        for e in cfg.get("edits", [])
    ]

    calibration = cfg.get("calibration_prompts", [])

    try:
        print(f"Applying {len(requests)} MEMIT edits...")
        editor.apply_batch(requests, calibration)

        # Test each edit
        for req in requests:
            test_prompt = f"{req.subject}是什么？"
            inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.8,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n[{req.subject}] -> {response[:200]}")

        if cfg.get("save_edit_log", True):
            log_path = Path(cfg.get("edit_log_path", "data/edits/memit_log.json"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            editor.save_state(log_path)
            print(f"\nEdit log saved to: {log_path}")

    finally:
        if cfg.get("restore_on_exit", True):
            print("Restoring original weights...")
            editor.restore_all()


if __name__ == "__main__":
    main()
