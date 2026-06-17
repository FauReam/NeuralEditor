"""Apply single-fact ROME edit.

Usage:
    python scripts/editing/rome_edit.py --config config/training/rome.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_editor import EditRequest, ROMEEditor
from src.utils.config_loader import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training/rome.yaml")
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

    editor = ROMEEditor(model, tokenizer)

    edit_cfg = cfg["edit"]
    request = EditRequest(
        subject=edit_cfg["subject"],
        relation=edit_cfg.get("relation"),
        target=edit_cfg["target"],
        layer_idx=edit_cfg.get("layer_idx", 15),
        lam=edit_cfg.get("lam", 5.0),
    )

    calibration = cfg.get("calibration_prompts", [])

    try:
        print(f"Applying ROME edit: '{request.subject}' -> '{request.target}'")
        editor.apply(request, calibration)

        # Test
        test_prompt = f"{request.subject}是什么？"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.8,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nTest prompt: {test_prompt}")
        print(f"Response: {response}")

        # Save log
        if cfg.get("save_edit_log", True):
            log_path = Path(cfg.get("edit_log_path", "data/edits/rome_log.json"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            editor.save_state(log_path)
            print(f"Edit log saved to: {log_path}")

    finally:
        if cfg.get("restore_on_exit", True):
            print("Restoring original weights...")
            editor.restore_all()


if __name__ == "__main__":
    main()
