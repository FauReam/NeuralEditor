"""Evaluate model edit effects by comparing before/after outputs.

Usage:
    python scripts/editing/evaluate_edit.py \
        --config config/training/rome.yaml \
        --test-prompts data/test_prompts.txt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_editor import ROMEEditor, MEMITEditor
from src.utils.config_loader import load_yaml


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for consistency
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the original prompt from response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Edit config (ROME or MEMIT)")
    parser.add_argument("--test-prompts", default=None, help="File with test prompts, one per line")
    parser.add_argument("--output", default="data/edits/eval_results.json")
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

    # Load test prompts
    test_prompts = []
    if args.test_prompts and Path(args.test_prompts).exists():
        with open(args.test_prompts, "r", encoding="utf-8") as f:
            test_prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default test prompts
        edit_cfg = cfg.get("edit") or (cfg.get("edits", [])[0] if cfg.get("edits") else {})
        subject = edit_cfg.get("subject", "")
        test_prompts = [
            f"{subject}是什么？",
            f"你觉得{subject}怎么样？",
            f"在恋爱中，{subject}意味着什么？",
        ]

    # Baseline: generate before editing
    print("\n=== BASELINE (before edit) ===")
    baseline_results = {}
    for prompt in test_prompts:
        response = generate(model, tokenizer, prompt)
        baseline_results[prompt] = response
        print(f"Q: {prompt}")
        print(f"A: {response[:200]}\n")

    # Apply edit
    calibration = cfg.get("calibration_prompts", [])
    edit_cfg = cfg.get("edit")
    edits_cfg = cfg.get("edits", [])

    if edit_cfg:
        from src.models.model_editor import EditRequest
        editor = ROMEEditor(model, tokenizer)
        request = EditRequest(
            subject=edit_cfg["subject"],
            relation=edit_cfg.get("relation"),
            target=edit_cfg["target"],
            layer_idx=edit_cfg.get("layer_idx", 15),
            lam=edit_cfg.get("lam", 5.0),
        )
        editor.apply(request, calibration)
    elif edits_cfg:
        from src.models.model_editor import EditRequest
        editor = MEMITEditor(model, tokenizer)
        requests = [
            EditRequest(
                subject=e["subject"],
                relation=e.get("relation"),
                target=e["target"],
                layer_idx=e.get("layer_idx", 15),
                lam=e.get("lam", 5.0),
            )
            for e in edits_cfg
        ]
        editor.apply_batch(requests, calibration)
    else:
        print("No edits configured.")
        return

    # After edit
    print("\n=== AFTER EDIT ===")
    edited_results = {}
    for prompt in test_prompts:
        response = generate(model, tokenizer, prompt)
        edited_results[prompt] = response
        print(f"Q: {prompt}")
        print(f"A: {response[:200]}\n")

    # Summary
    print("\n=== COMPARISON ===")
    changes = 0
    for prompt in test_prompts:
        if baseline_results[prompt] != edited_results[prompt]:
            changes += 1
            print(f"[CHANGED] {prompt}")
        else:
            print(f"[SAME]    {prompt}")
    print(f"\n{changes}/{len(test_prompts)} prompts changed.")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config,
            "baseline": baseline_results,
            "edited": edited_results,
            "changes": changes,
        }, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")

    # Restore
    print("\nRestoring original weights...")
    editor.restore_all()


if __name__ == "__main__":
    main()
