#!/usr/bin/env python3
"""对齐调整交互程序 — 投入JSON训练数据，训练LoRA，反馈拟合程度。

用法:
    # 基本用法：指定训练数据
    python scripts/alignment_tuner.py --data data/my_romance_data.jsonl

    # 指定测试prompts（评估用）
    python scripts/alignment_tuner.py --data data/my_romance_data.jsonl --test-prompts data/test_prompts.txt

    # 自定义训练参数
    python scripts/alignment_tuner.py --data data/my_romance_data.jsonl --r 32 --lr 5e-5 --epochs 5

    # 交互模式（训练后进入对话测试）
    python scripts/alignment_tuner.py --data data/my_romance_data.jsonl --interactive

    # 仅评估已有adapter（不训练）
    python scripts/alignment_tuner.py --data data/my_romance_data.jsonl --adapter lora_romance --eval-only
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.lora_trainer import LoRATrainer

console = Console()


# ──────────────────────────────────────────────
#  数据加载
# ──────────────────────────────────────────────

def load_json_data(path: str | Path) -> list[dict]:
    """加载JSON/JSONL数据，自动检测格式。"""
    path = Path(path)
    raw = path.read_text(encoding="utf-8").strip()

    # JSON数组格式
    if raw.startswith("["):
        return json.loads(raw)

    # JSONL格式（每行一个对象）
    data = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data


def extract_pairs(data: list[dict]) -> list[dict[str, str]]:
    """从多种格式中提取 (prompt, target) 对。

    支持格式:
      - {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
      - {"text": "<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>"}
      - {"instruction": "...", "output": "..."}
      - {"prompt": "...", "completion": "..."}
    """
    pairs = []
    for ex in data:
        # messages格式
        if "messages" in ex:
            msgs = ex["messages"]
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if user and assistant:
                pairs.append({"prompt": user, "target": assistant})
                continue

        # text格式 (Qwen chat template)
        if "text" in ex:
            text = ex["text"]
            u_match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.DOTALL)
            a_match = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL)
            if u_match and a_match:
                pairs.append({"prompt": u_match.group(1).strip(), "target": a_match.group(1).strip()})
                continue

        # instruction/output格式
        if "instruction" in ex and "output" in ex:
            pairs.append({"prompt": ex["instruction"], "target": ex["output"]})
            continue

        # prompt/completion格式
        if "prompt" in ex and "completion" in ex:
            pairs.append({"prompt": ex["prompt"], "target": ex["completion"]})
            continue

    return pairs


def validate_data(data: list[dict]) -> tuple[int, int, list[str]]:
    """验证数据，返回 (总数, 有效数, 警告列表)。"""
    total = len(data)
    pairs = extract_pairs(data)
    valid = len(pairs)
    warnings = []

    if total == 0:
        warnings.append("数据为空")
    elif valid == 0:
        warnings.append("未找到有效的训练对。支持格式: messages / text / instruction-output / prompt-completion")
    elif valid < total:
        warnings.append(f"{total - valid} 条数据无法解析，已跳过")

    return total, valid, warnings


# ──────────────────────────────────────────────
#  模型加载
# ──────────────────────────────────────────────

def load_base_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> tuple[Any, Any]:
    """加载4-bit量化基座模型。"""
    console.print(f"[dim]加载基座模型: {model_name}[/dim]")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()
    return model, tokenizer


def generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 128) -> str:
    """生成回复。"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 截掉输入prompt部分
    if full.startswith(text):
        return full[len(text):].strip()
    return full.strip()


# ──────────────────────────────────────────────
#  拟合度评估
# ──────────────────────────────────────────────

class FitEvaluator:
    """评估模型在训练数据上的拟合程度。"""

    def __init__(self):
        console.print("[dim]加载嵌入模型 (BAAI/bge-small-zh-v1.5)...[/dim]")
        self.embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    def semantic_similarity(self, text_a: str, text_b: str) -> float:
        """计算两个文本的余弦相似度 [0, 1]."""
        if not text_a.strip() or not text_b.strip():
            return 0.0
        emb_a = self.embedder.encode(text_a)
        emb_b = self.embedder.encode(text_b)
        sim = (emb_a @ emb_b) / (max(1e-8, (emb_a @ emb_a) ** 0.5 * (emb_b @ emb_b) ** 0.5))
        return float(max(0, min(1, sim)))

    def token_overlap(self, text_a: str, text_b: str) -> float:
        """基于字符级2-gram的Jaccard相似度。"""
        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        a = bigrams(text_a)
        b = bigrams(text_b)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        pairs: list[dict[str, str]],
        label: str = "",
        max_samples: int = 30,
    ) -> dict[str, Any]:
        """对一组 (prompt, target) 对评估拟合度。

        返回: {
            "semantic_similarities": [...],
            "token_overlaps": [...],
            "generations": [...],
            "mean_semantic": float,
            "mean_overlap": float,
        }
        """
        import random
        samples = random.sample(pairs, min(max_samples, len(pairs)))

        semantic_sims = []
        token_ovs = []
        generations = []

        for i, pair in enumerate(samples):
            prompt = pair["prompt"]
            target = pair["target"]
            gen = generate(model, tokenizer, prompt, max_new_tokens=128)

            sim = self.semantic_similarity(gen, target)
            tok = self.token_overlap(gen, target)

            semantic_sims.append(sim)
            token_ovs.append(tok)
            generations.append({"prompt": prompt, "target": target, "generated": gen})

            if (i + 1) % 5 == 0:
                console.print(f"  [{label}] 已评估 {i+1}/{len(samples)} ...")

        return {
            "semantic_similarities": semantic_sims,
            "token_overlaps": token_ovs,
            "generations": generations,
            "mean_semantic": sum(semantic_sims) / len(semantic_sims) if semantic_sims else 0,
            "mean_overlap": sum(token_ovs) / len(token_ovs) if token_ovs else 0,
        }

    def compare(
        self,
        baseline: dict[str, Any],
        trained: dict[str, Any],
        loss_history: list[dict] | None = None,
    ) -> None:
        """输出训练前后拟合度对比报告。"""
        console.print("\n")
        table = Table(title="📊 拟合程度评估报告", border_style="cyan")
        table.add_column("指标", style="bold", width=24)
        table.add_column("训练前 (Baseline)", justify="center", width=24)
        table.add_column("训练后 (Fine-tuned)", justify="center", width=24)
        table.add_column("变化", justify="center", width=16)

        # 语义相似度
        bs = baseline["mean_semantic"]
        ts = trained["mean_semantic"]
        delta_s = ts - bs
        sign_s = "↑" if delta_s > 0 else "↓" if delta_s < 0 else "→"
        color_s = "green" if delta_s > 0.01 else "red" if delta_s < -0.01 else "yellow"
        table.add_row(
            "语义相似度 (cosine)",
            f"{bs:.3f}",
            f"{ts:.3f}",
            f"[{color_s}]{sign_s} {delta_s:+.3f}[/{color_s}]",
        )

        # Token overlap
        bo = baseline["mean_overlap"]
        to = trained["mean_overlap"]
        delta_o = to - bo
        sign_o = "↑" if delta_o > 0 else "↓" if delta_o < 0 else "→"
        color_o = "green" if delta_o > 0.01 else "red" if delta_o < -0.01 else "yellow"
        table.add_row(
            "Token重叠度 (2-gram)",
            f"{bo:.3f}",
            f"{to:.3f}",
            f"[{color_o}]{sign_o} {delta_o:+.3f}[/{color_o}]",
        )

        # Training loss
        if loss_history:
            final_loss = loss_history[-1]["loss"] if loss_history else None
            initial_loss = loss_history[0]["loss"] if loss_history else None
            if final_loss and initial_loss:
                delta_l = initial_loss - final_loss
                table.add_row(
                    "训练 Loss",
                    f"—",
                    f"{final_loss:.4f}",
                    f"[green]↓ {delta_l:.4f}[/green] (下降 {delta_l/initial_loss*100:.0f}%)",
                )

        console.print(table)

        # 样本对比展示
        console.print("\n[bold]📝 样本生成对比 — 取语义相似度变化最大的3条:[/bold]\n")
        diffs = []
        for i, (b, t) in enumerate(zip(baseline["generations"], trained["generations"])):
            sim_b = self.semantic_similarity(b["generated"], b["target"])
            sim_t = self.semantic_similarity(t["generated"], t["target"])
            diffs.append((sim_t - sim_b, i, b, t, sim_b, sim_t))
        diffs.sort(key=lambda x: x[0], reverse=True)

        for rank, (delta, idx, b, t, sim_b, sim_t) in enumerate(diffs[:3]):
            color = "green" if delta > 0 else "red"
            console.print(Panel(
                f"[bold]Prompt:[/bold] {b['prompt'][:80]}...\n\n"
                f"[bold]Target:[/bold] {b['target'][:120]}...\n\n"
                f"[dim]训练前 (sim={sim_b:.3f}):[/dim] {b['generated'][:120]}...\n"
                f"[{'green' if delta > 0 else 'red'}]训练后 (sim={sim_t:.3f}, {delta:+.3f}):[/{'green' if delta > 0 else 'red'}] {t['generated'][:120]}...",
                title=f"#{rank+1} 拟合变化 [{color}]{delta:+.3f}[/{color}]",
                border_style=color,
            ))


# ──────────────────────────────────────────────
#  交互测试模式
# ──────────────────────────────────────────────

def interactive_test(model: Any, tokenizer: Any) -> None:
    """交互式测试：用户输入prompt，模型实时回复。"""
    console.print(Panel.fit(
        "[bold]交互测试模式[/bold]\n"
        "输入 prompt 查看模型回复。\n"
        "输入 [bold]/quit[/bold] 退出。\n"
        "输入 [bold]/compare <prompt>[/bold] 对比训练前后的差异（需要baseline模型）。",
        border_style="magenta",
    ))

    while True:
        try:
            user_input = console.input("[bold cyan]🧪 Prompt: [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            break

        console.print("[dim]生成中...[/dim]")
        response = generate(model, tokenizer, user_input, max_new_tokens=256)
        console.print(Panel(response, title="🤖 模型回复", border_style="green"))


# ──────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="对齐调整交互程序 — 投入JSON训练数据，反馈拟合程度"
    )
    parser.add_argument("--data", required=True, help="训练数据路径 (JSON/JSONL)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="基座模型名称")
    parser.add_argument("--output", default="lora_alignment", help="LoRA输出目录")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--max-samples", type=int, default=30, help="评估时最多使用多少样本")
    parser.add_argument("--eval-only", action="store_true", help="仅评估已有adapter，不训练")
    parser.add_argument("--adapter", default=None, help="已有adapter路径（用于--eval-only）")
    parser.add_argument("--interactive", action="store_true", help="训练后进入交互测试模式")
    parser.add_argument("--target-modules", default="all-linear", help="LoRA目标模块")
    args = parser.parse_args()

    # ── 阶段0：数据加载与验证 ──
    console.print(Panel.fit("[bold]🔬 Heartscape 对齐调整程序[/bold]", border_style="cyan"))
    console.print(f"[dim]数据路径: {args.data}[/dim]")

    data = load_json_data(args.data)
    total, valid, warnings = validate_data(data)

    console.print(f"数据总量: {total} 条, 有效训练对: {valid} 条")
    for w in warnings:
        console.print(f"[yellow]⚠ {w}[/yellow]")
    if valid == 0:
        console.print("[red]无法提取有效训练对，退出。[/red]")
        sys.exit(1)

    pairs = extract_pairs(data)
    console.print(f"[dim]示例: {pairs[0]['prompt'][:60]}... → {pairs[0]['target'][:60]}...[/dim]")

    evaluator = FitEvaluator()

    # ── 阶段1：Baseline评估 ──
    console.print("\n[bold]📋 阶段1: 训练前基线评估[/bold]")
    base_model, base_tokenizer = load_base_model(args.model)

    console.print(f"对 {min(args.max_samples, len(pairs))} 条样本进行基线评估...")
    baseline_results = evaluator.evaluate(
        base_model, base_tokenizer, pairs, label="Baseline", max_samples=args.max_samples
    )
    console.print(f"  基线语义相似度: {baseline_results['mean_semantic']:.3f}")
    console.print(f"  基线Token重叠度: {baseline_results['mean_overlap']:.3f}")

    # 释放baseline模型
    del base_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ── 阶段2：LoRA训练 ──
    if not args.eval_only:
        console.print("\n[bold]📋 阶段2: LoRA 训练[/bold]")
        console.print(f"  配置: r={args.r}, alpha={args.alpha}, lr={args.lr}, epochs={args.epochs}")
        console.print(f"  目标模块: {args.target_modules}")
        console.print(f"  输出目录: {args.output}")

        trainer = LoRATrainer({
            "model_name": args.model,
            "output_dir": args.output,
            "dataset_path": args.data,
            "lora": {
                "r": args.r,
                "alpha": args.alpha,
                "dropout": 0.0,
                "target_modules": args.target_modules,
                "bias": "none",
            },
            "training": {
                "num_train_epochs": args.epochs,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "learning_rate": args.lr,
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
        })

        trainer.train()

        # 保存训练配置
        report_path = Path(args.output) / "alignment_report.json"
        report = {
            "timestamp": datetime.now().isoformat(),
            "data": {"path": args.data, "total": total, "valid": valid},
            "training_config": trainer.cfg,
            "loss_history": trainer.loss_history,
            "baseline": {
                "mean_semantic": baseline_results["mean_semantic"],
                "mean_overlap": baseline_results["mean_overlap"],
            },
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        console.print(f"[dim]训练报告已保存: {report_path}[/dim]")

        loss_history = trainer.loss_history
        adapter_path = args.output
    else:
        if not args.adapter:
            console.print("[red]--eval-only 需要指定 --adapter 路径[/red]")
            sys.exit(1)
        adapter_path = args.adapter
        loss_history = None
        console.print(f"\n[dim]跳过训练，使用已有adapter: {adapter_path}[/dim]")

    # ── 阶段3：训练后评估 ──
    console.print("\n[bold]📋 阶段3: 训练后拟合评估[/bold]")
    trained_model, trained_tokenizer = load_base_model(args.model)
    trained_model = PeftModel.from_pretrained(trained_model, adapter_path)
    trained_model.eval()

    trained_results = evaluator.evaluate(
        trained_model, trained_tokenizer, pairs, label="Fine-tuned", max_samples=args.max_samples
    )
    console.print(f"  训练后语义相似度: {trained_results['mean_semantic']:.3f}")
    console.print(f"  训练后Token重叠度: {trained_results['mean_overlap']:.3f}")

    # ── 阶段4：对比报告 ──
    evaluator.compare(baseline_results, trained_results, loss_history)

    # 更新训练报告
    if not args.eval_only:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        report["trained"] = {
            "mean_semantic": trained_results["mean_semantic"],
            "mean_overlap": trained_results["mean_overlap"],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 阶段5：交互测试 ──
    if args.interactive:
        console.print("\n")
        interactive_test(trained_model, trained_tokenizer)

    # 清理
    del trained_model
    gc.collect()
    torch.cuda.empty_cache()
    console.print("\n[green]✅ 完成。[/green]")


if __name__ == "__main__":
    main()
