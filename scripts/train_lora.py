"""Train LoRA adapter with YAML/JSON config.

Usage:
    python scripts/train_lora.py --config config/training/lora.yaml
    python scripts/train_lora.py --config config/training/lora.json

Or override specific values:
    python scripts/train_lora.py --config config/training/lora.yaml --lr 2e-5 --epochs 2
"""

import argparse
from pathlib import Path

from src.training.lora_trainer import LoRATrainer


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter")
    parser.add_argument("--config", default="config/training/lora.yaml", help="Path to config file (YAML or JSON)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs")
    parser.add_argument("--r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--output", default=None, help="Override output directory")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.suffix in (".yaml", ".yml"):
        trainer = LoRATrainer.from_yaml(config_path)
    elif config_path.suffix == ".json":
        trainer = LoRATrainer.from_json(config_path)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    # Apply CLI overrides
    if args.lr is not None:
        trainer.cfg["training"]["learning_rate"] = args.lr
    if args.epochs is not None:
        trainer.cfg["training"]["num_train_epochs"] = args.epochs
    if args.r is not None:
        trainer.cfg["lora"]["r"] = args.r
    if args.output is not None:
        trainer.cfg["output_dir"] = args.output

    print("=" * 60)
    print(f"LoRA Training Config:")
    print(f"  Model: {trainer.cfg['model_name']}")
    print(f"  Output: {trainer.cfg['output_dir']}")
    print(f"  LoRA r={trainer.cfg['lora']['r']}, alpha={trainer.cfg['lora']['alpha']}")
    print(f"  LR: {trainer.cfg['training']['learning_rate']}")
    print(f"  Epochs: {trainer.cfg['training']['num_train_epochs']}")
    print("=" * 60)

    trainer.train()
    trainer.save_config()


if __name__ == "__main__":
    main()
