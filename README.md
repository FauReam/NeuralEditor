# Heartscape Engine

本地LLM驱动的恋爱模拟/角色扮演引擎。基于 Qwen2.5-7B-Instruct，针对 RTX 4070 优化，支持**LoRA 风格微调**与**ROME/MEMIT 模型编辑**三种不同层级的权重干预。

## 特性

- **纯本地推理**：基于 transformers + bitsandbytes，无需联网
- **4070 友好**：4-bit量化，VRAM占用 < 8GB，上下文 4K-8K 甜点区
- **角色系统**：JSON定义人格、好感度、剧情状态
- **记忆系统**：短期对话窗口 + 长期向量记忆（RAG）
- **剧情引擎**：YAML场景定义，状态机驱动分支
- **三级权重干预**：
  1. **提示工程**（零权重改动）
  2. **LoRA 微调**（轻量 adapter，可开关）
  3. **ROME / MEMIT**（单条/批量神经元编辑，副作用最小）

## 项目结构

```
heartscape-engine/
├── config/
│   ├── characters/         # 角色定义（YAML）
│   ├── prompts/            # 系统提示词模板
│   ├── scenes/             # 剧情分支定义
│   └── training/           # LoRA / ROME / MEMIT 配置文件
├── data/
│   ├── saves/              # JSON存档
│   ├── memories/           # Chroma向量库
│   ├── edits/              # 编辑日志
│   └── romance_chat_sample.jsonl  # LoRA训练样本
├── scripts/
│   ├── download_model.py   # 下载 Qwen GGUF
│   ├── train_lora.py       # LoRA训练（配置驱动）
│   ├── demo_no_llm.py      # 无模型剧情测试
│   └── editing/
│       ├── rome_edit.py    # 单条事实 ROME 编辑
│       ├── memit_edit.py   # 批量 MEMIT 编辑
│       └── evaluate_edit.py # 编辑前后对比评估
├── src/
│   ├── models/             # LLM推理、模型编辑
│   ├── core/               # 角色、剧情、记忆、状态机
│   ├── training/           # LoRATrainer
│   └── utils/              # 配置加载、JSON存储、场景解析
└── tests/                  # 单元测试
```

## 硬件要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | RTX 3060 12GB | RTX 4070 |
| RAM | 16GB | 32GB |
| 磁盘 | 10GB | 50GB（含模型与训练缓存） |

> **上下文上限**：4070 12GB 在 4-bit 量化下，稳定运行上限约 **8K-16K**。32K 为理论架构上限，本地运行会 OOM。

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev,train]"

# 下载模型（约4GB）
python scripts/download_model.py

# 启动交互
python -m src.main

# 无模型演示模式（测试剧情逻辑）
python -m src.main --demo
python scripts/demo_no_llm.py
```

### Makefile 快捷命令

```bash
make install        # pip install -e ".[dev,train]"
make run            # python -m src.main
make test           # pytest tests/ -v
make download-model # 下载 Qwen GGUF
make train-lora     # 使用默认配置训练 LoRA
make lint           # ruff + mypy
make format         # black + ruff --fix
```

## 三级权重干预指南

### Level 1：提示工程（不动权重）

修改 `config/characters/default.yaml` 中的 `system_prompt`，或调整 `config/settings.yaml` 中的 `temperature` / `top_p`。

```yaml
# config/characters/default.yaml
system_prompt: |
  你是{char_name}，性格{personality}。
  说话风格：{speaking_style}
  当前关系：{relationship}
  好感度：{affection}/100
```

### Level 2：LoRA 微调（轻量 Adapter）

适合整体风格调整（如浪漫对话语气、角色一致性）。训练后生成 `adapter_model.safetensors`，可随时开关。

#### 配置文件

```yaml
# config/training/lora.yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"
dataset_path: "data/romance_chat_sample.jsonl"

lora:
  r: 8                  # LoRA rank（保守 4-8，激进 16-32）
  alpha: 16             # 缩放因子（通常 2*r）
  dropout: 0.0          # 正则化（0 = 无）
  target_modules: ["q_proj", "v_proj"]  # 仅注意力矩阵

training:
  num_train_epochs: 1
  learning_rate: 1.0e-5 # 保守值，避免破坏安全对齐
  gradient_accumulation_steps: 4
  optim: "paged_adamw_8bit"
```

#### 训练命令

```bash
# 使用配置文件
python scripts/train_lora.py --config config/training/lora.yaml

# 命令行覆盖参数
python scripts/train_lora.py --config config/training/lora.yaml \
    --r 16 --lr 2e-5 --epochs 2 --output lora_v2

# 4070 上约 15-30 分钟
```

#### 加载 LoRA

```yaml
# config/settings.yaml
llm:
  lora_path: "lora_romance"   # 训练输出目录
```

或在运行时动态加载/卸载：

```python
from src.models.llm_engine import LLMEngine

llm = LLMEngine(model_path="...", lora_path="lora_romance")
llm.unload_lora()  # 恢复基础模型
```

### Level 3：ROME / MEMIT（外科神经元编辑）

适合**精确修正特定概念**（如"约会"被过度拒绝为敏感内容），不改整体行为分布。

核心原理：定位存储该概念的 MLP 层神经元，计算协方差矩阵后执行**秩一更新**。

#### ROME（单条事实编辑）

```yaml
# config/training/rome.yaml
edit:
  subject: "牵手"                    # 要编辑的概念
  relation: "在恋爱中是一种"          # 上下文
  target: "常见的亲密表达方式"        # 新关联
  layer_idx: 15                     # 编辑层（10-25 范围探索）
  lam: 5.0                          # 正则化（↑保守 ↓激进）

calibration_prompts:                # 校准分布用的通用句子
  - "今天天气不错，适合出去散步。"
  - "我喜欢在图书馆看书。"

restore_on_exit: true               # 脚本结束后自动恢复权重
```

```bash
python scripts/editing/rome_edit.py --config config/training/rome.yaml
```

#### MEMIT（批量事实编辑）

同时编辑多条相关概念，比顺序 ROME 更稳定。

```yaml
# config/training/memit.yaml
edits:
  - subject: "牵手"
    target: "常见的亲密表达方式"
    layer_idx: 15
    lam: 5.0
  - subject: "拥抱"
    target: "温暖的安慰方式"
    layer_idx: 15
    lam: 5.0
  - subject: "约会"
    target: "了解彼此的正常社交活动"
    layer_idx: 15
    lam: 5.0
```

```bash
python scripts/editing/memit_edit.py --config config/training/memit.yaml
```

#### 评估编辑效果

```bash
python scripts/editing/evaluate_edit.py \
    --config config/training/rome.yaml \
    --test-prompts data/test_prompts.txt \
    --output data/edits/eval_results.json
```

输出包含**编辑前后对比**，自动统计有多少 prompt 的响应发生了变化。

---

## 调参工作流

**第一步：诊断问题范围**

```bash
# 用 evaluate 脚本测试模型对目标概念的响应
python scripts/editing/evaluate_edit.py \
    --config config/training/rome.yaml
```

**第二步：选择干预层级**

| 问题特征 | 推荐方案 | 可调参数 |
|---------|---------|---------|
| 整体风格不自然（语气、用词） | **LoRA** | `r`, `lr`, `epochs`, `target_modules` |
| 少数特定词被过度拒绝 | **ROME** | `layer_idx`, `lam` |
| 多个相关概念都被误杀 | **MEMIT** | `layer_idx`, `lam`, 批量编辑列表 |
| 轻微 OOC（角色脱离） | **提示工程** | `system_prompt`, `temperature` |

**第三步：调参方向**

```yaml
# LoRA 参数影响
r: 4        # 改动范围极小，适合微调语气
r: 8        # 平衡（推荐起点）
r: 16-32   # 改动范围大，需更多数据防止过拟合

lr: 5e-6    # 极保守，几乎不改变原有能力
lr: 1e-5    # 标准值
lr: 2e-5    # 激进，训练更快但可能破坏对齐

# ROME 参数影响
layer_idx: 10   # 偏语义/词汇层，改动较浅
layer_idx: 15   # 中层，通常效果最佳（推荐）
layer_idx: 20   # 偏高层/抽象，改动深远但副作用大

lam: 10.0  # 几乎无效果，非常安全
lam: 5.0   # 保守，副作用小（推荐起点）
lam: 2.0   # 效果强，可能扩散到相关概念
lam: 0.5   # 极强，风险高
```

---

## 内容边界

本引擎用于成人向恋爱模拟/视觉小说。禁止用于：
- 生成暴力、危害他人人生安全或其他有害内容

## License

MIT
