#!/usr/bin/env python3
"""Heartscape Web 操作界面 — 浏览器中一键训练、测试模型。

启动方式:
    python scripts/web_ui.py
    然后浏览器打开 http://localhost:8765

零额外依赖，基于 Python 内置 http.server。
"""

import json
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════
#  全局状态
# ═══════════════════════════════════════════════

class AppState:
    """线程安全的全局应用状态。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"          # idle | running | done | error
        self.progress = 0.0           # 0-100
        self.current_step = 0
        self.total_steps = 0
        self.loss_history: list[dict] = []
        self.message = ""
        self.error = ""
        self.report: dict = {}
        self.baseline_semantic: float | None = None
        self.trained_semantic: float | None = None
        self.baseline_overlap: float | None = None
        self.trained_overlap: float | None = None
        # 模型引用（训练后保持）
        self.model: Any = None
        self.tokenizer: Any = None

    def reset(self):
        with self.lock:
            self.status = "idle"
            self.progress = 0.0
            self.current_step = 0
            self.total_steps = 0
            self.loss_history = []
            self.message = "就绪，等待训练。"
            self.error = ""
            self.report = {}
            self.baseline_semantic = None
            self.trained_semantic = None
            self.baseline_overlap = None
            self.trained_overlap = None

    def set_running(self, total_steps: int = 0):
        with self.lock:
            self.status = "running"
            self.total_steps = total_steps
            self.current_step = 0
            self.message = "训练中..."

    def update_step(self, step: int, loss: float):
        with self.lock:
            self.current_step = step
            self.loss_history.append({"step": step, "loss": round(loss, 6)})
            if self.total_steps > 0:
                self.progress = min(99, step / self.total_steps * 100)
            self.message = f"Step {step}: loss={loss:.4f}"

    def set_done(self):
        with self.lock:
            self.status = "done"
            self.progress = 100.0
            self.message = "训练完成！"

    def set_error(self, err: str):
        with self.lock:
            self.status = "error"
            self.error = str(err)
            self.message = f"错误: {err}"

    def to_dict(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "progress": round(self.progress, 1),
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "loss_history": self.loss_history[-20:],  # 最近20步
                "message": self.message,
                "error": self.error,
                "baseline_semantic": self.baseline_semantic,
                "trained_semantic": self.trained_semantic,
                "baseline_overlap": self.baseline_overlap,
                "trained_overlap": self.trained_overlap,
            }


state = AppState()


# ═══════════════════════════════════════════════
#  训练执行器（后台线程）
# ═══════════════════════════════════════════════

def run_training(
    data_path: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "lora_web",
    r: int = 32,
    alpha: int = 64,
    lr: float = 5e-5,
    epochs: int = 5,
    target_modules: str = "all-linear",
) -> None:
    """在后台线程中执行完整训练+评估流程。"""
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer

    try:
        # ── 加载数据 ──
        data = []
        raw = Path(data_path).read_text(encoding="utf-8").strip()
        if raw.startswith("["):
            data = json.loads(raw)
        else:
            for line in raw.splitlines():
                if line.strip():
                    data.append(json.loads(line))

        pairs = []
        for ex in data:
            if "messages" in ex:
                msgs = ex["messages"]
                u = next((m["content"] for m in msgs if m["role"] == "user"), "")
                a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                if u and a:
                    pairs.append({"prompt": u, "target": a})
            elif "text" in ex:
                t = ex["text"]
                um = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", t, re.DOTALL)
                am = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", t, re.DOTALL)
                if um and am:
                    pairs.append({"prompt": um.group(1).strip(), "target": am.group(1).strip()})
            elif "instruction" in ex and "output" in ex:
                pairs.append({"prompt": ex["instruction"], "target": ex["output"]})

        if not pairs:
            state.set_error("无法从数据中提取训练对")
            return

        # ── Baseline评估 ──
        state.message = "正在进行训练前基线评估..."
        embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, torch_dtype="auto",
        )
        model.eval()

        def _gen(prompt: str) -> str:
            msgs = [{"role": "user", "content": prompt}]
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(txt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=True,
                                     temperature=0.8, top_p=0.9,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
            full = tokenizer.decode(out[0], skip_special_tokens=True)
            return full[len(txt):].strip() if full.startswith(txt) else full.strip()

        import random
        samples = random.sample(pairs, min(15, len(pairs)))
        b_semantic = []
        b_overlap = []
        for p in samples:
            gen = _gen(p["prompt"])
            ea, eb = embedder.encode(gen), embedder.encode(p["target"])
            sim = float((ea @ eb) / (max(1e-8, (ea @ ea)**0.5 * (eb @ eb)**0.5)))
            b_semantic.append(max(0, min(1, sim)))
            bg = lambda s: set(s[i:i+2] for i in range(len(s)-1))
            oa, ob = bg(gen), bg(p["target"])
            b_overlap.append(len(oa & ob) / max(1e-8, len(oa | ob)))

        state.baseline_semantic = sum(b_semantic) / len(b_semantic)
        state.baseline_overlap = sum(b_overlap) / len(b_overlap)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        # ── LoRA 训练 ──
        from src.training.lora_trainer import LoRATrainer

        def on_progress(step, loss):
            state.update_step(step, loss)

        trainer = LoRATrainer(config={
            "model_name": model_name,
            "output_dir": output_dir,
            "dataset_path": data_path,
            "lora": {"r": r, "alpha": alpha, "dropout": 0.0,
                     "target_modules": target_modules, "bias": "none"},
            "training": {
                "num_train_epochs": epochs,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "learning_rate": lr,
                "max_grad_norm": 0.3, "warmup_ratio": 0.03,
                "lr_scheduler_type": "cosine", "logging_steps": 10,
                "save_strategy": "epoch", "bf16": True,
                "group_by_length": False, "optim": "paged_adamw_8bit",
            },
            "max_seq_length": 512,
        }, progress_callback=on_progress)

        # Estimate total steps
        total_steps_est = epochs * max(1, len(data)) // 4  # rough guess
        state.set_running(total_steps_est)

        trainer.train()

        # ── 训练后评估 ──
        state.message = "正在进行训练后评估..."
        model2 = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, torch_dtype="auto",
        )
        model2 = PeftModel.from_pretrained(model2, output_dir)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        if tokenizer2.pad_token is None:
            tokenizer2.pad_token = tokenizer2.eos_token

        def _gen2(prompt: str) -> str:
            msgs = [{"role": "user", "content": prompt}]
            txt = tokenizer2.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer2(txt, return_tensors="pt").to(model2.device)
            with torch.no_grad():
                out = model2.generate(**inputs, max_new_tokens=128, do_sample=True,
                                     temperature=0.8, top_p=0.9,
                                     pad_token_id=tokenizer2.pad_token_id,
                                     eos_token_id=tokenizer2.eos_token_id)
            full = tokenizer2.decode(out[0], skip_special_tokens=True)
            return full[len(txt):].strip() if full.startswith(txt) else full.strip()

        t_semantic = []
        t_overlap = []
        for p in samples:
            gen = _gen2(p["prompt"])
            ea, eb = embedder.encode(gen), embedder.encode(p["target"])
            sim = float((ea @ eb) / (max(1e-8, (ea @ ea)**0.5 * (eb @ eb)**0.5)))
            t_semantic.append(max(0, min(1, sim)))
            oa, ob = bg(gen), bg(p["target"])
            t_overlap.append(len(oa & ob) / max(1e-8, len(oa | ob)))

        state.trained_semantic = sum(t_semantic) / len(t_semantic)
        state.trained_overlap = sum(t_overlap) / len(t_overlap)

        # 保存模型和tokenizer用于后续交互
        state.model = model2
        state.tokenizer = tokenizer2

        state.set_done()

        # 保存报告
        report_path = Path(output_dir) / "web_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "data_path": data_path,
                "config": {"r": r, "alpha": alpha, "lr": lr, "epochs": epochs},
                "loss_history": state.loss_history,
                "baseline": {"semantic": state.baseline_semantic, "overlap": state.baseline_overlap},
                "trained": {"semantic": state.trained_semantic, "overlap": state.trained_overlap},
            }, f, ensure_ascii=False, indent=2)

    except Exception as e:
        traceback.print_exc()
        state.set_error(str(e))


# ═══════════════════════════════════════════════
#  HTTP 处理器
# ═══════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>💕 Heartscape 训练控制台</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:"PingFang SC","Microsoft YaHei",sans-serif;background:linear-gradient(135deg,#fdf6f0,#fce4ec);color:#3a2a1a;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:20px}
  h1{font-size:1.8em;color:#d4456a;margin:10px 0 5px;text-align:center}
  .sub{color:#999;font-size:.85em;margin-bottom:25px}
  .container{max-width:800px;width:100%}
  .card{background:#fff;border-radius:14px;padding:22px 26px;margin:14px 0;box-shadow:0 2px 10px rgba(180,120,130,.1)}
  .card h2{font-size:1.15em;color:#b8405e;margin-bottom:14px;display:flex;align-items:center;gap:8px}
  .row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:12px}
  .field{flex:1;min-width:140px}
  .field label{display:block;font-size:.85em;color:#888;margin-bottom:4px}
  .field input,.field select{width:100%;padding:9px 12px;border:1.5px solid #e8d0d8;border-radius:8px;font-size:.95em;outline:none;transition:border .2s}
  .field input:focus,.field select:focus{border-color:#d4456a}
  button{padding:10px 28px;border:none;border-radius:25px;font-size:1em;font-weight:bold;cursor:pointer;transition:all .2s}
  .btn-primary{background:#d4456a;color:#fff}
  .btn-primary:hover{background:#b8405e}
  .btn-primary:disabled{background:#e8c0cc;cursor:not-allowed}
  .btn-danger{background:#eee;color:#c00}
  .btn-danger:hover{background:#fdd}
  .progress-bar{height:28px;background:#f5e0e8;border-radius:14px;overflow:hidden;margin:8px 0}
  .progress-fill{height:100%;background:linear-gradient(90deg,#f0a0b0,#d4456a);border-radius:14px;transition:width .4s;display:flex;align-items:center;justify-content:center;color:#fff;font-size:.8em;font-weight:bold;min-width:40px}
  .status{display:inline-block;padding:4px 14px;border-radius:12px;font-size:.85em;font-weight:bold}
  .status.idle{background:#e8e8e8;color:#888}
  .status.running{background:#fff3e0;color:#e65100}
  .status.done{background:#e8f5e9;color:#2e7d32}
  .status.error{background:#ffebee;color:#c62828}
  .log-box{background:#2d1f24;color:#f5d0d8;padding:12px 16px;border-radius:10px;font-family:monospace;font-size:.8em;max-height:180px;overflow-y:auto;line-height:1.6}
  .chat-box{border:1.5px solid #e8d0d8;border-radius:10px;padding:14px;min-height:100px;max-height:300px;overflow-y:auto;margin-bottom:10px;background:#fdf6f0}
  .chat-msg{margin:8px 0;padding:8px 12px;border-radius:10px;max-width:85%}
  .chat-user{background:#f0a0b0;color:#fff;margin-left:auto;text-align:right;border-radius:10px 10px 0 10px}
  .chat-assistant{background:#fff;border:1px solid #f0d0d8;border-radius:10px 10px 10px 0}
  .chat-input-row{display:flex;gap:8px}
  .chat-input-row input{flex:1;padding:10px 14px;border:1.5px solid #e8d0d8;border-radius:20px;font-size:.95em;outline:none}
  .chat-input-row input:focus{border-color:#d4456a}
  .metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
  .metric{background:#fdf6f0;border-radius:10px;padding:14px;text-align:center}
  .metric .val{font-size:1.6em;font-weight:bold;color:#d4456a}
  .metric .lbl{font-size:.78em;color:#999;margin-top:2px}
  .metric .delta{font-size:.85em;margin-top:4px}
  .delta.up{color:#2e7d32}
  .delta.down{color:#c62828}
  .hidden{display:none}
  @media(max-width:600px){body{padding:10px}.row{flex-direction:column;gap:8px}.field{min-width:100%}}
</style>
</head>
<body>
<div class="container">
  <h1>💕 Heartscape 训练控制台</h1>
  <p class="sub">浏览器中一键微调恋爱模型 · 奶奶友好操作</p>

  <!-- 训练配置 -->
  <div class="card">
    <h2>⚙️ 训练配置</h2>
    <div class="row">
      <div class="field">
        <label>📁 训练数据文件</label>
        <select id="data-file">
          <option value="data/my_romance_data.jsonl">data/my_romance_data.jsonl（推荐）</option>
          <option value="data/romance_chat_sample.jsonl">data/romance_chat_sample.jsonl</option>
        </select>
      </div>
      <div class="field">
        <label>🎯 目标模块</label>
        <select id="target-modules">
          <option value="all-linear">all-linear（全部层）</option>
          <option value="q_proj,v_proj">q_proj+v_proj（仅注意力）</option>
        </select>
      </div>
    </div>
    <div class="row">
      <div class="field">
        <label>📐 LoRA Rank (r)</label>
        <input id="r" type="number" value="32" min="4" max="64">
      </div>
      <div class="field">
        <label>📏 Alpha</label>
        <input id="alpha" type="number" value="64" min="8" max="128">
      </div>
      <div class="field">
        <label>📚 Epochs</label>
        <input id="epochs" type="number" value="5" min="1" max="10">
      </div>
      <div class="field">
        <label>⚡ 学习率 (LR)</label>
        <input id="lr" type="text" value="5e-5">
      </div>
    </div>
    <button class="btn-primary" id="btn-train" onclick="startTraining()">🚀 开始训练</button>
    <button class="btn-danger hidden" id="btn-cancel" onclick="cancelTraining()">⏹ 取消</button>
  </div>

  <!-- 训练进度 -->
  <div class="card">
    <h2>📊 训练进度 <span class="status idle" id="status-badge">就绪</span></h2>
    <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%">0%</div></div>
    <p id="status-msg" style="font-size:.9em;color:#777;margin:8px 0">点击"开始训练"按钮启动。</p>
    <div class="log-box" id="log-box">等待训练开始...</div>
  </div>

  <!-- 拟合报告 -->
  <div class="card hidden" id="report-card">
    <h2>📋 拟合程度报告</h2>
    <div class="metric-grid">
      <div class="metric">
        <div class="lbl">语义相似度（训练前 → 训练后）</div>
        <div class="val" id="metric-semantic">—</div>
        <div class="delta" id="delta-semantic"></div>
      </div>
      <div class="metric">
        <div class="lbl">Token重叠度（训练前 → 训练后）</div>
        <div class="val" id="metric-overlap">—</div>
        <div class="delta" id="delta-overlap"></div>
      </div>
    </div>
  </div>

  <!-- 交互测试 -->
  <div class="card hidden" id="chat-card">
    <h2>💬 和小棠对话测试</h2>
    <div class="chat-box" id="chat-box">
      <div class="chat-msg chat-assistant">*微微一笑* 你好呀...我是小棠。有什么想和我聊的吗？</div>
    </div>
    <div class="chat-input-row">
      <input id="chat-input" placeholder="输入你想说的话..." onkeydown="if(event.key==='Enter')sendChat()">
      <button class="btn-primary" onclick="sendChat()" style="padding:10px 20px">发送</button>
    </div>
  </div>
</div>

<script>
let pollInterval = null;

function $(id){return document.getElementById(id)}

function startTraining(){
  $('btn-train').disabled = true;
  $('btn-train').textContent = '⏳ 训练中...';
  $('btn-cancel').classList.remove('hidden');
  $('report-card').classList.add('hidden');
  $('chat-card').classList.add('hidden');
  $('log-box').innerHTML = '';

  const body = JSON.stringify({
    data_file: $('data-file').value,
    target_modules: $('target-modules').value,
    r: parseInt($('r').value),
    alpha: parseInt($('alpha').value),
    epochs: parseInt($('epochs').value),
    lr: parseFloat($('lr').value),
  });

  fetch('/api/train', {method:'POST', headers:{'Content-Type':'application/json'}, body})
    .then(r => r.json())
    .then(d => { if(d.error) alert('启动失败: '+d.error); })
    .catch(e => alert('网络错误: '+e));

  pollInterval = setInterval(pollStatus, 1500);
}

function cancelTraining(){
  fetch('/api/cancel', {method:'POST'}).then(() => {
    clearInterval(pollInterval);
    resetUI();
  });
}

function pollStatus(){
  fetch('/api/status').then(r => r.json()).then(d => {
    $('progress-fill').style.width = d.progress + '%';
    $('progress-fill').textContent = d.progress + '%';
    $('status-msg').textContent = d.message;

    // Status badge
    const badge = $('status-badge');
    badge.className = 'status ' + d.status;
    const labels = {idle:'就绪',running:'训练中',done:'✅ 完成',error:'❌ 错误'};
    badge.textContent = labels[d.status] || d.status;

    // Loss log
    if(d.loss_history && d.loss_history.length){
      $('log-box').innerHTML = d.loss_history.map(
        h => `Step ${h.step}: loss=<b>${h.loss.toFixed(4)}</b>`
      ).join('<br>');
    }

    // Report
    if(d.status === 'done'){
      $('report-card').classList.remove('hidden');
      $('chat-card').classList.remove('hidden');
      if(d.baseline_semantic != null && d.trained_semantic != null){
        const ds = d.trained_semantic - d.baseline_semantic;
        $('metric-semantic').textContent = d.baseline_semantic.toFixed(3) + ' → ' + d.trained_semantic.toFixed(3);
        $('delta-semantic').innerHTML = '<span class="delta '+(ds>=0?'up':'down')+'">'+(ds>=0?'↑':'↓')+' '+ds.toFixed(3)+'</span>';
        const od = d.trained_overlap - d.baseline_overlap;
        $('metric-overlap').textContent = d.baseline_overlap.toFixed(3) + ' → ' + d.trained_overlap.toFixed(3);
        $('delta-overlap').innerHTML = '<span class="delta '+(od>=0?'up':'down')+'">'+(od>=0?'↑':'↓')+' '+od.toFixed(3)+'</span>';
      }
      clearInterval(pollInterval);
      resetTrainButton();
    }

    if(d.status === 'error'){
      $('log-box').innerHTML += '<br><span style="color:#f88">❌ '+d.error+'</span>';
      clearInterval(pollInterval);
      resetTrainButton();
    }
  }).catch(()=>{});
}

function resetTrainButton(){
  $('btn-train').disabled = false;
  $('btn-train').textContent = '🚀 重新训练';
  $('btn-cancel').classList.add('hidden');
}

function resetUI(){
  $('btn-train').disabled = false;
  $('btn-train').textContent = '🚀 开始训练';
  $('btn-cancel').classList.add('hidden');
  $('progress-fill').style.width = '0%';
  $('progress-fill').textContent = '0%';
}

function sendChat(){
  const input = $('chat-input');
  const msg = input.value.trim();
  if(!msg) return;

  const box = $('chat-box');
  box.innerHTML += '<div class="chat-msg chat-user">'+escapeHtml(msg)+'</div>';
  input.value = '';
  box.scrollTop = box.scrollHeight;

  box.innerHTML += '<div class="chat-msg chat-assistant" id="chat-loading">*思考中...*</div>';

  fetch('/api/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({message: msg})
  })
    .then(r => r.json())
    .then(d => {
      const loading = $('chat-loading');
      if(loading) loading.remove();
      box.innerHTML += '<div class="chat-msg chat-assistant">'+escapeHtml(d.response||d.error||'...')+'</div>';
      box.scrollTop = box.scrollHeight;
    })
    .catch(e => {
      const loading = $('chat-loading');
      if(loading) loading.remove();
      box.innerHTML += '<div class="chat-msg chat-assistant" style="color:#c00">网络错误</div>';
    });
}

function escapeHtml(s){
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// 页面加载时开始轮询（显示当前状态）
pollInterval = setInterval(pollStatus, 3000);
pollStatus();
</script>
</body>
</html>"""


class RequestHandler(BaseHTTPRequestHandler):
    """处理 HTTP 请求。"""

    def log_message(self, format, *args):
        # 静默日志
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send_html(HTML_PAGE)
        elif path == "/api/status":
            self._send_json(state.to_dict())
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/train":
            if state.status == "running":
                self._send_json({"error": "训练已在运行中"}, 409)
                return

            body = self._read_body()
            data_file = body.get("data_file", "data/my_romance_data.jsonl")
            r = int(body.get("r", 32))
            alpha = int(body.get("alpha", 64))
            epochs = int(body.get("epochs", 5))
            lr = float(body.get("lr", 5e-5))
            target_modules = body.get("target_modules", "all-linear")

            state.reset()
            thread = threading.Thread(
                target=run_training,
                kwargs={
                    "data_path": data_file,
                    "r": r, "alpha": alpha, "lr": lr,
                    "epochs": epochs, "target_modules": target_modules,
                },
                daemon=True,
            )
            thread.start()
            self._send_json({"status": "started"})

        elif path == "/api/cancel":
            state.set_error("用户取消")
            self._send_json({"status": "cancelled"})

        elif path == "/api/chat":
            if state.status != "done" or state.model is None:
                self._send_json({"error": "模型尚未训练完成"}, 400)
                return

            import torch
            body = self._read_body()
            msg = body.get("message", "").strip()
            if not msg:
                self._send_json({"error": "消息为空"}, 400)
                return

            try:
                msgs = [{"role": "user", "content": msg}]
                txt = state.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = state.tokenizer(txt, return_tensors="pt").to(state.model.device)
                with torch.no_grad():
                    out = state.model.generate(
                        **inputs, max_new_tokens=256, do_sample=True,
                        temperature=0.85, top_p=0.9,
                        pad_token_id=state.tokenizer.pad_token_id,
                        eos_token_id=state.tokenizer.eos_token_id,
                    )
                full = state.tokenizer.decode(out[0], skip_special_tokens=True)
                resp = full[len(txt):].strip() if full.startswith(txt) else full.strip()
                self._send_json({"response": resp})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    port = 8765
    server = HTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"\n  💕 Heartscape Web UI 已启动")
    print(f"  🌐 浏览器打开: http://localhost:{port}")
    print(f"  ⏹  按 Ctrl+C 停止服务器\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务器已停止。")
        server.server_close()


if __name__ == "__main__":
    main()