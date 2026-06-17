#!/usr/bin/env python3
"""NeuralEditor API Server — 为 tuner.html 和 romance.html 提供后端服务。

启动: python -m src.web.server
端口: 8765
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ═══════════════════════════════════════════════════════
#  Global state
# ═══════════════════════════════════════════════════════

class TunerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.status = "idle"
            self.progress = 0.0
            self.current_step = 0
            self.total_steps = 0
            self.loss_history: list[dict] = []
            self.message = "就绪"
            self.error = ""
            self.model: Any = None
            self.tokenizer: Any = None
            self.baseline_semantic: float | None = None
            self.trained_semantic: float | None = None
            self.baseline_overlap: float | None = None
            self.trained_overlap: float | None = None
            self.edit_results: list[dict] = []
            self.loaded_data: list[dict] = []
            self.loaded_data_path: str = ""

    def set_running(self, total: int = 0):
        with self.lock:
            self.status = "running"
            self.total_steps = total
            self.current_step = 0
            self.progress = 0.0
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
            self.message = "完成"

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
                "loss_history": self.loss_history[-30:],
                "message": self.message,
                "error": self.error,
                "baseline_semantic": self.baseline_semantic,
                "trained_semantic": self.trained_semantic,
                "baseline_overlap": self.baseline_overlap,
                "trained_overlap": self.trained_overlap,
                "edit_results": self.edit_results[-10:],
                "loaded_data_path": self.loaded_data_path,
                "data_count": len(self.loaded_data),
            }


class RomanceState:
    def __init__(self):
        self.lock = threading.Lock()
        self.engine = None
        self.llm = None
        self.character_path = ""
        self.model_path = ""

    def init_engine(self, character_path: str = "", model_path: str = ""):
        from src.core.character import Character
        from src.core.memory_system import MemorySystem
        from src.core.state_machine import StoryStateMachine
        from src.core.story_engine import StoryEngine
        from src.utils.config_loader import MemoryConfig, StoryConfig
        from src.utils.json_storage import JSONStorage
        from src.utils.scene_loader import load_scenes

        cp = character_path or "config/characters/default.yaml"
        self.character_path = cp
        self.model_path = model_path

        character = Character.from_yaml(cp)
        memory = MemorySystem(embedding_model="BAAI/bge-small-zh-v1.5",
                              vector_db_path="data/memories/chroma")
        sm = StoryStateMachine()
        storage = JSONStorage("data/saves")

        story_cfg = StoryConfig()
        self.engine = StoryEngine(
            character=character,
            memory_system=memory,
            state_machine=sm,
            story_config=story_cfg,
            storage=storage,
        )
        scenes = load_scenes("config/scenes/chapter1.yaml")
        self.engine.init_scenes(scenes)

        if model_path:
            try:
                from src.models.llm_engine import LLMEngine
                self.llm = LLMEngine(model_path=model_path)
            except Exception:
                self.llm = None
        return self.engine

    def to_dict(self) -> dict:
        if self.engine is None:
            return {"ready": False}
        char = self.engine.character
        sm = self.engine.state_machine
        return {
            "ready": True,
            "character": {
                "name": char.profile.name,
                "affection": char.profile.affection_score,
                "relationship": char.get_relationship_label(),
                "personality": char.profile.personality_traits,
                "background": char.profile.background,
                "speaking_style": char.profile.speaking_style,
            },
            "current_scene": sm.current_scene.scene_id if sm.current_scene else "",
            "scene_desc": sm.current_scene.description if sm.current_scene else "",
            "turn_count": self.engine.turn_count,
            "available_choices": [
                {"id": c.choice_id, "text": c.text, "affection": c.affection_delta}
                for c in sm.available_choices(self.engine.character.story_flags.unlocked_scenes)
            ] if sm.current_scene else [],
            "has_llm": self.llm is not None,
        }


tuner_state = TunerState()
romance_state = RomanceState()

# HTML files loaded lazily
_HTML_DIR = Path(__file__).parent


# ═══════════════════════════════════════════════════════
#  Training executor (background thread)
# ═══════════════════════════════════════════════════════

def _extract_pairs(data: list[dict]) -> list[dict]:
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
    return pairs


def _run_baseline_eval(pairs, model_name):
    import gc
    import random
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer

    tuner_state.message = "正在进行训练前基线评估..."

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

    def _gen(prompt):
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

    samples = random.sample(pairs, min(15, len(pairs)))
    b_sem, b_ov = [], []
    for p in samples:
        gen = _gen(p["prompt"])
        ea, eb = embedder.encode(gen), embedder.encode(p["target"])
        sim = float((ea @ eb) / (max(1e-8, (ea @ ea) ** 0.5 * (eb @ eb) ** 0.5)))
        b_sem.append(max(0, min(1, sim)))
        bg = lambda s: set(s[i:i + 2] for i in range(len(s) - 1))
        oa, ob = bg(gen), bg(p["target"])
        b_ov.append(len(oa & ob) / max(1e-8, len(oa | ob)))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sum(b_sem) / len(b_sem), sum(b_ov) / len(b_ov)


def run_training(data_path: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 output_dir: str = "lora_web", r: int = 32, alpha: int = 64,
                 lr: float = 5e-5, epochs: int = 5,
                 target_modules: str = "all-linear") -> None:
    import gc
    import random
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer

    try:
        # Load data
        raw = Path(data_path).read_text(encoding="utf-8").strip()
        data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
        pairs = _extract_pairs(data)
        if not pairs:
            tuner_state.set_error("无法从数据中提取训练对")
            return

        tuner_state.loaded_data = data
        tuner_state.loaded_data_path = data_path

        tuner_state.baseline_semantic, tuner_state.baseline_overlap = _run_baseline_eval(pairs, model_name)

        # LoRA training
        from src.training.lora_trainer import LoRATrainer

        def on_progress(step, loss):
            tuner_state.update_step(step, loss)

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

        total_est = epochs * max(1, len(data)) // 4
        tuner_state.set_running(total_est)
        trainer.train()

        # Post-training eval
        tuner_state.message = "正在进行训练后评估..."
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        tokenizer2 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        if tokenizer2.pad_token is None:
            tokenizer2.pad_token = tokenizer2.eos_token
        model2 = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, torch_dtype="auto",
        )
        model2 = PeftModel.from_pretrained(model2, output_dir)
        model2.eval()

        def _gen2(prompt):
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

        samples = random.sample(pairs, min(15, len(pairs)))
        t_sem, t_ov = [], []
        for p in samples:
            gen = _gen2(p["prompt"])
            ea, eb = embedder.encode(gen), embedder.encode(p["target"])
            sim = float((ea @ eb) / (max(1e-8, (ea @ ea) ** 0.5 * (eb @ eb) ** 0.5)))
            t_sem.append(max(0, min(1, sim)))
            bg = lambda s: set(s[i:i + 2] for i in range(len(s) - 1))
            oa, ob = bg(gen), bg(p["target"])
            t_ov.append(len(oa & ob) / max(1e-8, len(oa | ob)))

        tuner_state.trained_semantic = sum(t_sem) / len(t_sem)
        tuner_state.trained_overlap = sum(t_ov) / len(t_ov)
        tuner_state.model = model2
        tuner_state.tokenizer = tokenizer2
        tuner_state.set_done()

        report_path = Path(output_dir) / "web_report.json"
        report_path.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "data_path": data_path,
            "config": {"r": r, "alpha": alpha, "lr": lr, "epochs": epochs},
            "loss_history": tuner_state.loss_history,
            "baseline": {"semantic": tuner_state.baseline_semantic,
                         "overlap": tuner_state.baseline_overlap},
            "trained": {"semantic": tuner_state.trained_semantic,
                        "overlap": tuner_state.trained_overlap},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    except Exception as e:
        traceback.print_exc()
        tuner_state.set_error(str(e))


def _load_data_file(path: str) -> tuple[list[dict], str]:
    p = Path(path)
    if not p.exists():
        return [], f"文件不存在: {path}"
    try:
        raw = p.read_text(encoding="utf-8").strip()
        data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
        pairs = _extract_pairs(data)
        tuner_state.loaded_data = data
        tuner_state.loaded_data_path = path
        return data, f"加载成功: {len(data)} 条数据, {len(pairs)} 个训练对"
    except Exception as e:
        return [], f"加载失败: {e}"


# ═══════════════════════════════════════════════════════
#  HTTP Handler
# ═══════════════════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, content, code=200):
        body = content.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _serve_file(self, path: str):
        fp = _HTML_DIR / path
        if fp.exists():
            self._html(fp.read_text(encoding="utf-8"))
        else:
            self._json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── Routing ──

    def do_GET(self):
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)

        if path == "/" or path == "/tuner":
            self._serve_file("tuner.html")
        elif path == "/romance":
            self._serve_file("romance.html")
        elif path == "/api/tuner/status":
            self._json(tuner_state.to_dict())
        elif path == "/api/tuner/data/preview":
            n = int(qs.get("n", [10])[0])
            self._json({"samples": tuner_state.loaded_data[:n], "total": len(tuner_state.loaded_data)})
        elif path == "/api/romance/state":
            self._json(romance_state.to_dict())
        elif path == "/api/romance/saves":
            saves = []
            sd = Path("data/saves")
            if sd.exists():
                for f in sorted(sd.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                    saves.append({"name": f.stem, "mtime": datetime.fromtimestamp(f.stat().st_mtime).isoformat()})
            self._json({"saves": saves})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read()

        # ── Tuner: data ──
        if path == "/api/tuner/data/load":
            data, msg = _load_data_file(body.get("path", ""))
            self._json({"ok": len(data) > 0, "message": msg, "count": len(data)})

        # ── Tuner: LoRA ──
        elif path == "/api/tuner/lora/start":
            if tuner_state.status == "running":
                self._json({"error": "训练已在运行中"}, 409)
                return
            data_file = body.get("data_file", "data/my_romance_data.jsonl")
            tuner_state.reset()
            threading.Thread(target=run_training, kwargs={
                "data_path": data_file,
                "model_name": body.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
                "output_dir": body.get("output_dir", "lora_web"),
                "r": int(body.get("r", 32)),
                "alpha": int(body.get("alpha", 64)),
                "lr": float(body.get("lr", 5e-5)),
                "epochs": int(body.get("epochs", 5)),
                "target_modules": body.get("target_modules", "all-linear"),
            }, daemon=True).start()
            self._json({"status": "started"})

        elif path == "/api/tuner/lora/cancel":
            tuner_state.set_error("用户取消")
            self._json({"status": "cancelled"})

        # ── Tuner: ROME edit ──
        elif path == "/api/tuner/rome/apply":
            if tuner_state.model is None:
                try:
                    self._load_base_model()
                except Exception as e:
                    self._json({"error": f"模型未加载: {e}"}, 400)
                    return
            try:
                from src.models.model_editor import ROMEEditor, EditRequest
                editor = ROMEEditor(tuner_state.model, tuner_state.tokenizer)
                req = EditRequest(
                    subject=body["subject"],
                    relation=body.get("relation", ""),
                    target=body["target"],
                    layer_idx=int(body.get("layer_idx", 15)),
                    lam=float(body.get("lam", 5.0)),
                )
                calib = body.get("calibration_prompts", [
                    "今天天气真好", "我喜欢喝咖啡", "最近工作很忙",
                ])
                editor.apply(req, calib)
                test_prompt = body.get("test_prompt", f"{req.subject}{req.relation}")
                inputs = tuner_state.tokenizer(test_prompt, return_tensors="pt").to(tuner_state.model.device)
                import torch
                with torch.no_grad():
                    out = tuner_state.model.generate(**inputs, max_new_tokens=64, do_sample=False)
                result_text = tuner_state.tokenizer.decode(out[0], skip_special_tokens=True)
                tuner_state.edit_results.append({
                    "type": "ROME", "subject": req.subject, "target": req.target,
                    "test_output": result_text, "time": datetime.now().isoformat(),
                })
                self._json({"ok": True, "test_output": result_text})
            except Exception as e:
                traceback.print_exc()
                self._json({"error": str(e)}, 500)

        # ── Tuner: MEMIT edit ──
        elif path == "/api/tuner/memit/apply":
            if tuner_state.model is None:
                self._load_base_model()
            try:
                from src.models.model_editor import MEMITEditor, EditRequest
                editor = MEMITEditor(tuner_state.model, tuner_state.tokenizer)
                requests = [EditRequest(**r) for r in body["requests"]]
                calib = body.get("calibration_prompts", [
                    "今天天气真好", "我喜欢喝咖啡",
                ])
                editor.apply_batch(requests, calib)
                results = []
                import torch
                for req in requests:
                    prompt = f"{req.subject}{req.relation or ''}"
                    inputs = tuner_state.tokenizer(prompt, return_tensors="pt").to(tuner_state.model.device)
                    with torch.no_grad():
                        out = tuner_state.model.generate(**inputs, max_new_tokens=64, do_sample=False)
                    results.append({
                        "subject": req.subject, "target": req.target,
                        "output": tuner_state.tokenizer.decode(out[0], skip_special_tokens=True),
                    })
                tuner_state.edit_results.extend([
                    {"type": "MEMIT", "subject": r["subject"], "target": r["target"],
                     "test_output": r["output"], "time": datetime.now().isoformat()}
                    for r in results
                ])
                self._json({"ok": True, "results": results})
            except Exception as e:
                traceback.print_exc()
                self._json({"error": str(e)}, 500)

        # ── Tuner: chat test ──
        elif path == "/api/tuner/chat":
            if tuner_state.status != "done" or tuner_state.model is None:
                self._json({"error": "模型尚未训练完成，无法对话"}, 400)
                return
            import torch
            msg = body.get("message", "").strip()
            if not msg:
                self._json({"error": "消息为空"}, 400)
                return
            try:
                msgs = [{"role": "user", "content": msg}]
                txt = tuner_state.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tuner_state.tokenizer(txt, return_tensors="pt").to(tuner_state.model.device)
                with torch.no_grad():
                    out = tuner_state.model.generate(
                        **inputs, max_new_tokens=256, do_sample=True,
                        temperature=0.85, top_p=0.9,
                        pad_token_id=tuner_state.tokenizer.pad_token_id,
                        eos_token_id=tuner_state.tokenizer.eos_token_id,
                    )
                full = tuner_state.tokenizer.decode(out[0], skip_special_tokens=True)
                resp = full[len(txt):].strip() if full.startswith(txt) else full.strip()
                self._json({"response": resp})
            except Exception as e:
                self._json({"error": str(e)}, 500)

        # ── Tuner: model operations ──
        elif path == "/api/tuner/model/load_base":
            try:
                self._load_base_model(body.get("model_name", "Qwen/Qwen2.5-7B-Instruct"))
                self._json({"ok": True, "message": "基础模型加载完成"})
            except Exception as e:
                self._json({"error": str(e)}, 500)

        elif path == "/api/tuner/model/load_lora":
            try:
                self._load_base_model()
                from peft import PeftModel
                adapter_path = body.get("adapter_path", "lora_web")
                tuner_state.model = PeftModel.from_pretrained(tuner_state.model, adapter_path)
                tuner_state.status = "done"
                tuner_state.message = f"LoRA 适配器已加载: {adapter_path}"
                self._json({"ok": True, "message": tuner_state.message})
            except Exception as e:
                self._json({"error": str(e)}, 500)

        # ── Romance: new game ──
        elif path == "/api/romance/new":
            character_path = body.get("character", "config/characters/default.yaml")
            model_path = body.get("model_path", "")
            romance_state.init_engine(character_path, model_path)
            self._json(romance_state.to_dict())

        # ── Romance: chat ──
        elif path == "/api/romance/chat":
            if romance_state.engine is None:
                self._json({"error": "游戏未初始化"}, 400)
                return
            msg = body.get("message", "").strip()
            if not msg:
                self._json({"error": "消息为空"}, 400)
                return

            eng = romance_state.engine
            ctx = eng.process_player_input(msg)

            if romance_state.llm:
                from src.main import build_messages
                sp = eng.character.format_system_prompt(
                    scene=eng.state_machine.current_scene.description if eng.state_machine.current_scene else "日常"
                )
                messages = build_messages(ctx, sp)
                response = romance_state.llm.chat(messages)
            else:
                response = f"*{eng.character.profile.name}微微一笑* 嗯，我在听呢..."

            eng.record_assistant_response(response)

            self._json({
                "response": response,
                "character": {
                    "name": eng.character.profile.name,
                    "affection": eng.character.profile.affection_score,
                    "relationship": eng.character.get_relationship_label(),
                },
                "scene": eng.state_machine.current_scene.scene_id if eng.state_machine.current_scene else "",
                "choices": [
                    {"id": c.choice_id, "text": c.text, "affection": c.affection_delta}
                    for c in eng.state_machine.available_choices(eng.character.story_flags.unlocked_scenes)
                ] if eng.state_machine.current_scene else [],
            })

        # ── Romance: choice ──
        elif path == "/api/romance/choice":
            if romance_state.engine is None:
                self._json({"error": "游戏未初始化"}, 400)
                return
            choice_id = body.get("choice_id", "")
            sm = romance_state.engine.state_machine
            avail = sm.available_choices(romance_state.engine.character.story_flags.unlocked_scenes)
            chosen = next((c for c in avail if c.choice_id == choice_id), None)
            if chosen is None:
                self._json({"error": f"无效选择: {choice_id}"}, 400)
                return
            romance_state.engine.apply_choice(chosen)
            self._json({
                "ok": True,
                "affection_delta": chosen.affection_delta,
                "new_scene": sm.current_scene.scene_id if sm.current_scene else "",
                "scene_desc": sm.current_scene.description if sm.current_scene else "",
                "choices": [
                    {"id": c.choice_id, "text": c.text, "affection": c.affection_delta}
                    for c in sm.available_choices(romance_state.engine.character.story_flags.unlocked_scenes)
                ] if sm.current_scene else [],
                "character": {
                    "name": romance_state.engine.character.profile.name,
                    "affection": romance_state.engine.character.profile.affection_score,
                    "relationship": romance_state.engine.character.get_relationship_label(),
                },
            })

        # ── Romance: save/load ──
        elif path == "/api/romance/save":
            if romance_state.engine is None:
                self._json({"error": "游戏未初始化"}, 400)
                return
            slot = body.get("slot", "auto")
            romance_state.engine.save(slot)
            self._json({"ok": True, "slot": slot})

        elif path == "/api/romance/load":
            if romance_state.engine is None:
                romance_state.init_engine()
            slot = body.get("slot", "auto")
            ok = romance_state.engine.load(slot)
            if ok:
                self._json({"ok": True, "state": romance_state.to_dict()})
            else:
                self._json({"error": f"存档不存在: {slot}"}, 404)

        else:
            self._json({"error": "not found"}, 404)

    def _load_base_model(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        tuner_state.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )
        if tuner_state.tokenizer.pad_token is None:
            tuner_state.tokenizer.pad_token = tuner_state.tokenizer.eos_token
        tuner_state.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, torch_dtype="auto",
        )
        tuner_state.status = "done"
        tuner_state.message = f"基础模型已加载: {model_name}"


def main():
    import signal
    port = 8765
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\n  🧠 NeuralEditor API Server")
    print(f"  🔧 微调端: http://localhost:{port}/")
    print(f"  💕 恋爱端: http://localhost:{port}/romance")
    print(f"  ⏹  按 Ctrl+C 停止\n")

    def _sigint(signum, frame):
        print("\n  服务器已停止。")
        server.server_close()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务器已停止。")
        server.server_close()


if __name__ == "__main__":
    main()
