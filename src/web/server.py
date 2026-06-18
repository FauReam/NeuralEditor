#!/usr/bin/env python3
"""NeuralEditor API Server — 为 tuner.html 和 romance.html 提供后端服务。

启动: python -m src.web.server
端口: 8765

认证模式:
  - 服务端通过 Admin 面板（/admin）签发 API Key
  - 客户端在请求头中携带 X-API-Key 或 URL 参数 ?api_key=xxx
  - 每个 API Key 拥有独立的游戏会话
"""

import hashlib
import json
import os
import re
import secrets
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ═══════════════════════════════════════════════════════
#  API Key Manager
# ═══════════════════════════════════════════════════════

class APIKeyManager:
    """管理 API Key 的签发、验证、撤销。"""

    KEY_PREFIX = "ne_"
    STORE_PATH = Path("data/api_keys.json")
    ADMIN_PASSWORD_FILE = Path("data/admin_password.hash")

    def __init__(self):
        self.lock = threading.Lock()
        self._ensure_store()

    def _ensure_store(self):
        self.STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not self.STORE_PATH.exists():
            self.STORE_PATH.write_text(json.dumps({"keys": {}}, indent=2), encoding="utf-8")

    def _read_store(self) -> dict:
        try:
            raw = self.STORE_PATH.read_text(encoding="utf-8")
            return json.loads(raw)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"keys": {}}

    def _write_store(self, data: dict):
        self.STORE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def generate_key(
        self,
        description: str = "",
        max_requests: int = 0,
        expires_days: int = 0,
        permissions: list[str] | None = None,
    ) -> dict:
        """签发一个新的 API Key"""
        with self.lock:
            store = self._read_store()
            raw = secrets.token_hex(24)  # 48 hex chars
            api_key = f"{self.KEY_PREFIX}{raw}"

            now = datetime.now().isoformat()
            expires_at = (
                (datetime.now() + timedelta(days=expires_days)).isoformat()
                if expires_days > 0
                else None
            )

            key_info = {
                "key_hash": self._hash(api_key),
                "key_prefix": api_key[:8] + "..." + api_key[-4:],  # 显示用
                "description": description,
                "permissions": permissions or ["romance"],
                "max_requests": max_requests,
                "request_count": 0,
                "created_at": now,
                "expires_at": expires_at,
                "revoked": False,
                "last_used_at": None,
            }

            store["keys"][key_info["key_hash"]] = key_info
            self._write_store(store)

            # 返回完整 key（仅此刻可见）
            result = dict(key_info)
            result["api_key"] = api_key
            return result

    def validate_key(self, api_key: str) -> dict | None:
        """验证 API Key，返回 key_info 或 None"""
        if not api_key or not api_key.startswith(self.KEY_PREFIX):
            return None

        with self.lock:
            store = self._read_store()
            key_hash = self._hash(api_key)
            info = store["keys"].get(key_hash)
            if info is None:
                return None
            if info.get("revoked", False):
                return None
            if info.get("expires_at"):
                try:
                    expires = datetime.fromisoformat(info["expires_at"])
                    if datetime.now() > expires:
                        return None
                except (ValueError, TypeError):
                    pass
            if info.get("max_requests", 0) > 0 and info.get("request_count", 0) >= info["max_requests"]:
                return None

            # 更新使用计数和时间
            info["request_count"] = info.get("request_count", 0) + 1
            info["last_used_at"] = datetime.now().isoformat()
            self._write_store(store)

            return info

    def revoke_key(self, api_key_or_hash: str):
        """撤销一个 API Key

        接受完整 API Key 或其 SHA256 哈希值。
        先尝试直接匹配 hash，再尝试 hash 输入后匹配。
        """
        with self.lock:
            store = self._read_store()
            # 支持两种输入：完整 Key（需要 hash）或已有 hash
            if api_key_or_hash in store["keys"]:
                key_hash = api_key_or_hash
            else:
                key_hash = self._hash(api_key_or_hash)
                if key_hash not in store["keys"]:
                    return False
            store["keys"][key_hash]["revoked"] = True
            self._write_store(store)
            return True

    def list_keys(self) -> list[dict]:
        """列出所有 API Key（不包含完整 key）"""
        with self.lock:
            store = self._read_store()
            return list(store["keys"].values())

    # ── Admin password ──

    def set_admin_password(self, password: str):
        """设置 Admin 密码（SHA256 哈希存储）"""
        self.ADMIN_PASSWORD_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.ADMIN_PASSWORD_FILE.write_text(self._hash(password), encoding="utf-8")

    def check_admin_password(self, password: str) -> bool:
        """验证 Admin 密码"""
        if not self.ADMIN_PASSWORD_FILE.exists():
            # 首次启动没有密码，生成一个随机默认密码
            default_pw = secrets.token_hex(8)
            self.set_admin_password(default_pw)
            print(f"\n  🔑 首次启动，Admin 默认密码: {default_pw}")
            print(f"     请登录 /admin 后修改密码\n")
            return password == default_pw

        stored = self.ADMIN_PASSWORD_FILE.read_text(encoding="utf-8").strip()
        return secrets.compare_digest(stored, self._hash(password))

    @staticmethod
    def _hash(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


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
    """单个用户的 Romance 游戏会话状态"""

    def __init__(self):
        self.lock = threading.Lock()
        self.engine = None
        self.llm = None
        self.character_path = ""
        self.model_path = ""
        self.session_id = ""
        self.session_start = ""
        self.consent = False
        self.chat_log: list[dict] = []
        self.choices_log: list[dict] = []

    def init_session(self, consent: bool):
        self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_start = datetime.now().isoformat()
        self.consent = consent
        self.chat_log = []
        self.choices_log = []

    def log_message(self, role: str, content: str):
        self.chat_log.append({
            "role": role,
            "content": content,
            "time": datetime.now().isoformat(),
        })

    def log_choice(self, choice_id: str, text: str, delta: int):
        self.choices_log.append({
            "choice_id": choice_id,
            "text": text,
            "affection_delta": delta,
            "time": datetime.now().isoformat(),
        })

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
        with self.lock:
            if self.engine is None:
                return {"ready": False}
            char = self.engine.character
            sm = self.engine.state
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
                    for c in sm.available_choices(set(self.engine.character.story_flags.unlocked_scenes))
                ] if sm.current_scene else [],
                "has_llm": self.llm is not None,
            }


class RomanceSessionManager:
    """管理多用户 Romance 会话。每个 API Key 对应一个独立会话。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.sessions: dict[str, RomanceState] = {}  # key_hash → RomanceState

    def get(self, key_hash: str) -> RomanceState:
        with self.lock:
            if key_hash not in self.sessions:
                self.sessions[key_hash] = RomanceState()
            return self.sessions[key_hash]

    def remove(self, key_hash: str):
        with self.lock:
            self.sessions.pop(key_hash, None)

    def list_active(self) -> list[dict]:
        with self.lock:
            result = []
            for kh, state in self.sessions.items():
                result.append({
                    "key_prefix": kh[:12] + "..." if len(kh) > 12 else kh,
                    "has_engine": state.engine is not None,
                    "session_id": state.session_id,
                    "message_count": len(state.chat_log),
                    "consent": state.consent,
                })
            return result


tuner_state = TunerState()
romance_sessions = RomanceSessionManager()
api_key_manager = APIKeyManager()

# HTML files loaded lazily
_HTML_DIR = Path(__file__).parent


# ═══════════════════════════════════════════════════════
#  Auth helpers
# ═══════════════════════════════════════════════════════

ADMIN_SESSION_TOKENS: dict[str, float] = {}  # token → expiry_timestamp
_admin_sessions_lock = threading.Lock()


def _extract_api_key(handler: BaseHTTPRequestHandler) -> str | None:
    """从请求中提取 API Key。先查 Header，再查 URL 参数。"""
    # Header: X-API-Key
    key = handler.headers.get("X-API-Key", "")
    if key:
        return key.strip()

    # URL query: ?api_key=xxx
    qs = parse_qs(urlparse(handler.path).query)
    keys = qs.get("api_key", [])
    if keys:
        return keys[0].strip()

    # Bearer token
    auth = handler.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()

    return None


def _authenticate(handler: BaseHTTPRequestHandler) -> dict | None:
    """验证请求。返回 key_info 或 None。"""
    api_key = _extract_api_key(handler)
    if not api_key:
        return None
    return api_key_manager.validate_key(api_key)


def _check_admin(handler: BaseHTTPRequestHandler) -> bool:
    """检查是否是 Admin 请求（通过 Admin Session Token）"""
    cookies = handler.headers.get("Cookie", "")
    token = ""
    for part in cookies.replace(" ", "").split(";"):
        if part.startswith("admin_token="):
            token = part.split("=", 1)[1]
            break

    with _admin_sessions_lock:
        if token in ADMIN_SESSION_TOKENS:
            expiry = ADMIN_SESSION_TOKENS[token]
            if time.time() < expiry:
                return True
            del ADMIN_SESSION_TOKENS[token]
    return False


def _create_admin_session() -> str:
    token = secrets.token_hex(32)
    with _admin_sessions_lock:
        ADMIN_SESSION_TOKENS[token] = time.time() + 3600  # 1 小时有效
    return token


def _get_romance_state(handler: BaseHTTPRequestHandler) -> RomanceState:
    """根据请求的 API Key 获取对应的 RomanceState"""
    api_key = _extract_api_key(handler)
    if api_key:
        key_hash = api_key_manager._hash(api_key)
        return romance_sessions.get(key_hash)
    # 无 key 的本地请求使用默认会话
    return romance_sessions.get("__local__")


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
        raw = Path(data_path).read_text(encoding="utf-8").strip()
        data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
        pairs = _extract_pairs(data)
        if not pairs:
            tuner_state.set_error("无法从数据中提取训练对")
            return

        tuner_state.loaded_data = data
        tuner_state.loaded_data_path = data_path

        tuner_state.baseline_semantic, tuner_state.baseline_overlap = _run_baseline_eval(pairs, model_name)

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

    def _json(self, data, code=200, extra_headers: dict | None = None):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Credentials", "true")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key, Authorization")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.end_headers()

    # ── Routing ──

    def do_GET(self):
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)

        # ── Static pages ──
        if path == "/" or path == "/tuner":
            self._serve_file("tuner.html")
        elif path == "/romance":
            self._serve_file("romance.html")
        elif path == "/admin":
            self._serve_file("admin.html")
        elif path == "/client":
            self._serve_file("client.html")

        # ── Tuner API (no auth required for local use) ──
        elif path == "/api/tuner/status":
            self._json(tuner_state.to_dict())
        elif path == "/api/tuner/data/preview":
            n = int(qs.get("n", [10])[0])
            self._json({"samples": tuner_state.loaded_data[:n], "total": len(tuner_state.loaded_data)})

        # ── Romance API (API key required) ──
        elif path == "/api/romance/state":
            state = _get_romance_state(self)
            self._json(state.to_dict())

        elif path == "/api/romance/saves":
            saves = []
            sd = Path("data/saves")
            if sd.exists():
                for f in sorted(sd.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                    saves.append({"name": f.stem, "mtime": datetime.fromtimestamp(f.stat().st_mtime).isoformat()})
            self._json({"saves": saves})

        # ── Health / info ──
        elif path == "/api/health":
            self._json({
                "status": "ok",
                "version": "0.2.0",
                "server_time": datetime.now().isoformat(),
                "active_sessions": len(romance_sessions.list_active()),
            })

        elif path == "/api/server/info":
            self._json({
                "name": "NeuralEditor API Server",
                "version": "0.2.0",
                "endpoints": {
                    "romance": "/romance",
                    "admin": "/admin",
                    "tuner": "/",
                },
                "auth_required": True,
                "auth_header": "X-API-Key",
            })

        # ── Admin API ──
        elif path == "/api/admin/keys":
            if not _check_admin(self):
                self._json({"error": "需要 Admin 登录"}, 401)
                return
            keys = api_key_manager.list_keys()
            self._json({"keys": keys, "total": len(keys)})

        elif path == "/api/admin/sessions":
            if not _check_admin(self):
                self._json({"error": "需要 Admin 登录"}, 401)
                return
            self._json({"sessions": romance_sessions.list_active()})

        elif path == "/api/admin/check":
            self._json({"authenticated": _check_admin(self)})

        elif path == "/api/auth/status":
            key_info = _authenticate(self)
            if key_info:
                self._json({
                    "authenticated": True,
                    "key_prefix": key_info["key_prefix"],
                    "permissions": key_info["permissions"],
                    "description": key_info["description"],
                })
            else:
                self._json({"authenticated": False}, 401)

        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read()

        # ── Admin: login ──
        if path == "/api/admin/login":
            password = body.get("password", "")
            if api_key_manager.check_admin_password(password):
                token = _create_admin_session()
                self._json({"ok": True, "message": "登录成功"}, 200,
                          extra_headers={"Set-Cookie": f"admin_token={token}; Path=/; HttpOnly; SameSite=Lax"})
            else:
                self._json({"error": "密码错误"}, 403)

        elif path == "/api/admin/logout":
            self._json({"ok": True}, 200,
                      extra_headers={"Set-Cookie": "admin_token=; Path=/; Max-Age=0"})

        elif path == "/api/admin/password/change":
            if not _check_admin(self):
                self._json({"error": "需要 Admin 登录"}, 401)
                return
            old_pw = body.get("old_password", "")
            new_pw = body.get("new_password", "")
            if not new_pw or len(new_pw) < 4:
                self._json({"error": "新密码至少 4 位"}, 400)
                return
            if not api_key_manager.check_admin_password(old_pw):
                self._json({"error": "旧密码错误"}, 403)
                return
            api_key_manager.set_admin_password(new_pw)
            self._json({"ok": True, "message": "密码已更新"})

        # ── Admin: key management ──
        elif path == "/api/admin/keys/generate":
            if not _check_admin(self):
                self._json({"error": "需要 Admin 登录"}, 401)
                return
            key_data = api_key_manager.generate_key(
                description=body.get("description", ""),
                max_requests=int(body.get("max_requests", 0)),
                expires_days=int(body.get("expires_days", 0)),
                permissions=body.get("permissions", ["romance"]),
            )
            self._json(key_data)

        elif path == "/api/admin/keys/revoke":
            if not _check_admin(self):
                self._json({"error": "需要 Admin 登录"}, 401)
                return
            key_to_revoke = body.get("key_prefix", "") or body.get("api_key", "")
            if not key_to_revoke:
                self._json({"error": "请提供 key_prefix 或 api_key"}, 400)
                return
            ok = api_key_manager.revoke_key(key_to_revoke)
            self._json({"ok": ok, "message": "已撤销" if ok else "未找到该 Key"})

        # ── Auth: verify key (for client use) ──
        elif path == "/api/auth/verify":
            api_key = body.get("api_key", "")
            info = api_key_manager.validate_key(api_key)
            if info:
                self._json({
                    "valid": True,
                    "key_prefix": info["key_prefix"],
                    "permissions": info["permissions"],
                    "description": info["description"],
                })
            else:
                self._json({"valid": False, "error": "无效或过期的 API Key"}, 401)

        # ── Tuner: data ──
        elif path == "/api/tuner/data/load":
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

        # ── Romance: new game (auth optional, local fallback) ──
        elif path == "/api/romance/new":
            state = _get_romance_state(self)
            character_path = body.get("character", "config/characters/default.yaml")
            model_path = body.get("model_path", "")
            state.init_engine(character_path, model_path)
            self._json(state.to_dict())

        # ── Romance: chat ──
        elif path == "/api/romance/chat":
            state = _get_romance_state(self)
            if state.engine is None:
                self._json({"error": "游戏未初始化"}, 400)
                return
            msg = body.get("message", "").strip()
            if not msg:
                self._json({"error": "消息为空"}, 400)
                return

            state.log_message("user", msg)
            eng = state.engine
            ctx = eng.process_player_input(msg)

            if state.llm:
                from src.main import build_messages
                sp = eng.character.format_system_prompt(
                    scene=eng.state.current_scene.description if eng.state.current_scene else "日常"
                )
                messages = build_messages(ctx, sp)
                response = state.llm.chat(messages)
            else:
                response = f"*{eng.character.profile.name}微微一笑* 嗯，我在听呢..."

            eng.record_assistant_response(response)
            state.log_message("assistant", response)

            self._json({
                "response": response,
                "character": {
                    "name": eng.character.profile.name,
                    "affection": eng.character.profile.affection_score,
                    "relationship": eng.character.get_relationship_label(),
                },
                "scene": eng.state.current_scene.scene_id if eng.state.current_scene else "",
                "choices": [
                    {"id": c.choice_id, "text": c.text, "affection": c.affection_delta}
                    for c in eng.state.available_choices(set(eng.character.story_flags.unlocked_scenes))
                ] if eng.state.current_scene else [],
            })

        # ── Romance: choice ──
        elif path == "/api/romance/choice":
            state = _get_romance_state(self)
            if state.engine is None:
                self._json({"error": "游戏未初始化"}, 400)
                return
            choice_id = body.get("choice_id", "")
            sm = state.engine.state
            avail = sm.available_choices(set(state.engine.character.story_flags.unlocked_scenes))
            chosen = next((c for c in avail if c.choice_id == choice_id), None)
            if chosen is None:
                self._json({"error": f"无效选择: {choice_id}"}, 400)
                return
            state.engine.apply_choice(chosen)
            state.log_choice(choice_id, chosen.text, chosen.affection_delta)
            self._json({
                "ok": True,
                "affection_delta": chosen.affection_delta,
                "new_scene": sm.current_scene.scene_id if sm.current_scene else "",
                "scene_desc": sm.current_scene.description if sm.current_scene else "",
                "choices": [
                    {"id": c.choice_id, "text": c.text, "affection": c.affection_delta}
                    for c in sm.available_choices(set(state.engine.character.story_flags.unlocked_scenes))
                ] if sm.current_scene else [],
                "character": {
                    "name": state.engine.character.profile.name,
                    "affection": state.engine.character.profile.affection_score,
                    "relationship": state.engine.character.get_relationship_label(),
                },
            })

        # ── Romance: save/load ──
        elif path == "/api/romance/save":
            state = _get_romance_state(self)
            if state.engine is None:
                self._json({"error": "游戏未初始化"}, 400)
                return
            slot = body.get("slot", "auto")
            state.engine.save(slot)
            self._json({"ok": True, "slot": slot})

        elif path == "/api/romance/load":
            state = _get_romance_state(self)
            if state.engine is None:
                state.init_engine()
            slot = body.get("slot", "auto")
            ok = state.engine.load(slot)
            if ok:
                self._json({"ok": True, "state": state.to_dict()})
            else:
                self._json({"error": f"存档不存在: {slot}"}, 404)

        # ── Romance: session management ──
        elif path == "/api/romance/session/start":
            state = _get_romance_state(self)
            consent = body.get("consent", False)
            state.init_session(consent)
            self._json({"ok": True, "session_id": state.session_id, "consent": consent})

        elif path == "/api/romance/session/save":
            state = _get_romance_state(self)
            sid = state.session_id
            if not sid:
                self._json({"error": "没有活跃会话"}, 400)
                return
            session_dir = Path("data/sessions")
            session_dir.mkdir(parents=True, exist_ok=True)
            session_data = {
                "session_id": sid,
                "start_time": state.session_start,
                "end_time": datetime.now().isoformat(),
                "consent": state.consent,
                "character": state.to_dict().get("character", {}),
                "messages": state.chat_log,
                "choices": state.choices_log,
            }
            (session_dir / f"{sid}.json").write_text(
                json.dumps(session_data, ensure_ascii=False, indent=2), encoding="utf-8")
            self._json({"ok": True, "session_id": sid})

        elif path == "/api/romance/session/feedback":
            state = _get_romance_state(self)
            sid = body.get("session_id", state.session_id)
            if not sid:
                self._json({"error": "没有活跃会话"}, 400)
                return
            rating = int(body.get("rating", 0))
            feedback_text = body.get("feedback", "").strip()

            fb_dir = Path("data/feedback")
            fb_dir.mkdir(parents=True, exist_ok=True)

            fb_data = {
                "session_id": sid,
                "submitted_at": datetime.now().isoformat(),
                "rating": rating,
                "feedback": feedback_text,
                "messages": state.chat_log,
                "choices": state.choices_log,
                "character": state.to_dict().get("character", {}),
            }
            (fb_dir / f"{sid}.json").write_text(
                json.dumps(fb_data, ensure_ascii=False, indent=2), encoding="utf-8")
            self._json({"ok": True, "message": "反馈已提交，感谢你的参与！"})

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
    import io
    # Fix emoji encoding on Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    port = 8765
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    banner = (
        f"\n  [*] NeuralEditor API Server v0.2\n"
        f"  ----------------------------------------\n"
        f"  Tuner:    http://localhost:{port}/\n"
        f"  Romance:  http://localhost:{port}/romance\n"
        f"  Client:   http://localhost:{port}/client\n"
        f"  Admin:    http://localhost:{port}/admin\n"
        f"  ----------------------------------------\n"
        f"  API:  Header X-API-Key or ?api_key=xxx\n"
        f"  Stop: Ctrl+C\n"
    )
    print(banner)

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
