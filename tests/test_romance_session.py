"""Tests for RomanceState and RomanceSessionManager — session lifecycle and concurrency."""

import threading
from unittest.mock import MagicMock

import pytest

from src.web.server import RomanceState, RomanceSessionManager


# ── Helpers ──

def _make_state() -> RomanceState:
    """Create a RomanceState with no engine / LLM (safe for unit tests)."""
    state = RomanceState()
    state.engine = None
    state.llm = None
    return state


# ═══════════════════════════════════════════════════════════
#  Session initialization
# ═══════════════════════════════════════════════════════════

class TestSessionInit:
    def test_default_state(self):
        state = RomanceState()
        assert state.engine is None
        assert state.llm is None
        assert state.session_id == ""
        assert state.session_start == ""
        assert state.consent is False
        assert state.chat_log == []
        assert state.choices_log == []

    def test_init_session_sets_fields(self):
        state = _make_state()
        state.init_session(consent=True)
        assert state.session_id.startswith("session_")
        assert state.session_start != ""
        assert state.consent is True
        assert state.chat_log == []
        assert state.choices_log == []

    def test_init_session_resets_logs(self):
        state = _make_state()
        state.chat_log = [{"role": "user", "content": "old"}]
        state.choices_log = [{"choice_id": "x", "text": "old", "affection_delta": 0, "time": ""}]
        state.init_session(consent=False)
        assert state.chat_log == []
        assert state.choices_log == []

    def test_init_session_without_consent(self):
        state = _make_state()
        state.init_session(consent=False)
        assert state.consent is False
        assert state.session_id != ""


# ═══════════════════════════════════════════════════════════
#  Message and choice logging
# ═══════════════════════════════════════════════════════════

class TestLogging:
    def test_log_message(self):
        state = _make_state()
        state.log_message("user", "你好")
        state.log_message("assistant", "你好！")
        assert len(state.chat_log) == 2
        assert state.chat_log[0]["role"] == "user"
        assert state.chat_log[0]["content"] == "你好"
        assert "time" in state.chat_log[0]
        assert state.chat_log[1]["role"] == "assistant"

    def test_log_choice(self):
        state = _make_state()
        state.log_choice("c1", "打招呼", 5)
        assert len(state.choices_log) == 1
        assert state.choices_log[0]["choice_id"] == "c1"
        assert state.choices_log[0]["text"] == "打招呼"
        assert state.choices_log[0]["affection_delta"] == 5
        assert "time" in state.choices_log[0]

    def test_log_choice_negative_delta(self):
        state = _make_state()
        state.log_choice("c_bad", "说坏话", -10)
        assert state.choices_log[0]["affection_delta"] == -10


# ═══════════════════════════════════════════════════════════
#  State dictionary export
# ═══════════════════════════════════════════════════════════

class TestToDict:
    def test_to_dict_no_engine(self):
        state = _make_state()
        d = state.to_dict()
        assert d == {"ready": False}

    def test_to_dict_with_engine(self):
        """Simulate a fully-initialized engine via mocking."""
        state = RomanceState()
        state.lock = threading.Lock()
        mock_engine = MagicMock()
        mock_engine.character.profile.name = "测试角色"
        mock_engine.character.profile.affection_score = 42
        mock_engine.character.get_relationship_label.return_value = "朋友"
        mock_engine.character.profile.personality_traits = ["温柔"]
        mock_engine.character.profile.background = "小镇"
        mock_engine.character.profile.speaking_style = "轻声细语"
        mock_engine.state.current_scene.scene_id = "scene_01"
        mock_engine.state.current_scene.description = "咖啡馆"
        mock_engine.turn_count = 7
        mock_engine.state.available_choices.return_value = []
        mock_engine.character.story_flags.unlocked_scenes = set()
        state.engine = mock_engine
        state.llm = None

        d = state.to_dict()
        assert d["ready"] is True
        assert d["character"]["name"] == "测试角色"
        assert d["character"]["affection"] == 42
        assert d["character"]["relationship"] == "朋友"
        assert d["current_scene"] == "scene_01"
        assert d["scene_desc"] == "咖啡馆"
        assert d["turn_count"] == 7
        assert d["has_llm"] is False

    def test_to_dict_has_llm(self):
        state = RomanceState()
        state.lock = threading.Lock()
        mock_engine = MagicMock()
        mock_engine.character.profile.name = "x"
        mock_engine.character.profile.affection_score = 0
        mock_engine.character.get_relationship_label.return_value = "陌生人"
        mock_engine.character.profile.personality_traits = []
        mock_engine.character.profile.background = ""
        mock_engine.character.profile.speaking_style = ""
        mock_engine.state.current_scene = None
        mock_engine.turn_count = 0
        mock_engine.character.story_flags.unlocked_scenes = set()
        state.engine = mock_engine
        state.llm = object()  # any non-None value
        d = state.to_dict()
        assert d["has_llm"] is True


# ═══════════════════════════════════════════════════════════
#  Thread safety (concurrent access)
# ═══════════════════════════════════════════════════════════

class TestConcurrency:
    def test_concurrent_log_messages(self):
        """Multiple threads logging messages should not lose data."""
        state = _make_state()
        threads = []
        n_per_thread = 50
        n_threads = 4

        def _log(thread_id: int):
            for i in range(n_per_thread):
                state.log_message("user", f"t{thread_id}-msg{i}")

        for tid in range(n_threads):
            t = threading.Thread(target=_log, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(state.chat_log) == n_threads * n_per_thread

    def test_concurrent_log_choices(self):
        state = _make_state()
        threads = []
        n_per_thread = 30
        n_threads = 4

        def _log(tid: int):
            for i in range(n_per_thread):
                state.log_choice(f"c_t{tid}_{i}", f"choice-{tid}-{i}", tid)

        for tid in range(n_threads):
            t = threading.Thread(target=_log, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(state.choices_log) == n_threads * n_per_thread


# ═══════════════════════════════════════════════════════════
#  RomanceSessionManager
# ═══════════════════════════════════════════════════════════

class TestSessionManager:
    def test_get_creates_new_session(self):
        mgr = RomanceSessionManager()
        state = mgr.get("hash_abc")
        assert state is not None
        assert isinstance(state, RomanceState)

    def test_get_returns_same_session(self):
        mgr = RomanceSessionManager()
        s1 = mgr.get("hash_xyz")
        s2 = mgr.get("hash_xyz")
        assert s1 is s2

    def test_get_different_keys_isolated(self):
        mgr = RomanceSessionManager()
        s_a = mgr.get("hash_a")
        s_b = mgr.get("hash_b")
        assert s_a is not s_b
        s_a.log_message("user", "a-msg")
        assert len(s_b.chat_log) == 0

    def test_remove_session(self):
        mgr = RomanceSessionManager()
        mgr.get("hash_rem")
        mgr.remove("hash_rem")
        # A new get should create a fresh session
        s_new = mgr.get("hash_rem")
        assert s_new.session_id == ""

    def test_list_active(self):
        mgr = RomanceSessionManager()
        s = mgr.get("hash_list")
        s.init_session(consent=True)
        s.log_message("user", "hello")
        active = mgr.list_active()
        assert len(active) >= 1
        found = [a for a in active if a["key_prefix"].startswith("hash_list")]
        assert len(found) == 1
        assert found[0]["message_count"] == 1
        assert found[0]["consent"] is True

    def test_list_active_empty(self):
        mgr = RomanceSessionManager()
        assert mgr.list_active() == []

    def test_concurrent_session_get(self):
        """Concurrent gets for the same key should return the same session."""
        mgr = RomanceSessionManager()
        results = []

        def _get():
            results.append(mgr.get("concurrent_key"))

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All must be the same object
        first = results[0]
        for r in results[1:]:
            assert r is first
