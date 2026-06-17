"""Tests for story engine integration."""

import tempfile

from src.core.character import Character
from src.core.memory_system import MemorySystem
from src.core.state_machine import Scene, StoryStateMachine
from src.core.story_engine import StoryEngine
from src.utils.config_loader import MemoryConfig, StoryConfig
from src.utils.json_storage import JSONStorage


class TestStoryEngine:
    def setup_method(self):
        self.char = Character()
        self.mem = MemorySystem()
        self.sm = StoryStateMachine()
        self.config = StoryConfig(auto_save_interval=2)
        self.tmpdir = tempfile.mkdtemp()
        self.storage = JSONStorage(self.tmpdir)

    def test_process_turn(self):
        engine = StoryEngine(self.char, self.mem, self.sm, self.config)
        context = engine.process_player_input("你好")

        assert context["scene"] == ""
        assert len(context["short_term"]) == 1
        assert engine.turn_count == 1

    def test_save_and_load(self):
        engine = StoryEngine(
            self.char, self.mem, self.sm, self.config, self.storage
        )
        engine.turn_count = 5
        engine.save("test_slot")

        new_engine = StoryEngine(
            self.char, self.mem, self.sm, self.config, self.storage
        )
        assert new_engine.load("test_slot") is True
        assert new_engine.turn_count == 5

    def test_load_missing(self):
        engine = StoryEngine(
            self.char, self.mem, self.sm, self.config, self.storage
        )
        assert engine.load("missing") is False

    def test_auto_save(self):
        engine = StoryEngine(
            self.char, self.mem, self.sm, self.config, self.storage
        )
        engine.process_player_input("A")
        engine.process_player_input("B")
        # auto_save_interval=2, should have saved
        assert self.storage.load("char_default_auto") is not None
