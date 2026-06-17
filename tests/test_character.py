"""Tests for character state management."""

import pytest
from src.core.character import Character, Profile


class TestCharacter:
    def test_default_creation(self):
        char = Character()
        assert char.profile.name == "小棠"
        assert char.profile.affection_score == 0
        assert char.get_relationship_label() == "陌生人"

    def test_affection_clamping(self):
        char = Character()
        char.profile.affection_score = 150
        # Pydantic validator should clamp on creation/validation
        char2 = Character.from_dict(char.to_dict())
        assert char2.profile.affection_score == 100

    def test_adjust_affection(self):
        char = Character()
        char.adjust_affection(50)
        assert char.profile.affection_score == 40  # 50 * 0.8 damping

    def test_relationship_labels(self):
        char = Character()
        char.profile.affection_score = -50
        assert char.get_relationship_label() == "疏远"

        char.profile.affection_score = 20
        assert char.get_relationship_label() == "朋友"

        char.profile.affection_score = 50
        assert char.get_relationship_label() == "亲密"

        char.profile.affection_score = 80
        assert char.get_relationship_label() == "恋人"

    def test_format_system_prompt(self):
        char = Character()
        prompt = char.format_system_prompt(
            template="你是{name}，{relationship}",
            scene="图书馆"
        )
        assert "小棠" in prompt
        assert "陌生人" in prompt

    def test_format_system_prompt_no_template(self):
        """If character has its own system_prompt, use it."""
        char = Character()
        char.system_prompt = "自定义提示词"
        prompt = char.format_system_prompt(template="模板")
        assert prompt == "自定义提示词"

    def test_roundtrip_serialization(self):
        char = Character()
        char.profile.affection_score = 42
        char.memory.important_events.append("测试事件")

        data = char.to_dict()
        restored = Character.from_dict(data)

        assert restored.profile.affection_score == 42
        assert restored.memory.important_events == ["测试事件"]
