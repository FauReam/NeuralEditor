"""Character state management."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.utils.config_loader import load_yaml


class Profile(BaseModel):
    name: str = "小棠"
    age: int | None = None
    personality_traits: list[str] = Field(default_factory=list)
    speaking_style: str = ""
    background: str = ""
    relationship_status: str = "stranger"
    affection_score: int = 0


class Memory(BaseModel):
    short_term: list[str] = Field(default_factory=list)
    long_term_summaries: list[str] = Field(default_factory=list)
    important_events: list[str] = Field(default_factory=list)


class StoryFlags(BaseModel):
    chapter: int = 1
    unlocked_scenes: list[str] = Field(default_factory=list)
    player_choices: dict[str, Any] = Field(default_factory=dict)


class Character(BaseModel):
    character_id: str = "char_default"
    profile: Profile = Field(default_factory=Profile)
    memory: Memory = Field(default_factory=Memory)
    story_flags: StoryFlags = Field(default_factory=StoryFlags)
    system_prompt: str | None = None

    @field_validator("profile")
    @classmethod
    def clamp_affection(cls, v: Profile) -> Profile:
        v.affection_score = max(-100, min(100, v.affection_score))
        return v

    def adjust_affection(self, delta: int, damping: float = 0.8) -> int:
        """Apply damped affection change. Returns new score."""
        actual = int(delta * damping)
        self.profile.affection_score = max(
            -100, min(100, self.profile.affection_score + actual)
        )
        return self.profile.affection_score

    def get_relationship_label(self) -> str:
        s = self.profile.affection_score
        if s <= -30:
            return "疏远"
        elif s <= 10:
            return "陌生人"
        elif s <= 40:
            return "朋友"
        elif s <= 70:
            return "亲密"
        else:
            return "恋人"

    def format_system_prompt(self, template: str = "", scene: str = "日常") -> str:
        """Render system prompt from template."""
        prompt = self.system_prompt
        if prompt is None or not prompt.strip():
            prompt = template

        memories = "\n".join(
            self.memory.long_term_summaries + self.memory.important_events[-5:]
        )
        if not memories:
            memories = "（暂无）"

        # Build safe format kwargs, filtering only keys present in prompt
        kwargs = {
            "name": self.profile.name,
            "age": self.profile.age or "?",
            "background": self.profile.background,
            "personality": "、".join(self.profile.personality_traits),
            "speaking_style": self.profile.speaking_style,
            "relationship": self.get_relationship_label(),
            "affection": self.profile.affection_score,
            "memories": memories,
            "scene": scene,
            "char_name": self.profile.name,
            "personality_traits": "、".join(self.profile.personality_traits),
            "scene_description": scene,
        }
        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            # If template uses unknown keys, return prompt as-is
            return prompt

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Character":
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Character":
        data = load_yaml(path)
        return cls.model_validate(data)
