"""Scene-based state machine for story branching."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Scene:
    scene_id: str
    description: str
    prerequisites: set[str] = None
    choices: list["Choice"] = None
    on_enter: Callable | None = None

    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = set()
        if self.choices is None:
            self.choices = []


@dataclass
class Choice:
    choice_id: str
    text: str
    affection_delta: int = 0
    next_scene: str | None = None
    unlocks: list[str] = None
    conditions: set[str] = None

    def __post_init__(self):
        if self.unlocks is None:
            self.unlocks = []
        if self.conditions is None:
            self.conditions = set()


class StoryStateMachine:
    """Simple state machine for visual-novel style branching."""

    def __init__(self):
        self.scenes: dict[str, Scene] = {}
        self.current_scene: Scene | None = None
        self.history: list[str] = []

    def register_scene(self, scene: Scene) -> None:
        self.scenes[scene.scene_id] = scene

    def set_start(self, scene_id: str) -> bool:
        if scene_id in self.scenes:
            self.current_scene = self.scenes[scene_id]
            self.history.append(scene_id)
            return True
        return False

    def available_choices(self, unlocked_flags: set[str]) -> list[Choice]:
        if self.current_scene is None:
            return []
        return [
            c for c in self.current_scene.choices
            if c.conditions is None or c.conditions <= unlocked_flags
        ]

    def choose(self, choice: Choice) -> str | None:
        """Process a choice, return next scene id or None."""
        if choice.next_scene and choice.next_scene in self.scenes:
            self.current_scene = self.scenes[choice.next_scene]
            self.history.append(choice.next_scene)
            return choice.next_scene
        return None
