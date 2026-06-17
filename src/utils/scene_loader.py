"""Load scene definitions from YAML files."""

from pathlib import Path
from typing import Any

from src.core.state_machine import Choice, Scene
from src.utils.config_loader import load_yaml


def load_scenes(path: str | Path) -> list[Scene]:
    """Load scene list from a YAML file."""
    data = load_yaml(path)
    raw_scenes = data.get("scenes", [])
    scenes = []
    for rs in raw_scenes:
        choices = []
        for rc in rs.get("choices", []):
            choices.append(Choice(
                choice_id=rc["choice_id"],
                text=rc["text"],
                affection_delta=rc.get("affection_delta", 0),
                next_scene=rc.get("next_scene"),
                unlocks=rc.get("unlocks", []),
                conditions=set(rc.get("conditions", [])),
            ))
        scenes.append(Scene(
            scene_id=rs["scene_id"],
            description=rs["description"],
            prerequisites=set(rs.get("prerequisites", [])),
            choices=choices,
        ))
    return scenes
