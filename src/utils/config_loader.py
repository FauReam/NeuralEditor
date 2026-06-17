"""YAML/JSON config loader with validation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model_path: str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    context_length: int = 4096
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    lora_path: str | None = None
    use_ollama: bool = False
    ollama_model: str = "qwen2.5:7b"


class MemoryConfig(BaseModel):
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    vector_db_path: str = "data/memories/chroma"
    short_term_turns: int = 10
    long_term_top_k: int = 3
    similarity_threshold: float = 0.75


class AffectionConfig(BaseModel):
    min: int = -100
    max: int = 100
    initial: int = 0
    damping: float = 0.8


class CharacterDefaults(BaseModel):
    default_path: str = "config/characters/default.yaml"
    affection: AffectionConfig = Field(default_factory=AffectionConfig)


class StoryConfig(BaseModel):
    auto_save_interval: int = 5
    max_history_turns: int = 100


class EngineConfig(BaseModel):
    name: str = "Heartscape Engine"
    version: str = "0.1.0"
    save_dir: str = "data/saves"
    memory_dir: str = "data/memories"


class Settings(BaseModel):
    engine: EngineConfig = Field(default_factory=EngineConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    character: CharacterDefaults = Field(default_factory=CharacterDefaults)
    story: StoryConfig = Field(default_factory=StoryConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    data = load_yaml(path)
    return Settings.model_validate(data)
