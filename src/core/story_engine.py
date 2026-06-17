"""Main story engine orchestrating character, memory, and state."""

from pathlib import Path
from typing import Any

from src.core.character import Character
from src.core.memory_system import MemorySystem
from src.core.state_machine import Choice, Scene, StoryStateMachine
from src.utils.config_loader import MemoryConfig, StoryConfig
from src.utils.json_storage import JSONStorage


class StoryEngine:
    """Orchestrates character state, memory, and story progression."""

    def __init__(
        self,
        character: Character,
        memory_system: MemorySystem,
        state_machine: StoryStateMachine,
        story_config: StoryConfig,
        storage: JSONStorage | None = None,
    ):
        self.character = character
        self.memory = memory_system
        self.state = state_machine
        self.config = story_config
        self.storage = storage
        self.turn_count = 0

    @classmethod
    def from_defaults(
        cls,
        character_path: str | Path,
        memory_config: MemoryConfig,
        story_config: StoryConfig,
        save_dir: str = "data/saves",
    ) -> "StoryEngine":
        character = Character.from_yaml(character_path)
        memory = MemorySystem(
            embedding_model=memory_config.embedding_model,
            vector_db_path=memory_config.vector_db_path,
            short_term_turns=memory_config.short_term_turns,
            long_term_top_k=memory_config.long_term_top_k,
            similarity_threshold=memory_config.similarity_threshold,
        )
        sm = StoryStateMachine()
        storage = JSONStorage(save_dir)
        return cls(character, memory, sm, story_config, storage)

    def init_scenes(self, scenes: list[Scene]) -> None:
        for scene in scenes:
            self.state.register_scene(scene)
        # Auto-start first unlocked scene
        if scenes:
            flags = set(self.character.story_flags.unlocked_scenes)
            for scene in scenes:
                if scene.prerequisites <= flags:
                    self.state.set_start(scene.scene_id)
                    break

    def process_player_input(self, text: str) -> dict[str, Any]:
        """Main entry: handle player input, return structured result."""
        self.memory.add_turn("user", text)
        self.turn_count += 1

        # Retrieve relevant long-term memories for context
        relevant_memories = self.memory.retrieve_relevant(text)

        # Build prompt context (to be passed to LLM engine)
        context = {
            "character": self.character,
            "short_term": self.memory.get_short_term_context(),
            "long_term": relevant_memories,
            "scene": self.state.current_scene.description if self.state.current_scene else "",
            "turn_count": self.turn_count,
        }

        # Auto-save
        if self.storage and self.turn_count % self.config.auto_save_interval == 0:
            self.save()

        return context

    def record_assistant_response(self, text: str) -> None:
        self.memory.add_turn("assistant", text)

    def apply_choice(self, choice: Choice) -> None:
        """Apply a story choice, updating affection and flags."""
        if choice.affection_delta != 0:
            self.character.adjust_affection(choice.affection_delta)

        for unlock in choice.unlocks:
            if unlock not in self.character.story_flags.unlocked_scenes:
                self.character.story_flags.unlocked_scenes.append(unlock)

        self.state.choose(choice)

    def save(self, slot: str | None = None) -> None:
        if self.storage is None:
            return
        slot = slot or f"{self.character.character_id}_auto"
        data = {
            "character": self.character.to_dict(),
            "turn_count": self.turn_count,
            "history": self.state.history,
            "short_term_memory": self.memory.get_short_term_context(),
        }
        self.storage.save(slot, data)

    def load(self, slot: str) -> bool:
        if self.storage is None:
            return False
        data = self.storage.load(slot)
        if data is None:
            return False
        self.character = Character.from_dict(data["character"])
        self.turn_count = data.get("turn_count", 0)
        self.memory.short_term = data.get("short_term_memory", [])
        return True
