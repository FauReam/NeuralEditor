"""Demo script for testing story engine without LLM.

Usage:
    python scripts/demo_no_llm.py --scenes config/scenes/chapter1.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from src.core.character import Character
from src.core.memory_system import MemorySystem
from src.core.state_machine import Scene, StoryStateMachine
from src.core.story_engine import StoryEngine
from src.utils.config_loader import MemoryConfig, StoryConfig
from src.utils.json_storage import JSONStorage
from src.utils.scene_loader import load_scenes

console = Console()


def main():
    char = Character()
    mem = MemorySystem()
    sm = StoryStateMachine()
    config = StoryConfig(auto_save_interval=5)
    storage = JSONStorage("data/saves")

    engine = StoryEngine(char, mem, sm, config, storage)

    # Load scenes if provided
    scenes_path = "config/scenes/chapter1.yaml"
    if Path(scenes_path).exists():
        scenes = load_scenes(scenes_path)
        engine.init_scenes(scenes)
        console.print(f"[dim]已加载 {len(scenes)} 个场景[/dim]")

    char_name = engine.character.profile.name
    affection = engine.character.profile.affection_score
    relationship = engine.character.get_relationship_label()

    console.print(Panel.fit(
        f"[bold magenta]{char_name}[/bold magenta]\n"
        f"关系: {relationship} | 好感度: {affection}/100\n"
        f"[dim]无LLM演示模式 | 输入 /quit 退出[/dim]",
        title="Heartscape Engine Demo",
        border_style="magenta",
    ))

    while True:
        user_input = console.input("[bold cyan]你: [/bold cyan]").strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input[1:].lower()
            if cmd == "quit":
                console.print("[dim]再见。[/dim]")
                break
            elif cmd == "save":
                engine.save()
                console.print("[green]已保存。[/green]")
                continue
            elif cmd == "status":
                console.print(
                    f"好感度: {engine.character.profile.affection_score}, "
                    f"关系: {engine.character.get_relationship_label()}, "
                    f"回合: {engine.turn_count}"
                )
                continue
            continue

        context = engine.process_player_input(user_input)
        scene_desc = context.get("scene", "")
        if scene_desc:
            console.print(f"[dim]当前场景: {scene_desc}[/dim]")

        # In demo mode, just echo with a simulated response
        response = f"*（演示）* 嗯...「{user_input}」，让我想想。"
        engine.record_assistant_response(response)
        console.print(f"[bold magenta]{char_name}:[/bold magenta] {response}")

        # Show scene choices if any
        if engine.state.current_scene:
            choices = engine.state.available_choices(
                set(engine.character.story_flags.unlocked_scenes)
            )
            if choices:
                console.print("[dim]可选动作:[/dim]")
                for c in choices:
                    console.print(f"  [dim]- {c.text}[/dim]")


if __name__ == "__main__":
    main()
