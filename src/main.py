"""CLI entry point for Heartscape Engine."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.character import Character
from src.core.story_engine import StoryEngine
from src.models.llm_engine import LLMEngine
from src.utils.config_loader import load_settings
from src.utils.scene_loader import load_scenes

console = Console()


def build_messages(context: dict, system_prompt: str) -> list[dict[str, str]]:
    """Build message list from story context."""
    messages = []
    short_term = context.get("short_term", [])

    # Combine system prompt with long-term memories
    long_term = context.get("long_term", [])
    if long_term:
        memory_text = "\n".join(long_term)
        system_content = f"{system_prompt}\n\n[相关记忆]\n{memory_text}"
    else:
        system_content = system_prompt

    if system_content.strip():
        messages.append({"role": "system", "content": system_content})

    messages.extend(short_term)
    return messages


@click.command()
@click.option(
    "--character",
    "-c",
    default="config/characters/default.yaml",
    help="角色配置文件路径",
)
@click.option(
    "--settings",
    "-s",
    default="config/settings.yaml",
    help="全局设置文件路径",
)
@click.option(
    "--scenes",
    default=None,
    help="场景定义YAML路径（可选）",
)
@click.option(
    "--save-slot",
    default=None,
    help="加载指定存档槽",
)
@click.option(
    "--demo",
    is_flag=True,
    help="演示模式：不加载LLM，仅测试剧情逻辑",
)
def cli(character: str, settings: str, scenes: str | None, save_slot: str | None, demo: bool) -> None:
    """Heartscape Engine — 本地LLM恋爱模拟"""
    settings_obj = load_settings(settings)

    # Initialize story engine
    engine = StoryEngine.from_defaults(
        character_path=character,
        memory_config=settings_obj.memory,
        story_config=settings_obj.story,
        save_dir=settings_obj.engine.save_dir,
    )

    # Load scenes if provided
    if scenes and Path(scenes).exists():
        scene_list = load_scenes(scenes)
        engine.init_scenes(scene_list)
        console.print(f"[dim]已加载 {len(scene_list)} 个场景[/dim]")

    # Load save if requested
    if save_slot:
        if engine.load(save_slot):
            console.print(f"[green]已加载存档: {save_slot}[/green]")
        else:
            console.print(f"[yellow]存档不存在: {save_slot}[/yellow]")

    # Initialize LLM (skip in demo mode)
    llm = None
    if not demo:
        try:
            llm = LLMEngine(
                model_path=settings_obj.llm.model_path,
                context_length=settings_obj.llm.context_length,
                max_tokens=settings_obj.llm.max_tokens,
                temperature=settings_obj.llm.temperature,
                top_p=settings_obj.llm.top_p,
                repeat_penalty=settings_obj.llm.repeat_penalty,
                lora_path=settings_obj.llm.lora_path,
            )
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            console.print("[yellow]请先从HuggingFace下载模型并放入 models/ 目录[/yellow]")
            console.print("[yellow]或使用 --demo 模式测试剧情逻辑[/yellow]")
            sys.exit(1)
    else:
        console.print("[yellow]演示模式：LLM 推理被模拟[/yellow]")

    # Render header
    char_name = engine.character.profile.name
    affection = engine.character.profile.affection_score
    relationship = engine.character.get_relationship_label()

    console.print(Panel.fit(
        f"[bold magenta]{char_name}[/bold magenta]\n"
        f"关系: {relationship} | 好感度: {affection}/100\n"
        f"[dim]输入 /save, /load, /quit 控制进度[/dim]",
        title="Heartscape Engine",
        border_style="magenta",
    ))

    # Main loop
    while True:
        try:
            user_input = console.input("[bold cyan]你: [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input[1:].lower()
            if cmd == "quit":
                engine.save()
                console.print("[dim]已自动保存，再见。[/dim]")
                break
            elif cmd == "save":
                slot = input("存档名（回车=auto）: ").strip() or None
                engine.save(slot)
                console.print("[green]已保存。[/green]")
                continue
            elif cmd == "load":
                slot = input("存档名: ").strip()
                if engine.load(slot):
                    console.print("[green]加载成功。[/green]")
                else:
                    console.print("[red]存档不存在。[/red]")
                continue
            else:
                console.print(f"[yellow]未知命令: {cmd}[/yellow]")
                continue

        # Process turn
        context = engine.process_player_input(user_input)

        if demo or llm is None:
            # Demo mode: echo with role-play prefix
            response = f"*（演示模式）* 我收到了：「{user_input}」"
            engine.record_assistant_response(response)
            console.print(f"[bold magenta]{char_name}:[/bold magenta] {response}")
            continue

        # Build prompt
        system_prompt = engine.character.format_system_prompt(
            template="",
            scene=context.get("scene", ""),
        )
        messages = build_messages(context, system_prompt)

        # Generate
        with console.status("[dim]思考中...[/dim]", spinner="dots"):
            try:
                response = llm.chat(messages)
            except Exception as e:
                console.print(f"[red]生成错误: {e}[/red]")
                continue

        engine.record_assistant_response(response)

        # Display
        console.print(f"[bold magenta]{char_name}:[/bold magenta] {response}")

        # Show affection change if significant
        new_affection = engine.character.profile.affection_score
        if new_affection != affection:
            delta = new_affection - affection
            sign = "+" if delta > 0 else ""
            console.print(f"[dim]好感度变化: {sign}{delta} → {new_affection}[/dim]")
            affection = new_affection


if __name__ == "__main__":
    cli()
