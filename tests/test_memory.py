"""Tests for memory system."""

from src.core.memory_system import MemorySystem


class TestMemorySystem:
    def test_short_term_window(self):
        mem = MemorySystem(short_term_turns=3)
        for i in range(10):
            mem.add_turn("user", f"msg{i}")

        context = mem.get_short_term_context()
        # Should keep last 3 pairs = 6 messages
        assert len(context) == 6
        assert context[-1]["content"] == "msg9"

    def test_short_term_preserves_order(self):
        mem = MemorySystem(short_term_turns=2)
        mem.add_turn("user", "A")
        mem.add_turn("assistant", "B")
        mem.add_turn("user", "C")

        context = mem.get_short_term_context()
        assert [m["content"] for m in context] == ["A", "B", "C"]

    def test_chroma_unavailable_graceful(self):
        """If chroma not installed, retrieval should return empty list."""
        mem = MemorySystem()
        # Should not raise even without chroma
        assert mem.retrieve_relevant("test") == []
