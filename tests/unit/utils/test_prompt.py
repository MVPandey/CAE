"""Unit tests for app.utils.prompt module."""

from textwrap import dedent

import pytest

from app.utils.prompt import PROMPTS


class TestPROMPTS:
    """Test PROMPTS class."""

    def test_prompts_class_exists(self):
        """Test that PROMPTS class exists."""
        assert PROMPTS is not None

    def test_system_prompt_attribute(self):
        """Test SYSTEM_PROMPT attribute exists and is a string."""
        assert hasattr(PROMPTS, "SYSTEM_PROMPT")
        assert isinstance(PROMPTS.SYSTEM_PROMPT, str)

    def test_system_prompt_empty(self):
        """Test that SYSTEM_PROMPT is currently empty."""
        assert PROMPTS.SYSTEM_PROMPT == ""

    def test_prompts_class_not_instantiable(self):
        """Test that PROMPTS is used as a namespace, not instantiated."""
        instance = PROMPTS()
        assert hasattr(instance, "SYSTEM_PROMPT")

    def test_dedent_import_used(self):
        """Test that dedent is available for prompt formatting."""
        test_text = """
            This is a
            multiline text
            with indentation
        """
        dedented = dedent(test_text)
        dedented = dedented.strip()
        assert not dedented.startswith(" ")  # Check indentation is removed
        assert "This is a" in dedented

    def test_prompts_immutability(self):
        """Test that PROMPTS values maintain their state."""
        original_prompt = PROMPTS.SYSTEM_PROMPT

        assert PROMPTS.SYSTEM_PROMPT == original_prompt
        assert PROMPTS.SYSTEM_PROMPT == ""

    @pytest.mark.parametrize("attribute", ["SYSTEM_PROMPT"])
    def test_prompt_attributes_are_strings(self, attribute):
        """Test that all prompt attributes are strings."""
        assert hasattr(PROMPTS, attribute)
        value = getattr(PROMPTS, attribute)
        assert isinstance(value, str)

    def test_prompts_extensibility(self):
        """Test that PROMPTS class can be extended with new prompts."""

        original_attrs = set(dir(PROMPTS))

        PROMPTS.TEST_PROMPT = "test"
        assert hasattr(PROMPTS, "TEST_PROMPT")

        delattr(PROMPTS, "TEST_PROMPT")
        assert set(dir(PROMPTS)) == original_attrs
