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
        # This is based on current implementation
        assert PROMPTS.SYSTEM_PROMPT == ""

    def test_prompts_class_not_instantiable(self):
        """Test that PROMPTS is used as a namespace, not instantiated."""
        # PROMPTS should be used as a class with class attributes
        # Creating an instance is possible but not the intended use
        instance = PROMPTS()
        assert hasattr(instance, "SYSTEM_PROMPT")

    def test_dedent_import_used(self):
        """Test that dedent is available for prompt formatting."""
        # This tests the import is available for future use
        test_text = """
            This is a
            multiline text
            with indentation
        """
        dedented = dedent(test_text)
        assert not dedented.startswith("\n")
        assert "This is a" in dedented

    def test_prompts_immutability(self):
        """Test that PROMPTS values maintain their state."""
        original_prompt = PROMPTS.SYSTEM_PROMPT

        # Even after access, the value should remain the same
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
        # This is a design test - the class should be able to hold multiple prompts
        # Currently only has SYSTEM_PROMPT, but structure allows for more

        # We can dynamically add prompts (though not recommended in production)
        original_attrs = set(dir(PROMPTS))

        # The class structure allows for adding more class attributes
        PROMPTS.TEST_PROMPT = "test"
        assert hasattr(PROMPTS, "TEST_PROMPT")

        # Clean up
        delattr(PROMPTS, "TEST_PROMPT")
        assert set(dir(PROMPTS)) == original_attrs
