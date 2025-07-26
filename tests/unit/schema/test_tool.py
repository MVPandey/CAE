"""Unit tests for tool schema models."""

from typing import Callable

import pytest
from pydantic import ValidationError

from app.schema.llm.tool import (
    AbstractTool,
    ToolCall,
    ToolCallFunction,
    ToolFunction,
    ToolFunctionParameters,
    ToolParameterProperty,
    ToolSchema,
)


class TestToolParameterProperty:
    """Tests for ToolParameterProperty model."""

    def test_basic_string_property(self):
        """Test creating a basic string property."""
        prop = ToolParameterProperty(type="string", description="The user's name")
        assert prop.type == "string"
        assert prop.description == "The user's name"
        assert prop.enum is None
        assert prop.minimum is None
        assert prop.maximum is None
        assert prop.default is None
        assert prop.items is None
        assert prop.properties is None
        assert prop.required is None

    def test_enum_property(self):
        """Test creating an enum property."""
        prop = ToolParameterProperty(
            type="string", description="Temperature unit", enum=["celsius", "fahrenheit", "kelvin"]
        )
        assert prop.type == "string"
        assert prop.enum == ["celsius", "fahrenheit", "kelvin"]

    def test_numeric_property_with_bounds(self):
        """Test creating a numeric property with min/max."""
        prop = ToolParameterProperty(type="integer", description="Age in years", minimum=0, maximum=150, default=25)
        assert prop.type == "integer"
        assert prop.minimum == 0
        assert prop.maximum == 150
        assert prop.default == 25

    def test_array_property(self):
        """Test creating an array property."""
        prop = ToolParameterProperty(type="array", description="List of tags", items={"type": "string", "minLength": 1})
        assert prop.type == "array"
        assert prop.items == {"type": "string", "minLength": 1}

    def test_nested_object_property(self):
        """Test creating a nested object property."""
        prop = ToolParameterProperty(
            type="object",
            description="Address information",
            properties={
                "street": {"type": "string", "description": "Street address"},
                "city": {"type": "string", "description": "City name"},
                "zipcode": {"type": "string", "description": "ZIP code"},
            },
            required=["street", "city"],
        )
        assert prop.type == "object"
        assert "street" in prop.properties
        assert "city" in prop.properties
        assert prop.required == ["street", "city"]

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ToolParameterProperty()

        errors = exc_info.value.errors()
        field_names = {error["loc"][0] for error in errors}
        assert "type" in field_names
        assert "description" in field_names


class TestToolFunctionParameters:
    """Tests for ToolFunctionParameters model."""

    def test_simple_parameters(self):
        """Test creating simple function parameters."""
        params = ToolFunctionParameters(
            properties={
                "name": ToolParameterProperty(type="string", description="Person's name"),
                "age": ToolParameterProperty(type="integer", description="Person's age", minimum=0),
            },
            required=["name"],
        )

        assert params.type == "object"
        assert "name" in params.properties
        assert "age" in params.properties
        assert params.required == ["name"]

    def test_empty_parameters(self):
        """Test creating empty parameters (no properties)."""
        params = ToolFunctionParameters(properties={})
        assert params.type == "object"
        assert params.properties == {}
        assert params.required is None

    def test_parameters_type_must_be_object(self):
        """Test that type is always 'object'."""
        params = ToolFunctionParameters(properties={"test": ToolParameterProperty(type="string", description="test")})
        assert params.type == "object"


class TestToolFunction:
    """Tests for ToolFunction model."""

    def test_tool_function_creation(self):
        """Test creating a tool function."""
        params = ToolFunctionParameters(
            properties={
                "location": ToolParameterProperty(type="string", description="The city and country"),
                "unit": ToolParameterProperty(
                    type="string", description="Temperature unit", enum=["celsius", "fahrenheit"], default="celsius"
                ),
            },
            required=["location"],
        )

        func = ToolFunction(name="get_weather", description="Get the current weather for a location", parameters=params)

        assert func.name == "get_weather"
        assert func.description == "Get the current weather for a location"
        assert func.parameters == params

    def test_tool_function_missing_fields(self):
        """Test that missing fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ToolFunction(name="test")

        errors = exc_info.value.errors()
        field_names = {error["loc"][0] for error in errors}
        assert "description" in field_names
        assert "parameters" in field_names


class TestToolSchema:
    """Tests for ToolSchema model."""

    def test_tool_schema_creation(self):
        """Test creating a complete tool schema."""
        params = ToolFunctionParameters(
            properties={"query": ToolParameterProperty(type="string", description="Search query")}, required=["query"]
        )

        func = ToolFunction(name="search", description="Search for information", parameters=params)

        schema = ToolSchema(function=func)

        assert schema.type == "function"
        assert schema.function == func
        assert schema.function.name == "search"

    def test_tool_schema_immutability(self):
        """Test that ToolSchema is immutable (frozen)."""
        params = ToolFunctionParameters(properties={"test": ToolParameterProperty(type="string", description="test")})
        func = ToolFunction(name="test", description="test", parameters=params)
        schema = ToolSchema(function=func)

        with pytest.raises(ValidationError, match="Instance is frozen"):
            schema.type = "something_else"

    def test_tool_schema_example(self):
        """Test the example from the schema configuration."""
        params = ToolFunctionParameters(
            properties={
                "location": ToolParameterProperty(
                    type="string", description="The city and country, eg. San Francisco, USA"
                ),
                "format": ToolParameterProperty(
                    type="string", enum=["celsius", "fahrenheit"], description="The temperature unit to use"
                ),
            },
            required=["location", "format"],
        )

        func = ToolFunction(name="get_current_weather", description="Get the current weather", parameters=params)

        schema = ToolSchema(function=func)

        assert schema.function.name == "get_current_weather"
        assert "location" in schema.function.parameters.properties
        assert "format" in schema.function.parameters.properties
        assert schema.function.parameters.required == ["location", "format"]


class TestAbstractTool:
    """Tests for AbstractTool abstract base class."""

    def test_abstract_tool_cannot_be_instantiated(self):
        """Test that AbstractTool cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            AbstractTool()

        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_concrete_tool_implementation(self):
        """Test implementing a concrete tool."""

        class WeatherTool(AbstractTool):
            tool_schema = ToolSchema(
                function=ToolFunction(
                    name="get_weather",
                    description="Get weather information",
                    parameters=ToolFunctionParameters(
                        properties={"location": ToolParameterProperty(type="string", description="Location")},
                        required=["location"],
                    ),
                )
            )

            @classmethod
            def tool_function(cls) -> Callable:
                def get_weather(location: str) -> str:
                    return f"Weather in {location}: Sunny, 72°F"

                return get_weather

        assert WeatherTool.tool_schema.function.name == "get_weather"

        weather_func = WeatherTool.tool_function()
        result = weather_func("San Francisco")
        assert result == "Weather in San Francisco: Sunny, 72°F"

    def test_concrete_tool_missing_implementation(self):
        """Test that concrete tool must implement tool_function."""
        with pytest.raises(TypeError) as exc_info:

            class IncompleteTool(AbstractTool):
                tool_schema = ToolSchema(
                    function=ToolFunction(
                        name="incomplete",
                        description="Incomplete tool",
                        parameters=ToolFunctionParameters(properties={}),
                    )
                )

            IncompleteTool()

        assert "Can't instantiate abstract class" in str(exc_info.value)


class TestToolCall:
    """Tests for ToolCall and ToolCallFunction models."""

    def test_tool_call_function_creation(self):
        """Test creating a ToolCallFunction."""
        func = ToolCallFunction(name="get_weather", arguments='{"location": "San Francisco", "unit": "fahrenheit"}')

        assert func.name == "get_weather"
        assert func.arguments == '{"location": "San Francisco", "unit": "fahrenheit"}'

    def test_tool_call_creation(self):
        """Test creating a complete ToolCall."""
        func = ToolCallFunction(name="search", arguments='{"query": "Python tutorials"}')

        call = ToolCall(id="call_abc123", function=func)

        assert call.id == "call_abc123"
        assert call.type == "function"
        assert call.function.name == "search"
        assert call.function.arguments == '{"query": "Python tutorials"}'

    def test_tool_call_type_default(self):
        """Test that type defaults to 'function'."""
        call = ToolCall(id="test_id", function=ToolCallFunction(name="test", arguments="{}"))
        assert call.type == "function"

    def test_tool_call_missing_fields(self):
        """Test that missing fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ToolCall()

        errors = exc_info.value.errors()
        field_names = {error["loc"][0] for error in errors}
        assert "id" in field_names
        assert "function" in field_names

    def test_tool_call_function_missing_fields(self):
        """Test that ToolCallFunction requires all fields."""
        with pytest.raises(ValidationError) as exc_info:
            ToolCallFunction(name="test")

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "arguments" for error in errors)
