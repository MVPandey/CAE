from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolParameterProperty(BaseModel):
    """
    Represents a property in a tool parameter according to OpenAI's schema
    """

    type: str = Field(..., description="The type of the parameter (string, integer, etc)")
    description: str = Field(..., description="Description of what the parameter does")
    enum: list[str] | None = Field(None, description="List of allowed values for enum types")
    minimum: int | None = Field(None, description="Minimum value for numeric types")
    maximum: int | None = Field(None, description="Maximum value for numeric types")
    default: Any | None = Field(None, description="Default value for the parameter")
    items: dict[str, Any] | None = Field(None, description="Schema for array items")
    properties: dict[str, Any] | None = Field(None, description="Nested object properties")
    required: list[str] | None = Field(None, description="Required fields for nested objects")


class ToolFunctionParameters(BaseModel):
    """
    Parameters definition for a tool function
    """

    type: Literal["object"] = "object"
    properties: dict[str, ToolParameterProperty] = Field(..., description="Dictionary of parameter properties")
    required: list[str] | None = Field(None, description="List of required parameter names")


class ToolFunction(BaseModel):
    """
    Function definition within a tool schema
    """

    name: str = Field(..., description="The name of the function to be called")
    description: str = Field(..., description="A description of what the function does")
    parameters: ToolFunctionParameters = Field(..., description="The parameters the function accepts")


class ToolSchema(BaseModel):
    """
    Complete tool schema in the OpenAI format
    https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
    """

    type: Literal["function"] = "function"
    function: ToolFunction = Field(..., description="The function definition")

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and country, eg. San Francisco, USA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use",
                            },
                        },
                        "required": ["location", "format"],
                    },
                },
            }
        },
    )


class AbstractTool(ABC, BaseModel):
    """
    Abstract base class for tools. Any class inheriting from this must implement
    the function classmethod, while tool_schema is provided as a class variable.
    """

    tool_schema: ClassVar[ToolSchema]

    @classmethod
    @abstractmethod
    def tool_function(cls) -> Callable:
        """Return the tool's implementation function"""
        pass


class ToolCallFunction(BaseModel):
    name: str = Field(..., description="Name of the tool function to be called")
    arguments: str = Field(..., description="JSON string of arguments for the function")


class ToolCall(BaseModel):
    id: str = Field(..., description="Unique identifier for this tool call")
    type: Literal["function"] = Field(
        "function",
        description="Type of the tool call (currently only 'function' is supported)",
    )
    function: ToolCallFunction = Field(..., description="Function call details")
