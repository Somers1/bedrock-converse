import asyncio
import base64
import copy
import inspect
import io
import json
import logging
import re
import time
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from dataclasses import field
from datetime import datetime
from functools import cached_property
from typing import Any, List, Dict, Optional, Union, get_origin, get_args, get_type_hints, Callable
from typing import Literal, ByteString
from zoneinfo import ZoneInfo

import boto3
import json5
import json_repair
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel, ValidationError, Field

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PILImage = None
    PIL_AVAILABLE = False

_PIL_WARNING_LOGGED = False

from .tools import tool as agent_tool, Tools
from .bases import BaseCallbackHandler
from .cassette import Cassette

logger = logging.getLogger(__name__)

# AWS Bedrock maximum image dimension
MAX_IMAGE_DIMENSION = 8000


def resize_image_if_needed(image_bytes: bytes, image_format: str) -> bytes:
    """
    Resize image if any dimension exceeds AWS Bedrock's limit of 8000 pixels.
    Returns the original bytes if no resizing is needed or PIL is unavailable.
    """
    global _PIL_WARNING_LOGGED
    if not PIL_AVAILABLE:
        if not _PIL_WARNING_LOGGED:
            logger.warning('PIL not available - skipping image resize. Install pillow to enable automatic image resizing.')
            _PIL_WARNING_LOGGED = True
        return image_bytes

    try:
        img = PILImage.open(io.BytesIO(image_bytes))
        width, height = img.size

        if width <= MAX_IMAGE_DIMENSION and height <= MAX_IMAGE_DIMENSION:
            return image_bytes

        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = MAX_IMAGE_DIMENSION
            new_height = int(height * (MAX_IMAGE_DIMENSION / width))
        else:
            new_height = MAX_IMAGE_DIMENSION
            new_width = int(width * (MAX_IMAGE_DIMENSION / height))

        logger.info(f'Resizing image from {width}x{height} to {new_width}x{new_height}')

        # Resize and save to bytes
        img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

        # Convert format name for PIL
        pil_format = image_format.upper()
        if pil_format == 'JPEG':
            pil_format = 'JPEG'
        elif pil_format == 'JPG':
            pil_format = 'JPEG'

        output = io.BytesIO()
        # Handle RGBA images for formats that don't support alpha
        if img.mode == 'RGBA' and pil_format == 'JPEG':
            img = img.convert('RGB')
        img.save(output, format=pil_format)
        return output.getvalue()
    except Exception as e:
        logger.warning(f'Failed to resize image: {e}')
        return image_bytes


def _to_camel_case(snake_str: str) -> str:
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def _from_camel_case(camel_str: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class InvalidFormat(ValueError):
    pass


@dataclass
class ToolRegistry:
    tools: Dict[str, Callable] = field(default_factory=dict)

    def register(self, tool):
        # Check if it's a Tools class instance
        if hasattr(tool, 'get_tools'):
            # Register all tools from the class
            registered_tools = []
            for class_tool in tool.get_tools():
                tool_name = class_tool._tool_spec['name']
                self.tools[tool_name] = class_tool
                registered_tools.append(class_tool)
            return registered_tools
        elif hasattr(tool, '_tool_spec'):
            tool_name = tool._tool_spec['name']
            self.tools[tool_name] = tool
            return tool
        else:
            raise ValueError(f"Object {tool} is not a valid tool (not decorated with @tool or not a Tools instance)")

    def _resolve_tool_name(self, tool_name: str) -> str:
        if tool_name in self.tools:
            return tool_name
        matches = [k for k in self.tools if k.endswith(f"_{tool_name}")]
        if len(matches) == 1:
            return matches[0]
        return tool_name

    @staticmethod
    def _camel_to_snake(name):
        """Convert a camelCase string to snake_case."""
        import re
        return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    @staticmethod
    def _fix_camel_keys(arguments, known_params):
        """Convert camelCase keys to snake_case, but ONLY when the key doesn't
        match a known parameter and its snake_case version does."""
        fixed = {}
        for key, value in arguments.items():
            if key in known_params:
                fixed[key] = value
            else:
                snake = ToolRegistry._camel_to_snake(key)
                if snake != key and snake in known_params:
                    fixed[snake] = value
                else:
                    fixed[key] = value  # leave unknown keys as-is
        return fixed

    def execute(self, tool_name: str, arguments: dict) -> Any:
        tool_name = self._resolve_tool_name(tool_name)
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        tool = self.tools[tool_name]
        # Auto-validate Pydantic models from type hints (from sortz)
        func = tool._original_function if hasattr(tool, '_original_function') else tool
        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}
        # Get known parameter names from the function signature
        import inspect
        try:
            known_params = set(inspect.signature(func).parameters.keys())
        except (ValueError, TypeError):
            known_params = set(type_hints.keys())
        # Convert camelCase keys to snake_case only when unrecognised and snake version matches
        arguments = self._fix_camel_keys(arguments, known_params)
        validated_args = {}
        for key, value in arguments.items():
            hint = type_hints.get(key)
            if hint and isinstance(value, dict) and hasattr(hint, 'model_validate'):
                validated_args[key] = hint.model_validate(value)
            elif hint and get_origin(hint) is list and value is not None and not isinstance(value, (list, tuple)):
                validated_args[key] = [value]
            else:
                validated_args[key] = value

        if hasattr(tool, '_execute'):
            return tool._execute(**validated_args)
        else:
            return tool(**validated_args)

    def get_tool(self, tool_name: str) -> Optional[Callable]:
        return self.tools.get(tool_name)

    def set_model_switch(self, tool_name: str, switch):
        self.tools[self._resolve_tool_name(tool_name)]._model_switch = switch

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def clear(self):
        self.tools.clear()


class FromDictMixin:
    _FROM_DICT_EXCLUSIONS = []
    _FROM_DICT_SERIALIZATION_EXCLUSIONS = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if data is None:
            return None
        type_hints = get_type_hints(cls)
        kwargs = {}
        data = {_from_camel_case(k): v for k, v in data.items()}
        for field_info in fields(cls):
            field_name = field_info.name
            if field_name in cls._FROM_DICT_EXCLUSIONS:
                continue
            if field_name not in data:
                continue
            value = data[field_name]
            field_type = type_hints.get(field_name, Any)
            if field_name in cls._FROM_DICT_SERIALIZATION_EXCLUSIONS:
                kwargs[field_name] = value
            else:
                kwargs[field_name] = cls._from_convert_value(value, field_type)
        return cls(**kwargs)

    @classmethod
    def _from_convert_value(cls, value: Any, field_type: Any) -> Any:
        if value is None:
            return None
        origin = get_origin(field_type)
        args = get_args(field_type)
        if origin is Union and type(None) in args:
            actual_type = next(arg for arg in args if arg is not type(None))
            return cls._from_convert_value(value, actual_type)
        elif origin is list or origin is List:
            item_type = args[0] if args else Any
            return [cls._from_convert_value(item, item_type) for item in value]
        elif origin is dict or origin is Dict:
            key_type, value_type = args if args else (Any, Any)
            return {cls._from_convert_value(k, key_type): cls._from_convert_value(v, value_type) for k, v in
                    value.items()}
        elif hasattr(field_type, 'from_dict'):
            return field_type.from_dict(value)
        else:
            return value


class ToDictMixin:
    _TO_DICT_EXCLUSIONS = []
    _TO_DICT_SERIALIZATION_EXCLUSIONS = []
    _SKIP_CAMEL_CASE = False  # Override to True to keep snake_case keys

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for class_field in fields(self):
            if class_field.name in self._TO_DICT_EXCLUSIONS:
                continue
            value = getattr(self, class_field.name)
            if value is None:
                continue
            key_name = class_field.name if self._SKIP_CAMEL_CASE else _to_camel_case(class_field.name)
            if class_field.name in self._TO_DICT_SERIALIZATION_EXCLUSIONS:
                result[key_name] = value
            else:
                result[key_name] = self._to_convert_value(value)
        return result

    def _to_convert_value(self, value: Any) -> Any:
        if hasattr(value, 'to_dict') and callable(value.to_dict):
            return value.to_dict()
        elif isinstance(value, list):
            return [self._to_convert_value(item) for item in value]
        elif isinstance(value, dict):
            return {_to_camel_case(k): self._to_convert_value(v) for k, v in value.items()}
        else:
            return value


@dataclass
class S3Location(ToDictMixin, FromDictMixin):
    uri: str
    bucket_owner: Optional[str] = None


@dataclass
class FileSource(ToDictMixin, FromDictMixin):
    bytes: ByteString


@dataclass
class Image(ToDictMixin, FromDictMixin):
    format: Literal["png", "jpeg", "gif", "webp"]
    source: FileSource

    def __post_init__(self):
        valid_formats = typing.get_args(self.__annotations__['format'])
        if self.format == 'jpg':
            self.format = 'jpeg'
        if self.format not in valid_formats:
            raise InvalidFormat(f"Invalid format: {self.format}. Must be one of: {', '.join(valid_formats)}")


@dataclass
class Document(ToDictMixin, FromDictMixin):
    format: Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
    name: str
    source: FileSource

    def __post_init__(self):
        valid_formats = typing.get_args(self.__annotations__['format'])
        if self.format not in valid_formats:
            raise InvalidFormat(f"Invalid format: {self.format}. Must be one of: {', '.join(valid_formats)}")
        self.clean_name()

    def clean_name(self):
        self.name = self.name.encode('ascii', 'ignore').decode('ascii')
        self.name = re.sub(r'[^a-zA-Z0-9\s\-\(\)\[\]]', '', self.name)
        self.name = re.sub(r'\s{2,}', ' ', self.name)
        self.name = self.name.strip()


@dataclass
class VideoSource(ToDictMixin, FromDictMixin):
    bytes: Optional[ByteString] = None
    s3_location: Optional[S3Location] = None


@dataclass
class Video(ToDictMixin, FromDictMixin):
    format: Literal["mkv", "mov", "mp4", "webm", "flv", "mpeg", "mpg", "wmv", "three_gp"]
    source: VideoSource


@dataclass
class ToolUse(FromDictMixin, ToDictMixin):
    _FROM_DICT_SERIALIZATION_EXCLUSIONS = ['input']

    tool_use_id: str
    name: str
    input: Any  # Can be any JSON structure


@dataclass
class ToolResultContent(ToDictMixin, FromDictMixin):
    json: Optional[Any] = None
    text: Optional[str] = None
    image: Optional[Image] = None
    document: Optional[Document] = None
    video: Optional[Video] = None


@dataclass
class ToolResult(ToDictMixin, FromDictMixin):
    tool_use_id: str
    content: List[ToolResultContent]
    status: Literal["success", "error"]


@dataclass
class GuardContentText(ToDictMixin):
    text: str
    qualifiers: List[Literal["grounding_source", "query", "guard_content"]]


@dataclass
class GuardContent(ToDictMixin):
    text: Optional[GuardContentText] = None
    image: Optional[Image] = None


@dataclass
class CachePoint(ToDictMixin):
    type: Literal["default"] = "default"
    ttl: Literal["5m", "1h"] | None = None

    def to_dict(self):
        d = {"type": self.type}
        if self.ttl is not None:
            d["ttl"] = self.ttl
        return d


@dataclass
class ReasoningText(ToDictMixin, FromDictMixin):
    text: str
    signature: Optional[str] = None


@dataclass
class ReasoningContent(ToDictMixin, FromDictMixin):
    _TO_DICT_SERIALIZATION_EXCLUSIONS = ['responses_item']
    _FROM_DICT_SERIALIZATION_EXCLUSIONS = ['responses_item']
    reasoning_text: Optional[ReasoningText] = None
    redacted_content: Optional[ByteString] = None
    responses_item: Optional[Dict[str, Any]] = None


@dataclass
class SystemContent(ToDictMixin, FromDictMixin):
    text: Optional[str] = None
    guard_content: Optional[GuardContent] = None
    cache_point: Optional[CachePoint] = None


@dataclass
class ConverseInferenceConfig(ToDictMixin):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None


@dataclass
class ToolSpec(ToDictMixin):
    _TO_DICT_SERIALIZATION_EXCLUSIONS = ['input_schema']

    name: str
    description: str
    input_schema: Dict[Literal["json"], Any]

    @classmethod
    def from_pydantic(cls, pydantic_model):
        return cls(
            name=pydantic_model.__name__,
            description=f"Output data in the format of {pydantic_model.__name__}",
            input_schema={"json": pydantic_model.model_json_schema()}
        )

    @classmethod
    def from_function(cls, func):
        if hasattr(func, '_tool_spec'):
            spec = func._tool_spec
            return cls(
                name=spec["name"],
                description=spec["description"],
                input_schema=spec["input_schema"]
            )
        else:
            raise ValueError(f"Function {func.__name__} is not decorated with @tool")


@dataclass
class Tool(ToDictMixin):
    tool_spec: Optional[ToolSpec] = None
    cache_point: Optional[CachePoint] = None

    @classmethod
    def from_pydantic(cls, pydantic_model):
        return cls(tool_spec=ToolSpec.from_pydantic(pydantic_model))

    @classmethod
    def from_function(cls, func):
        return cls(tool_spec=ToolSpec.from_function(func))


@dataclass
class ToolChoiceAuto(ToDictMixin):
    pass


@dataclass
class ToolChoiceAny(ToDictMixin):
    pass


@dataclass
class ToolChoiceTool(ToDictMixin):
    name: str


@dataclass
class ToolChoice(ToDictMixin):
    auto: Optional[ToolChoiceAuto] = None
    any: Optional[ToolChoiceAny] = None
    tool: Optional[ToolChoiceTool] = None


@dataclass
class ConverseToolConfig(ToDictMixin):
    tools: List[Tool] = field(default_factory=list)
    tool_choice: Optional[ToolChoice] = None

    def add_cache_point(self, ttl: Literal["5m", "1h"] | None = None):
        self.tools.append(Tool(cache_point=CachePoint(ttl=ttl)))


@dataclass
class ConverseGuardrailConfig(ToDictMixin):
    guardrail_identifier: str
    guardrail_version: str
    trace: Literal["enabled", "disabled", "enabled_full"] = "disabled"


@dataclass
class PromptVariable(ToDictMixin):
    text: str


@dataclass
class ConversePerformanceConfig(ToDictMixin):
    latency: Literal["standard", "optimized"] = "standard"


@dataclass
class ThinkingConfig(ToDictMixin):
    """
    Configuration for Claude's extended thinking/reasoning feature.

    Note: AWS Bedrock API expects snake_case for this config.
    """
    _SKIP_CAMEL_CASE = True  # AWS Bedrock expects snake_case for thinking config

    type: Literal["enabled", "disabled", "adaptive"] = "enabled"
    budget_tokens: Optional[int | str] = None
    display: Optional[Literal["summarized", "omitted"]] = None


@dataclass
class OutputConfig(ToDictMixin):
    """
    Output configuration for adaptive thinking models (Claude 4.5+).

    Note: AWS Bedrock API expects snake_case for this config.
    """
    _SKIP_CAMEL_CASE = True

    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"


@dataclass
class AdditionalModelRequestFields(ToDictMixin):
    """
    Additional model request fields for AWS Bedrock.

    Note: AWS Bedrock expects snake_case for these fields.
    """
    _SKIP_CAMEL_CASE = True  # AWS Bedrock expects snake_case

    thinking: Optional[ThinkingConfig] = None
    output_config: Optional[OutputConfig] = None


@dataclass
class MessageContent(ToDictMixin, FromDictMixin):
    text: Optional[str] = None
    image: Optional[Image] = None
    document: Optional[Document] = None
    video: Optional[Video] = None
    tool_use: Optional[ToolUse] = None
    tool_result: Optional[ToolResult] = None
    guard_content: Optional[GuardContent] = None
    cache_point: Optional[CachePoint] = None
    reasoning_content: Optional[ReasoningContent] = None

    def reduce_size(self):
        if self.text:
            self.text = self.text[:400000].replace('\n', ' ')

    @property
    def is_unsigned_reasoning(self):
        reasoning = self.reasoning_content
        return bool(reasoning and (reasoning.responses_item or
                                   reasoning.reasoning_text and not reasoning.reasoning_text.signature and not reasoning.redacted_content))


@dataclass
class Message(ToDictMixin, FromDictMixin):
    content: List[MessageContent] = field(default_factory=list)
    role: Literal["user", "assistant"] = 'user'

    def add_current_time(self, tz=ZoneInfo('UTC')):
        if isinstance(tz, str):
            tz = ZoneInfo(tz)
        now = datetime.now().astimezone(tz)
        iso = now.isoformat()
        human = now.strftime('%A %d %B %Y at %I:%M %p')
        self.add_text(f'<current_time>{iso} ({human})</current_time>')

    def add_text(self, text, tag=None):
        if text is not None and text.strip('\n').strip():
            if tag:
                text = f'<{tag}>{text}</{tag}>'
            self.content.append(MessageContent(text=text))
        return self

    def add_image(self, source, image_format, skip_on_invalid=False):
        # Resize image if it exceeds AWS Bedrock's 8000 pixel limit
        source = resize_image_if_needed(source, image_format)
        if not skip_on_invalid:
            self.content.append(MessageContent(image=Image(source=FileSource(bytes=source), format=image_format)))
        else:
            try:
                self.content.append(MessageContent(image=Image(source=FileSource(bytes=source), format=image_format)))
            except InvalidFormat as e:
                logger.warning(f'Could not add image to prompt {image_format} is invalid: {e}')
        return self

    def add_cache_point(self, ttl: Literal["5m", "1h"] | None = None):
        self.content.append(MessageContent(cache_point=CachePoint(ttl=ttl)))
        return self

    def add_tool_result(self, tool_use_id, text, status="success"):
        self.content.append(MessageContent(tool_result=ToolResult(
            tool_use_id=tool_use_id, content=[ToolResultContent(text=text)], status=status)))
        return self

    def get_document_names(self):
        return {content.document.name for content in self.content if content.document}

    def add_document(self, source, name, skip_on_invalid=False):
        split_name = name.split('.')
        document_format = split_name[-1].lower()
        name = '_'.join(split_name[:-1])
        try:
            document = Document(format=document_format, name=name, source=FileSource(bytes=source))
        except InvalidFormat as e:
            if not skip_on_invalid:
                raise
            logger.warning(f'Could not add document to prompt {name} is invalid: {e}')
            return self
        if document.name in self.get_document_names():
            document.name += f'_{uuid.uuid4().hex[:6]}'
        self.content.append(MessageContent(document=document))
        return self

    def add_video(self, video):
        raise NotImplementedError

    def reduce_tokens(self):
        for content in self.content:
            if content.text:
                content.text = content.text.replace('\n', ' ').replace('\r', ' ')

    def to_dict(self):
        token = (id(self.content), len(self.content))
        if getattr(self, '_dict_token', None) != token:
            self._dict_cache = super().to_dict()
            self._dict_token = token
        return {**self._dict_cache, 'content': list(self._dict_cache['content'])}


@dataclass
class Prompt(Message):
    pass


@dataclass
class ConverseOutput(FromDictMixin):
    message: Optional[Message] = None


@dataclass
class CacheDetail(FromDictMixin):
    input_tokens: int = 0
    ttl: Literal["5m", "1h"] = "5m"


@dataclass
class TokenUsage(FromDictMixin):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cache_details: List[CacheDetail] = field(default_factory=list)

    def __str__(self):
        return (f"input_tokens: {self.input_tokens}"
                f"\noutput_tokens: {self.output_tokens}"
                f"\ntotal_tokens: {self.total_tokens}"
                f"\ncache_read_input_tokens: {self.cache_read_input_tokens}"
                f"\ncache_write_input_tokens: {self.cache_write_input_tokens}")


@dataclass
class ConverseMetrics(FromDictMixin):
    latency_ms: int = 0


@dataclass
class GuardrailAssessment(FromDictMixin):
    topic_policy: Optional[Dict] = None
    content_policy: Optional[Dict] = None
    word_policy: Optional[Dict] = None
    sensitive_information_policy: Optional[Dict] = None
    contextual_grounding_policy: Optional[Dict] = None
    invocation_metrics: Optional[Dict] = None


@dataclass
class GuardrailTrace(FromDictMixin):
    model_output: Optional[List[str]] = None
    input_assessment: Optional[Dict[str, GuardrailAssessment]] = None
    output_assessments: Optional[Dict[str, List[GuardrailAssessment]]] = None
    action_reason: Optional[str] = None


@dataclass
class PromptRouterTrace(FromDictMixin):
    invoked_model_id: str


@dataclass
class ConverseTrace(FromDictMixin):
    guardrail: Optional[GuardrailTrace] = None
    prompt_router: Optional[PromptRouterTrace] = None


@dataclass
class ModelCost:
    model_name: str
    input: float = 0
    output: float = 0
    cached_write: float = 0
    cached_read: float = 0


@dataclass
class ConverseCost:
    usage: TokenUsage
    model_id: str

    MODELS = [
        ModelCost(model_name='claude-sonnet-4', input=0.003, output=0.015, cached_write=0.00375, cached_read=0.0003),
        ModelCost(model_name='claude-opus-4', input=0.005, output=0.025, cached_write=0.00625, cached_read=0.0005),
        ModelCost(model_name='claude-haiku-4', input=0.001, output=0.005, cached_write=0.00125, cached_read=0.0001),
        ModelCost(model_name='claude-3-7-sonnet', input=0.003, output=0.015, cached_write=0.00375, cached_read=0.0003),
        ModelCost(model_name='claude-3-5-sonnet', input=0.003, output=0.015, cached_write=0.00375, cached_read=0.0003),
        ModelCost(model_name='claude-3-5-haiku', input=0.0008, output=0.004, cached_write=0.001, cached_read=0.00008),
        ModelCost(model_name='claude-haiku-4-5', input=0.001, output=0.005, cached_write=0.00125, cached_read=0.0001),
        ModelCost(model_name='amazon.nova-pro', input=0.0008, output=0.0032),
        ModelCost(model_name='claude-3-haiku', input=0.00025, output=0.00125),
        ModelCost(model_name='amazon.nova-lite', input=0.00006, output=0.00024),
        ModelCost(model_name='gemini-2.0-flash-001', input=0.0001, output=0.0004),
        ModelCost(model_name='llama4-maverick', input=0.00024, output=0.00097),
        ModelCost(model_name='kimi-k2.5', input=0.0006, output=0.003, cached_write=0.0006, cached_read=0.0001),
        ModelCost(model_name='kimi', input=0.0006, output=0.0025, cached_write=0.0006, cached_read=0.0001),
        ModelCost(model_name='glm-5', input=0.001, output=0.0032, cached_write=0.001, cached_read=0.0002),
    ]

    def __str__(self):
        return (f"input_cost: {self.input_cost}"
                f"\noutput_cost: {self.output_cost}"
                f"\ntotal_cost: {self.total_cost}"
                f"\ncached_read_cost: {self.cached_read_cost}"
                f"\ncached_write_cost: {self.cached_write_cost}")

    @cached_property
    def cost(self):
        for model_cost in self.MODELS:
            if model_cost.model_name in self.model_id.lower():
                return model_cost
        return ModelCost(model_name='unknown')

    @property
    def input_cost(self):
        return self.cost.input * self.usage.input_tokens / 1000

    @property
    def output_cost(self):
        return self.cost.output * self.usage.output_tokens / 1000

    @property
    def cached_write_cost(self):
        return self.cost.cached_write * self.usage.cache_write_input_tokens / 1000

    @property
    def cached_read_cost(self):
        return self.cost.cached_read * self.usage.cache_read_input_tokens / 1000

    @property
    def total_cost(self):
        return sum([self.input_cost, self.output_cost, self.cached_write_cost, self.cached_read_cost])


@dataclass
class ConverseResponse(FromDictMixin):
    output: Optional[ConverseOutput] = None
    stop_reason: Optional[str] = None
    truncated: bool = False
    usage: Optional[TokenUsage] = None
    metrics: Optional[ConverseMetrics] = None
    additional_model_response_fields: Optional[Any] = None
    trace: Optional[ConverseTrace] = None
    performance_config: Optional[ConversePerformanceConfig] = None
    response_metadata: Optional[Dict] = None
    model_id = None

    @property
    def content(self):
        return self.output.message.content[-1].text

    @property
    def cost(self):
        return ConverseCost(model_id=self.model_id, usage=self.usage)


class BedrockStreamError(RuntimeError):
    def __init__(self, event_type, event):
        self.event_type = event_type
        self.event = event
        message = event.get("message") or event.get("originalMessage") or str(event)
        super().__init__(f"{event_type}: {message}")


class IncompleteToolUseError(RuntimeError):
    def __init__(self, name, input_text, stop_reason):
        self.name = name
        self.input_text = input_text
        self.stop_reason = stop_reason
        super().__init__(f"incomplete tool input for {name!r} (stop_reason={stop_reason}): {input_text!r}")


class StreamResponseBuilder:
    STREAM_ERROR_EVENTS = {
        "internalServerException",
        "modelStreamErrorException",
        "validationException",
        "throttlingException",
        "serviceUnavailableException",
    }

    def __init__(self):
        self.role = "assistant"
        self.blocks = {}
        self.block_order = []
        self.stop_reason = None
        self.usage = None
        self.metrics = None
        self.trace = None
        self.performance_config = None
        self.additional_model_response_fields = None

    def absorb(self, raw_event):
        if error_type := next((key for key in self.STREAM_ERROR_EVENTS if key in raw_event), None):
            raise BedrockStreamError(error_type, raw_event[error_type])
        if "messageStart" in raw_event:
            self.role = raw_event["messageStart"].get("role", "assistant")
            yield {"type": "message_start", "role": self.role}
        elif "contentBlockStart" in raw_event:
            evt = raw_event["contentBlockStart"]
            idx = evt["contentBlockIndex"]
            start = evt.get("start", {})
            if not start:
                logger.debug("bedrock_stream_empty_content_block_start index=%s", idx)
            if "toolUse" in start:
                tu = start["toolUse"]
                self.blocks[idx] = {"type": "tool_use", "tool_use_id": tu["toolUseId"], "name": tu["name"], "input_text": ""}
                self.block_order.append(idx)
                yield {"type": "content_block_start", "index": idx, "block_type": "tool_use", "tool_use_id": tu["toolUseId"], "name": tu["name"]}
        elif "contentBlockDelta" in raw_event:
            evt = raw_event["contentBlockDelta"]
            idx = evt["contentBlockIndex"]
            delta = evt.get("delta", {})
            if "reasoningContent" in delta:
                reasoning = delta["reasoningContent"]
                logger.info("bedrock_stream_reasoning_delta_raw index=%s chars=%s has_signature=%s", idx, len(reasoning.get("text", "")), bool(reasoning.get("signature")))
            if idx not in self.blocks:
                if "text" in delta:
                    self.blocks[idx] = {"type": "text", "text": ""}
                    self.block_order.append(idx)
                    yield {"type": "content_block_start", "index": idx, "block_type": "text"}
                elif "reasoningContent" in delta:
                    self.blocks[idx] = {"type": "reasoning", "text": "", "signature": None, "redacted_content": None}
                    self.block_order.append(idx)
                    logger.info("bedrock_stream_reasoning_start index=%s", idx)
                    yield {"type": "content_block_start", "index": idx, "block_type": "reasoning"}
                elif "toolUse" in delta:
                    self.blocks[idx] = {"type": "tool_use", "tool_use_id": None, "name": None, "input_text": ""}
                    self.block_order.append(idx)
            block = self.blocks[idx]
            if "text" in delta:
                block["text"] += delta["text"]
                yield {"type": "text_delta", "index": idx, "text": delta["text"]}
            elif "toolUse" in delta:
                fragment = delta["toolUse"].get("input", "")
                block["input_text"] += fragment
                yield {"type": "tool_use_input_delta", "index": idx, "partial_json": fragment}
            elif "reasoningContent" in delta:
                reasoning = delta["reasoningContent"]
                block["text"] += reasoning.get("text", "")
                block["signature"] = reasoning.get("signature", block["signature"])
                block["redacted_content"] = reasoning.get("redactedContent", block["redacted_content"])
                yield {"type": "reasoning_delta", "index": idx, "reasoning": reasoning}
        elif "contentBlockStop" in raw_event:
            idx = raw_event["contentBlockStop"]["contentBlockIndex"]
            yield {"type": "content_block_stop", "index": idx}
        elif "messageStop" in raw_event:
            stop = raw_event["messageStop"]
            self.stop_reason = stop.get("stopReason")
            self.additional_model_response_fields = stop.get("additionalModelResponseFields")
            yield {"type": "message_stop", "stop_reason": self.stop_reason}
        elif "metadata" in raw_event:
            meta = raw_event["metadata"]
            if "usage" in meta:
                self.usage = TokenUsage.from_dict(meta["usage"])
            if "metrics" in meta:
                self.metrics = ConverseMetrics.from_dict(meta["metrics"])
            if "trace" in meta:
                self.trace = ConverseTrace.from_dict(meta["trace"])
            if "performanceConfig" in meta:
                self.performance_config = ConversePerformanceConfig.from_dict(meta["performanceConfig"])
            yield {"type": "metadata", "usage": self.usage, "metrics": self.metrics}

    def build(self):
        contents = []
        truncated = False
        for idx in self.block_order:
            block = self.blocks[idx]
            if block["type"] == "text":
                contents.append(MessageContent(text=block["text"]))
            elif block["type"] == "tool_use":
                try:
                    tool_input = self.tool_input(block)
                except IncompleteToolUseError:
                    if self.stop_reason != "max_tokens":
                        raise
                    truncated = True
                    continue
                contents.append(MessageContent(tool_use=ToolUse(tool_use_id=block["tool_use_id"], name=block["name"], input=tool_input)))
            elif block["type"] == "reasoning":
                reasoning_text = ReasoningText(text=block["text"], signature=block["signature"])
                contents.append(MessageContent(reasoning_content=ReasoningContent(reasoning_text=reasoning_text, redacted_content=block["redacted_content"])))
        message = Message(role=self.role, content=contents)
        return ConverseResponse(output=ConverseOutput(message=message), stop_reason=self.stop_reason, truncated=truncated, usage=self.usage, metrics=self.metrics, trace=self.trace, performance_config=self.performance_config, additional_model_response_fields=self.additional_model_response_fields)

    def tool_input(self, block):
        if not block["input_text"]:
            return {}
        try:
            return json.loads(block["input_text"])
        except json.JSONDecodeError as error:
            raise IncompleteToolUseError(block["name"], block["input_text"], self.stop_reason) from error


@dataclass
class Converse(ToDictMixin, FromDictMixin):
    model_id: str
    messages: List[Message] = field(default_factory=list)
    system: List[SystemContent] = field(default_factory=list)
    inference_config: Optional[ConverseInferenceConfig] = None
    tool_config: Optional[ConverseToolConfig] = None
    guardrail_config: Optional[ConverseGuardrailConfig] = None
    additional_model_request_fields: Optional[AdditionalModelRequestFields] = None
    prompt_variables: Optional[Dict[str, PromptVariable]] = None
    additional_model_response_field_paths: Optional[List[str]] = None
    request_metadata: Optional[Dict[str, str]] = None
    performance_config: Optional[ConversePerformanceConfig] = None
    _client: boto3.client = None
    region_name: str = None
    callbacks: List[BaseCallbackHandler] = field(default_factory=list)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    _async_client: boto3.client = None
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    cache_key: Optional[str] = None
    # Attempts beyond the first when the provider throttles a request (Bedrock ThrottlingException, HTTP 429
    # via mantle). Each retry backs off exponentially (2s, 4s, ... capped at 30s); agent streams emit a
    # rate_limited event before each wait so callers can surface the pause.
    rate_limit_retries: int = 5
    cassette_scope: str = ''
    _TO_DICT_EXCLUSIONS = ['region_name', '_client', 'callbacks', 'aws_access_key_id', 'aws_secret_access_key',
                           '_async_client', 'tool_registry', 'cache_key', 'cassette_scope', 'rate_limit_retries']
    CACHE_SUPPORTED_MODELS = ['claude', 'nova']

    def add_message(self):
        message = Message()
        self.messages.append(message)
        return message

    def as_agent(self):
        return ConverseAgent(
            model_id=self.model_id,
            messages=self.messages,
            system=self.system,
            inference_config=self.inference_config,
            tool_config=self.tool_config,
            guardrail_config=self.guardrail_config,
        )

    @property
    def session(self):
        if self.aws_access_key_id and self.aws_secret_access_key:
            return boto3.Session(
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
        else:
            return boto3.Session(region_name=self.region_name)

    @property
    def bedrock_client(self):
        return self.session.client('bedrock-runtime', config=Config(read_timeout=180))

    @property
    def cassette_key(self):
        return self.model_id.rpartition('.')[2]

    @property
    def client(self):
        if self._client is None:
            self._client = Cassette.wrap(self)
        return self._client

    def _format_invoke_message(self, message):
        if isinstance(message, str):
            message = Message().add_text(message)
        return self.messages + [message]

    def invoke(self, message: Message | str, stream=False):
        if stream:
            return self.stream(self._format_invoke_message(message))
        response = self._get_response(self._format_invoke_message(message))
        return self.format_response(response)

    async def ainvoke(self, message: Message | str):
        response = await self._aget_response(self._format_invoke_message(message))
        return self.format_response(response)

    def format_response(self, response):
        return response

    def add_callback(self, callback):
        self.callbacks.append(callback)
        return self

    def rate_limited(self, error):
        return isinstance(error, ClientError) and error.response['Error']['Code'] in ('ThrottlingException', 'TooManyRequestsException')

    def rate_limit_delay(self, attempt):
        return min(2 ** (attempt + 1), 30)

    def retry_rate_limits(self, request):
        for attempt in range(self.rate_limit_retries + 1):
            try:
                return request()
            except Exception as error:
                if attempt == self.rate_limit_retries or not self.rate_limited(error):
                    raise
                delay = self.rate_limit_delay(attempt)
                logger.warning(f"rate limited, retrying in {delay}s (attempt {attempt + 1}/{self.rate_limit_retries + 1}): {error}")
                time.sleep(delay)

    def _get_response(self, messages=None):
        for callback in self.callbacks:
            try: callback.on_converse_start(self)
            except Exception as e: logger.warning(f"Callback error: {e}")
        payload = self.build_payload(messages)
        try:
            response = ConverseResponse.from_dict(self.retry_rate_limits(lambda: self.client.converse(**payload)))
        except Exception as error:
            for callback in self.callbacks:
                try:
                    if hasattr(callback, 'on_converse_error'):
                        callback.on_converse_error(self, error)
                except Exception as callback_error:
                    logger.warning(f"Callback error: {callback_error}")
            raise
        response.model_id = self.model_id
        for callback in self.callbacks:
            try: callback.on_converse_end(response)
            except Exception as e: logger.warning(f"Callback error: {e}")
        return response

    @property
    def caching_supported(self):
        return any(model in self.model_id.lower() for model in self.CACHE_SUPPORTED_MODELS)

    def build_payload(self, messages):
        self.remove_invalid_caching(messages)
        payload = self.to_dict()
        if messages:
            payload['messages'] = [m.to_dict() for m in messages]
        return payload

    def remove_invalid_caching(self, messages):
        if not self.caching_supported:
            logger.warning(f'Removing caching since {self.model_id} does not support it.')
            for message in self.messages:
                message.content = [content for content in message.content if not content.cache_point]
            if messages:
                for message in messages:
                    message.content = [content for content in message.content if not content.cache_point]
            self.system = [system for system in self.system if not system.cache_point]
            if self.tool_config:
                self.tool_config.tools = [tool for tool in self.tool_config.tools if not tool.cache_point]

    async def _aget_response(self, messages=None):
        for callback in self.callbacks:
            try:
                if hasattr(callback, 'on_converse_start'): callback.on_converse_start(self)
            except Exception as e: logger.warning(f"Callback error: {e}")
        loop = asyncio.get_event_loop()
        payload = self.build_payload(messages)
        try:
            response_dict = await loop.run_in_executor(None, lambda: self.client.converse(**payload))
        except Exception as error:
            for callback in self.callbacks:
                try:
                    if hasattr(callback, 'on_converse_error'):
                        callback.on_converse_error(self, error)
                except Exception as callback_error:
                    logger.warning(f"Callback error: {callback_error}")
            raise
        response = ConverseResponse.from_dict(response_dict)
        response.model_id = self.model_id
        for callback in self.callbacks:
            try:
                if hasattr(callback, 'on_converse_end'): callback.on_converse_end(response)
            except Exception as e: logger.warning(f"Callback error: {e}")
        return response

    def converse(self, message: Message | str = None, stream=False):
        if isinstance(message, str):
            message = Message().add_text(message)
        if message:
            self.messages.append(message)
        if stream:
            return self.stream_converse_response()
        response = self._get_response()
        self.messages.append(response.output.message)
        return self.format_response(response)

    def stream(self, messages=None):
        for callback in self.callbacks:
            try:
                if hasattr(callback, 'on_converse_start'): callback.on_converse_start(self)
            except Exception as e: logger.warning(f"Callback error: {e}")
        payload = self.build_payload(messages)
        try:
            raw = self.client.converse_stream(**payload)
            builder = StreamResponseBuilder()
            for raw_event in raw['stream']:
                for normalized in builder.absorb(raw_event):
                    yield normalized
            response = builder.build()
        except Exception as error:
            for callback in self.callbacks:
                try:
                    if hasattr(callback, 'on_converse_error'): callback.on_converse_error(self, error)
                except Exception as cb_e: logger.warning(f"Callback error: {cb_e}")
            raise
        response.model_id = self.model_id
        for callback in self.callbacks:
            try:
                if hasattr(callback, 'on_converse_end'): callback.on_converse_end(response)
            except Exception as e: logger.warning(f"Callback error: {e}")
        return response

    def stream_converse(self, message: Message | str = None):
        return self.converse(message, stream=True)

    def stream_converse_response(self):
        response = yield from self.stream()
        self.messages.append(response.output.message)
        return self.format_response(response)

    async def aconverse(self, message: Message | str = None):
        if isinstance(message, str):
            message = Message().add_text(message)
        if message:
            self.messages.append(message)
        response = await self._aget_response()
        self.messages.append(response.output.message)
        return self.format_response(response)

    def bind_tools(self, tools: list | Tools):
        if isinstance(tools, Tools):
            self.add_tool(tools)
            return self
        self.tool_config = ConverseToolConfig()
        for tool in tools:
            self.add_tool(tool)
        return self

    @property
    def current_tool_names(self):
        return [tool.tool_spec.name for tool in self.tool_config.tools if tool.tool_spec is not None]

    def add_tool(self, tool):
        if self.tool_config is None:
            self.tool_config = ConverseToolConfig()

        # Check if it's a Tools class instance
        if hasattr(tool, 'get_tools'):
            # Register all tools from the class
            registered_tools = self.tool_registry.register(tool)
            converse_tools = []
            for registered_tool in registered_tools:
                converse_tool = Tool.from_function(registered_tool)
                if converse_tool.tool_spec.name in self.current_tool_names:
                    logger.info(f'{converse_tool.tool_spec.name} already in tool config skipping.')
                else:
                    self.tool_config.tools.append(converse_tool)
                converse_tools.append(converse_tool)
            return converse_tools

        # Handle single tool
        converse_tool = None
        if callable(tool) and hasattr(tool, '_tool_spec'):
            converse_tool = Tool.from_function(tool)
            self.tool_registry.register(tool)
        elif inspect.isclass(tool) and issubclass(tool, BaseModel):
            converse_tool = Tool.from_pydantic(tool)

        if converse_tool is None:
            raise ValueError(
                'Provided tool is not a tool. Please use a pydantic model, the @tool decorator, or a Tools class instance')
        if converse_tool.tool_spec.name in self.current_tool_names:
            logger.info(f'{converse_tool.tool_spec.name} already in tool config skipping.')
        else:
            self.tool_config.tools.append(converse_tool)
        return converse_tool

    def add_system(self, system):
        self.system.append(SystemContent(text=system))
        return self

    def add_system_cache_point(self, ttl: Literal["5m", "1h"] | None = None):
        self.system.append(SystemContent(cache_point=CachePoint(ttl=ttl)))
        return self

    def with_cache_key(self, key):
        self.cache_key = str(key) if key is not None else None
        return self

    def set_tool_choice(self, tool_name):
        if inspect.isclass(tool_name) and issubclass(tool_name, BaseModel):
            tool_name = tool_name.__name__
        self.tool_config.tool_choice = ToolChoice(tool=ToolChoiceTool(name=tool_name))
        return self

    @property
    def thinking_enabled(self) -> bool:
        """Check if extended thinking is currently enabled."""
        return (
            self.additional_model_request_fields is not None
            and self.additional_model_request_fields.thinking is not None
            and self.additional_model_request_fields.thinking.type in ("enabled", "adaptive")
        )

    def with_thinking(self, tokens: int | str = 1024, enabled: bool = True):
        thinking_config = ThinkingConfig(
            type="enabled" if enabled else "disabled",
            budget_tokens=tokens
        )
        if self.additional_model_request_fields is None:
            self.additional_model_request_fields = AdditionalModelRequestFields()
        self.additional_model_request_fields.thinking = thinking_config

        # AWS Bedrock requires temperature=1 and top_p disabled when thinking is enabled
        if enabled:
            if self.inference_config is None:
                self.inference_config = ConverseInferenceConfig()
            self.inference_config.temperature = 1
            self.inference_config.top_p = None
        return self

    def with_adaptive_thinking(self, effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium", display: Literal["summarized", "omitted"] = "summarized"):
        if self.additional_model_request_fields is None:
            self.additional_model_request_fields = AdditionalModelRequestFields()
        self.additional_model_request_fields.thinking = ThinkingConfig(type="adaptive", display=display)
        self.additional_model_request_fields.output_config = OutputConfig(effort=effort)
        if self.inference_config is None:
            self.inference_config = ConverseInferenceConfig()
        self.inference_config.temperature = 1
        self.inference_config.top_p = None
        return self

    @property
    def structured_output_class(self):
        return structured_model_factory(self.model_id)

    def with_structured_output(self, output_model, force_choice=True, skip_add_tool=False, first_tool_only=True):
        assert not (skip_add_tool is True and len(
            self.tool_config.tools) == 0), "If you skip_add_tool you must add tools manually using bind_tools."

        # noinspection PyArgumentList
        return self.structured_output_class(
            model_id=self.model_id,
            messages=self.messages.copy(),
            system=self.system,
            inference_config=self.inference_config,
            tool_config=self.tool_config,
            guardrail_config=self.guardrail_config,
            additional_model_request_fields=self.additional_model_request_fields,
            prompt_variables=self.prompt_variables,
            additional_model_response_field_paths=self.additional_model_response_field_paths,
            request_metadata=self.request_metadata,
            performance_config=self.performance_config,
            region_name=self.region_name,
            callbacks=self.callbacks,
            _client=self._client,
            _async_client=self._async_client,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            output_model=output_model,
            force_choice=force_choice,
            skip_add_tool=skip_add_tool,
            first_tool_only=first_tool_only
        )


@dataclass
class StructuredConverse(Converse):
    output_model: BaseModel = None
    force_choice: bool = True
    skip_add_tool: bool = False
    first_tool_only: bool = True
    backup_model: Optional[Union[str, 'Converse']] = None

    @property
    def cassette_key(self):
        return self.output_model.__name__

    def __post_init__(self):
        super()._TO_DICT_EXCLUSIONS.extend(['output_model', 'force_choice', 'skip_add_tool', 'first_tool_only', 'backup_model'])
        if self.output_model is None:
            raise ValueError(f'Need to specify output_model for StructuredConverse')
        if not self.skip_add_tool:
            self.add_tool(self.output_model)
        if self.force_choice and self.supports_forced_tool_choice:
            self.set_tool_choice(self.output_model.__name__)
        elif self.thinking_enabled and not self.skip_add_tool:
            self.add_system(f'You are in Structured Output mode. You MUST call the {self.output_model.__name__} as your final response.')

    @property
    def supports_forced_tool_choice(self):
        return any(m in self.model_id for m in ('claude', 'kimi')) and not self.thinking_enabled

    def with_backup_model(self, model: Union[str, 'Converse']):
        """
        Set a backup model to use if validation fails.

        Args:
            model: Either a model ID string or a full Converse instance.
                   If a Converse instance is provided, it will be used with
                   the same output_model for structured output.
        """
        self.backup_model = model
        return self

    def invoke(self, message: Message | str, retries=1, _is_backup=False, stream=False):
        if stream:
            return self.stream_invoke(message, retries=retries, _is_backup=_is_backup)
        response = self._get_response(self._format_invoke_message(message))
        try:
            result = self.format_response(response)
            if result is None:
                raise ValueError("No structured output in response")
            return result
        except (ValidationError, ValueError) as e:
            if retries <= 0:
                if self.backup_model and not _is_backup:
                    return self._invoke_backup(message, e)
                raise
            logger.error(e)
            return self.invoke(self.validation_retry_message(message, e, retries), retries=retries - 1, _is_backup=_is_backup)

    def stream_invoke(self, message: Message | str, retries=1, _is_backup=False):
        response = yield from self.stream(self._format_invoke_message(message))
        try:
            result = self.format_response(response)
            if result is None:
                raise ValueError("No structured output in response")
            return result
        except (ValidationError, ValueError) as e:
            if retries <= 0:
                if self.backup_model and not _is_backup:
                    return (yield from self._stream_invoke_backup(message, e))
                raise
            logger.error(e)
            return (yield from self.stream_invoke(self.validation_retry_message(message, e, retries), retries=retries - 1, _is_backup=_is_backup))

    def validation_retry_message(self, message, error, retries):
        if isinstance(message, str):
            message = Message().add_text(message)
        message.add_text(
            f'Your last {self.output_model.__name__} response failed validation. You have {retries} retries left. Please correct the following errors and try again:\n{error}')
        return message

    def _stream_invoke_backup(self, message: Message | str, error):
        if isinstance(self.backup_model, str):
            logger.warning(f"Validation failed on {self.model_id}, falling back to {self.backup_model}")
            original_model_id = self.model_id
            self.model_id = self.backup_model
            try:
                return (yield from self.stream_invoke(message, retries=1, _is_backup=True))
            finally:
                self.model_id = original_model_id
        backup_model_id = self.backup_model.model_id
        logger.warning(f"Validation failed on {self.model_id}, falling back to {backup_model_id}")
        structured_backup = self.backup_model.with_structured_output(
            self.output_model,
            force_choice=self.force_choice,
            skip_add_tool=self.skip_add_tool,
            first_tool_only=self.first_tool_only
        )
        return (yield from structured_backup.stream_invoke(message, retries=1, _is_backup=True))

    def _invoke_backup(self, message: Message | str, error):
        """Handle backup model invocation for both string and Converse instance backup models."""
        if isinstance(self.backup_model, str):
            logger.warning(f"Validation failed on {self.model_id}, falling back to {self.backup_model}")
            original_model_id = self.model_id
            self.model_id = self.backup_model
            try:
                return self.invoke(message, retries=1, _is_backup=True)
            finally:
                self.model_id = original_model_id
        else:
            backup_model_id = self.backup_model.model_id
            logger.warning(f"Validation failed on {self.model_id}, falling back to {backup_model_id}")
            structured_backup = self.backup_model.with_structured_output(
                self.output_model,
                force_choice=self.force_choice,
                skip_add_tool=self.skip_add_tool,
                first_tool_only=self.first_tool_only
            )
            return structured_backup.invoke(message, retries=1, _is_backup=True)

    async def ainvoke(self, message: Message | str, retries=1, _is_backup=False):
        response = await self._aget_response(self._format_invoke_message(message))
        try:
            result = self.format_response(response)
            if result is None:
                raise ValueError("No structured output in response")
            return result
        except (ValidationError, ValueError) as e:
            if retries <= 0:
                if self.backup_model and not _is_backup:
                    return await self._ainvoke_backup(message, e)
                raise
            logger.error(e)
            message.add_text(
                f'Your last response failed validation. You have {retries} retries left. Please correct the following errors and try again:\n{e}')
            return await self.ainvoke(message, retries=retries - 1, _is_backup=_is_backup)

    async def _ainvoke_backup(self, message: Message | str, error):
        """Handle async backup model invocation for both string and Converse instance backup models."""
        if isinstance(self.backup_model, str):
            logger.warning(f"Validation failed on {self.model_id}, falling back to {self.backup_model}")
            original_model_id = self.model_id
            self.model_id = self.backup_model
            try:
                return await self.ainvoke(message, retries=1, _is_backup=True)
            finally:
                self.model_id = original_model_id
        else:
            backup_model_id = self.backup_model.model_id
            logger.warning(f"Validation failed on {self.model_id}, falling back to {backup_model_id}")
            structured_backup = self.backup_model.with_structured_output(
                self.output_model,
                force_choice=self.force_choice,
                skip_add_tool=self.skip_add_tool,
                first_tool_only=self.first_tool_only
            )
            return await structured_backup.ainvoke(message, retries=1, _is_backup=True)

    def format_response(self, response):
        response_objects = []
        response_texts = []
        for content in response.output.message.content:
            if content.tool_use and content.tool_use.name == self.output_model.__name__:
                response_objects.append(self.output_model.model_validate(content.tool_use.input))
            if content.text:
                response_texts.append(content.text)
        if not response_objects:
            response_text = '\n'.join(response_texts)
            logging.error(f"Failed to call structured output. Response text: \n{response_text}")
            return None
        # Return first only
        if self.first_tool_only:
            return response_objects[0]
        return response_objects


@dataclass
class StructuredMaverick(StructuredConverse):
    def __post_init__(self):
        super()._TO_DICT_EXCLUSIONS.extend(['output_model', 'force_choice', 'skip_add_tool', 'first_tool_only', 'backup_model'])
        if self.output_model is None:
            raise ValueError(f'Need to specify output_model for StructuredConverse')
        schema = self.output_model.model_json_schema()
        prompt_addition = f"""

You must respond with valid JSON that matches this schema:
{json.dumps(schema, indent=2)}

CRITICAL RULES:
- Output ONLY valid JSON starting with {{ and ending with }}
- Use double quotes for all strings
- Include all required fields
- Use null for missing optional values
- No comments, no markdown, no explanations"""
        if self.system:
            self.system[0] = SystemContent(text=self.system[0].text + prompt_addition)
        else:
            self.add_system(prompt_addition)

    @staticmethod
    def _extract_json(text):
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{': depth += 1
            elif ch == '}': depth -= 1
            if depth == 0:
                return text[start:i + 1]
        return None

    def format_response(self, response):
        response_text = ''.join(content.text for content in response.output.message.content if content.text)
        json_str = self._extract_json(response_text)
        if not json_str:
            logging.error(f"No JSON found in response: {response_text}")
            return None
        try:
            parsed_data = json.loads(json_str)
            return self.output_model.model_validate(parsed_data)
        except json.JSONDecodeError as e:
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r':\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)', lambda m: ': ' + m.group(1).replace(',', ''),
                              json_str)
            try:
                parsed_data = json5.loads(json_repair.repair_json(json_str))
                return self.output_model.model_validate(parsed_data)
            except Exception:
                logging.error(f"Failed to parse JSON: {e}\nResponse: {response_text}")
                return None


CUSTOM_STRUCTURED_MODELS = {
    'llama4-maverick': StructuredMaverick
}


def structured_model_factory(model_id):
    for model, structured_class in CUSTOM_STRUCTURED_MODELS.items():
        if model in model_id.lower():
            return structured_class
    return StructuredConverse


class Finish(BaseModel):
    """ Return this object when you have completed you task """
    final_response: str


@dataclass
class ConverseAgent(Converse):
    max_iterations: int = 15
    exit_tool: Optional[Tool] = None
    # When no exit_tool, on_text hook, or structured output is set, the loop auto-binds the Finish
    # tool as a default way to end. Set False to rely on the natural text-only exit instead, for chat
    # agents that end by replying in plain text rather than burying a final message in a tool call.
    auto_exit_tool: bool = True
    structured_output: Optional[BaseModel] = None
    debug: bool = False
    _list_wrapped: bool = False  # Track if we wrapped a List type
    _on_text: Optional[callable] = None
    ref_registry: dict = field(default_factory=dict)

    # Suppress text content when the model responds with both text and tool calls in the same turn.
    # Prevents a class of hallucination where the model writes conversational text (questions, recaps)
    # alongside tool calls, then in the next iteration pattern-completes a fake user response to its
    # own text. With this enabled, text is stripped from mixed text+tool responses — the model can
    # only communicate via tools during the loop. Text-only responses still work as the exit signal.
    suppress_text_during_loop: bool = True
    prompt_caching: bool = False
    cache_ttl: str = "5m"

    # When set, tool calls the model emits together in one turn run concurrently on a thread pool
    # instead of one-after-another. Ref resolution, the exit tool, and model-switch tools stay serial
    # on the main thread, so the ref registry and message history are never touched off-thread.
    parallel_tools: bool = False
    # Optional (fn) -> result wrapper applied around each pooled tool body, for callers that need
    # per-thread setup/teardown (e.g. closing a thread-local DB connection). Default runs fn directly.
    tool_thread_hook: Optional[Callable] = None

    # A streamed tool call whose accumulated input fails to parse on a normal stop_reason (a
    # dropped/garbled Bedrock delta frame mid tool_use, despite content_block_stop) is unrepairable —
    # its args would be guessed. The stream is re-requested this many times, emitting a stream_reset
    # event before each retry. A max_tokens truncation is NOT corruption — it routes to continuation.
    stream_retries: int = 2

    # When a turn stops on max_tokens (the output cap reached mid-answer or mid tool_use) any unparseable
    # trailing tool_use is dropped and the partial turn is kept. This bounds how many times the loop then
    # auto-continues the turn (asking the model to pick up where it left off). At 0 (the default) the loop
    # does not auto-continue: it emits a continuation_required event and ends, leaving a valid partial turn
    # the caller can continue manually. >0 enables bounded seamless auto-continuation.
    max_continuations: int = 0

    # When set, called as (tool_name, content) for each successful tool result, where content is the
    # List[ToolResultContent] about to be sent to the model. Returns replacement content to substitute
    # (e.g. an oversize result offloaded to a file and replaced by a short pointer) or None to keep as-is.
    # Runs in-loop before the result is sent or persisted, so the substituted content is what gets cached.
    tool_result_hook: Optional[Callable] = None
    interrupt_exceptions: tuple[type[BaseException], ...] = field(default_factory=tuple)

    def __post_init__(self):
        super()._TO_DICT_EXCLUSIONS.extend(['max_iterations', 'exit_tool', 'auto_exit_tool', 'structured_output', 'debug', '_list_wrapped', '_on_text', 'suppress_text_during_loop', 'ref_registry', 'prompt_caching', 'cache_ttl', 'parallel_tools', 'tool_thread_hook', 'stream_retries', 'max_continuations', 'tool_result_hook', 'interrupt_exceptions'])

    CONTINUATION_PROMPT = "Your previous message was cut off because it reached the output token limit. Continue exactly where you left off — do not repeat anything you already wrote, and if you were in the middle of a tool call, re-issue it in full."

    def with_prompt_caching(self, ttl="5m"):
        self.prompt_caching = True
        self.cache_ttl = ttl
        return self

    def build_payload(self, messages):
        payload = super().build_payload(messages)
        if self.prompt_caching and self.caching_supported:
            self.cache_rolling_messages(payload.get('messages') or [])
        return payload

    def cache_rolling_messages(self, messages):
        if messages and not any('cachePoint' in content for content in messages[-1]['content']):
            messages[-1]['content'].append(self.cache_block(self.cache_ttl))

    def cache_block(self, ttl):
        return MessageContent(cache_point=CachePoint(ttl=ttl)).to_dict()

    def on_text(self, hook: callable):
        """Register a hook called when the agent responds with text instead of tools.
        The hook receives the text string. If it returns a value, that becomes the
        agent's return value and the loop ends. If it returns None, the loop continues."""
        self._on_text = hook
        return self

    def on_tool_result(self, hook: callable):
        """Register a hook called with (tool_name, content) for each successful tool result before it is
        sent to the model, where content is the List[ToolResultContent]. Return replacement content to
        substitute (e.g. an oversize result offloaded to a file), or None to keep the original."""
        self.tool_result_hook = hook
        return self

    def interrupt_on(self, *exceptions):
        self.interrupt_exceptions = tuple(exceptions)
        return self

    def bind_exit_tool(self, tool):
        """Bind an exit tool. If tool is a string, looks up an already-bound tool by name suffix
        (e.g. 'send_message' matches 'ChatTools_send_message'). Otherwise adds and binds it."""
        if isinstance(tool, str):
            for ct in (self.tool_config.tools if self.tool_config else []):
                if ct.tool_spec.name.endswith(f'_{tool}') or ct.tool_spec.name == tool:
                    logger.info(f'Found and bound tool {ct.tool_spec.name} as agent exit tool.')
                    self.exit_tool = ct
                    self._annotate_exit_tool()
                    return self
            raise ValueError(f"No bound tool matching '{tool}' found in current tools")
        self.exit_tool = self.add_tool(tool)
        self._annotate_exit_tool()
        return self

    def _annotate_exit_tool(self):
        """Append exit tool hint to the tool's description so the agent knows calling it ends the loop."""
        if not self.exit_tool:
            return
        spec = self.exit_tool.tool_spec
        hint = " [EXIT TOOL: Calling this tool ends the current agent loop. Complete all other work before calling it.]"
        if hint not in spec.description:
            spec.description += hint

    def with_structured_output(self, base_model, **kwargs):
        from typing import get_origin, get_args
        from pydantic import Field, create_model

        self._list_wrapped = False
        origin = get_origin(base_model)

        # Handle List[BaseModel] types
        if origin is list:
            args = get_args(base_model)
            if args:
                inner_type = args[0]
                # Create a wrapper model that contains the list
                wrapper_model = create_model(
                    f'{inner_type.__name__}List',
                    items=(list[inner_type], Field(description=f'List of {inner_type.__name__} items')),
                    __base__=BaseModel
                )
                self._list_wrapped = True
                base_model = wrapper_model

        self.bind_exit_tool(base_model)
        self.structured_output = base_model

    def unbind_structured_output(self):
        self.exit_tool = None
        self.structured_output = None
        self._list_wrapped = False

    def _fire_run_end(self, result):
        for cb in self.callbacks:
            if hasattr(cb, 'on_run_end'):
                cb.on_run_end(self, result)
        return result

    def execute_model_switch(self, switch, tool_use):
        target = next(t for t in self.tool_config.tools if t.tool_spec and t.tool_spec.name == tool_use.name)
        saved = self.snapshot_config()
        self.apply_converse_config(switch.converse)
        self.tool_config = ConverseToolConfig(tools=[target])
        self.messages.append(Message(role="user").add_tool_result(tool_use.tool_use_id, switch.message, status="error"))
        try:
            response = self._get_response()
        finally:
            self.restore_config(saved)
            self.messages.pop()
        return self.rewrite_tool_input(response, tool_use)

    def rewrite_tool_input(self, response, tool_use):
        new_input = tool_use.input
        for c in (response.output.message.content if response.output.message else []):
            if c.tool_use and c.tool_use.name == tool_use.name:
                new_input = c.tool_use.input
                break
        for c in self.messages[-1].content:
            if c.tool_use and c.tool_use.tool_use_id == tool_use.tool_use_id:
                c.tool_use.input = new_input
                break
        return new_input

    def snapshot_config(self):
        return (self.model_id, self.tool_config, self.region_name, self._client, self.inference_config,
                self.additional_model_request_fields, self.guardrail_config, self.performance_config)

    def restore_config(self, saved):
        (self.model_id, self.tool_config, self.region_name, self._client, self.inference_config,
         self.additional_model_request_fields, self.guardrail_config, self.performance_config) = saved

    def apply_converse_config(self, converse):
        self.model_id = converse.model_id
        if converse.region_name:
            self.region_name, self._client = converse.region_name, None
        if converse.inference_config:
            self.inference_config = converse.inference_config
        if converse.additional_model_request_fields:
            self.additional_model_request_fields = converse.additional_model_request_fields
        if converse.guardrail_config:
            self.guardrail_config = converse.guardrail_config
        if converse.performance_config:
            self.performance_config = converse.performance_config

    REF_PATTERN = re.compile(r'\[ref:([\w-]+)=([^\]]+)\]')

    def resolve_refs(self, tool_input):
        resolved = {}
        for key, value in tool_input.items():
            if key.endswith('_ref') and isinstance(value, str):
                if value in self.ref_registry:
                    id_key = key[:-4] + '_id'
                    resolved[id_key] = self.ref_registry[value]
                else:
                    resolved[key] = value
            elif isinstance(value, str):
                resolved[key] = self.resolve_message_refs(value)
            else:
                resolved[key] = value
        return resolved

    def extract_refs(self, result_str):
        clean = self.REF_PATTERN.sub('', result_str).rstrip()
        clean = re.sub(r' {2,}', ' ', clean)
        for match in self.REF_PATTERN.finditer(result_str):
            self.ref_registry[match.group(1)] = match.group(2)
        return clean

    def run(self, message: Message | str = None, max_iterations=None, first_tool_only=True, stream=False):
        if stream:
            return self.stream_run(message, max_iterations=max_iterations, first_tool_only=first_tool_only)
        loop = self.run_loop(message, max_iterations=max_iterations, first_tool_only=first_tool_only, streaming=False)
        try:
            while True:
                next(loop)
        except StopIteration as stop:
            return stop.value

    def streamed_response(self):
        corrupt = limited = 0
        while True:
            try:
                return (yield from self.stream())
            except IncompleteToolUseError as error:
                corrupt += 1
                if corrupt > self.stream_retries:
                    raise
                logger.warning(f"corrupt tool input, re-requesting stream (attempt {corrupt}/{self.stream_retries + 1}): {error}")
                yield {"type": "stream_reset", "reason": "corrupt_tool_input"}
            except Exception as error:
                if not self.rate_limited(error):
                    raise
                limited += 1
                if limited > self.rate_limit_retries:
                    raise
                delay = self.rate_limit_delay(limited - 1)
                logger.warning(f"rate limited, re-requesting stream in {delay}s (attempt {limited}/{self.rate_limit_retries + 1}): {error}")
                yield {"type": "rate_limited", "attempt": limited, "retries": self.rate_limit_retries, "delay": delay}
                time.sleep(delay)

    def continue_capped_turn(self):
        self.prune_dangling_reasoning()
        if self.messages[-1].content:
            self.messages.append(Message(role="user").add_text(self.CONTINUATION_PROMPT))
        else:
            self.messages.pop()

    def prune_dangling_reasoning(self):
        self.messages[-1].content = [c for c in self.messages[-1].content if not c.is_unsigned_reasoning]

    def run_loop(self, message=None, max_iterations=None, first_tool_only=True, streaming=False):
        max_iterations = max_iterations or self.max_iterations
        self.ref_registry = {}
        if self.structured_output:
            self.bind_exit_tool(self.structured_output)
        elif self.exit_tool is None and not self._on_text and self.auto_exit_tool:
            self.bind_exit_tool(Finish)
        if isinstance(message, str):
            message = Message().add_text(message)
        if message:
            self.messages.append(message)
        for cb in self.callbacks:
            if hasattr(cb, 'on_run_start'):
                cb.on_run_start(self)
        continuations = 0
        for iteration in range(max_iterations):
            yield {"type": "iteration_start", "iteration": iteration}
            try:
                response = (yield from self.streamed_response()) if streaming else self._get_response()
            except IncompleteToolUseError as error:
                logger.error(f"stream returned corrupt tool input after {self.stream_retries} retries: {error}")
                result = "The response was interrupted before it completed. Please try again."
                yield {"type": "done", "result": result, "corrupt_tool_input": True}
                return self._fire_run_end(result)
            capped = response.stop_reason == "max_tokens"
            if not response.output.message.content:
                if capped and continuations < self.max_continuations:
                    continuations += 1
                    yield {"type": "max_tokens_continue", "continuation": continuations}
                    continue
                if capped:
                    yield {"type": "continuation_required"}
                    yield {"type": "done", "result": None, "truncated": True}
                    return self._fire_run_end(None)
                last_content_text = self.messages[-1].content[-1].text
                logger.error(last_content_text)
                yield {"type": "done", "result": last_content_text}
                return self._fire_run_end(last_content_text)
            has_tools = any(c.tool_use for c in response.output.message.content)
            if self.suppress_text_during_loop and has_tools:
                response.output.message.content = [c for c in response.output.message.content if not c.text or c.tool_use]
            self.messages.append(response.output.message)
            tool_results, exit_tool_results = yield from self.execute_tool_uses(response.output.message.content)
            if not tool_results:
                if capped and continuations < self.max_continuations:
                    continuations += 1
                    self.continue_capped_turn()
                    yield {"type": "max_tokens_continue", "continuation": continuations}
                    continue
                if capped:
                    self.prune_dangling_reasoning()
                    yield {"type": "continuation_required"}
                # Returning text ends the loop; a trailing assistant message would trigger
                # "must end with user message" on the next API call.
                text_parts = [c.text for c in response.output.message.content if c.text]
                if text_parts and self._on_text:
                    text = self.resolve_message_refs('\n'.join(text_parts))
                    if (on_text_result := self._on_text(text)) is not None:
                        yield {"type": "done", "result": on_text_result}
                        return self._fire_run_end(on_text_result)
                text = self.resolve_message_refs('\n'.join(text_parts)) if text_parts else None
                yield {"type": "done", "result": text}
                return self._fire_run_end(text)
            continuations = 0
            tool_message = Message(role="user")
            for result in tool_results:
                tool_message.content.append(MessageContent(tool_result=result))
            self.messages.append(tool_message)
            if exit_tool_results:
                if any(r.status == "error" for r in tool_results):
                    logger.warning("Exit tool called but other tools errored — looping back for retry")
                    continue
                result = self.finalize_exit(exit_tool_results, first_tool_only)
                yield {"type": "done", "result": result}
                return self._fire_run_end(result)
        result = f"Agent reached maximum iterations ({max_iterations}) without calling exit tool"
        yield {"type": "done", "result": result, "max_iterations_reached": True}
        return self._fire_run_end(result)

    def execute_tool_uses(self, contents):
        tool_uses = [content.tool_use for content in contents if content.tool_use]
        if self.parallel_tools and len(tool_uses) > 1:
            return (yield from self.execute_tool_uses_parallel(tool_uses))
        return (yield from self.execute_tool_uses_serial(tool_uses))

    def execute_tool_uses_serial(self, tool_uses):
        tool_results = []
        exit_tool_results = []
        for tool_use in tool_uses:
            tool_input = yield from self.announce_tool_call(tool_use)
            outcome = self.run_one_tool(tool_use, tool_input)
            yield from self.collect_outcome(tool_use, outcome, tool_results, exit_tool_results)
        return tool_results, exit_tool_results

    def execute_tool_uses_parallel(self, tool_uses):
        calls = []
        for tool_use in tool_uses:
            tool_input = yield from self.announce_tool_call(tool_use)
            calls.append((tool_use, tool_input))
        pooled = [(tool_use, tool_input) for tool_use, tool_input in calls if self.is_parallel_safe(tool_use)]
        serial = [(tool_use, tool_input) for tool_use, tool_input in calls if not self.is_parallel_safe(tool_use)]
        outcomes = {}
        with ThreadPoolExecutor(max_workers=len(pooled) or 1) as pool:
            futures = {pool.submit(self.run_one_tool_threaded, tool_use, tool_input): tool_use.tool_use_id for tool_use, tool_input in pooled}
            for tool_use, tool_input in serial:
                outcomes[tool_use.tool_use_id] = self.run_one_tool(tool_use, tool_input)
            for future in as_completed(futures):
                outcomes[futures[future]] = future.result()
        tool_results = []
        exit_tool_results = []
        for tool_use, tool_input in calls:
            yield from self.collect_outcome(tool_use, outcomes[tool_use.tool_use_id], tool_results, exit_tool_results)
        return tool_results, exit_tool_results

    def announce_tool_call(self, tool_use):
        tool_input = self.resolve_refs(tool_use.input)
        yield {"type": "tool_call", "tool_use_id": tool_use.tool_use_id, "name": tool_use.name, "input": tool_input}
        if self.debug:
            logger.warning(f'Called {tool_use.name} for {tool_input}')
        for cb in self.callbacks:
            if hasattr(cb, 'on_tool_start'):
                cb.on_tool_start(tool_use.name, tool_input, tool_use.tool_use_id)
        return tool_input

    def collect_outcome(self, tool_use, outcome, tool_results, exit_tool_results):
        result, exc, elapsed, used_input, is_exit = outcome
        tool_result, event = self.build_tool_result(tool_use, result, exc)
        tool_results.append(tool_result)
        yield event
        if is_exit and exc is None:
            exit_tool_results.append(result)
        self.fire_tool_end(tool_use.name, used_input, tool_use.tool_use_id, result if exc is None else str(exc), "error" if exc else "success", elapsed)

    def is_parallel_safe(self, tool_use):
        if self.is_exit_tool(tool_use.name):
            return False
        return getattr(self.tool_registry.get_tool(tool_use.name), '_model_switch', None) is None

    def run_one_tool(self, tool_use, tool_input):
        start = time.time()
        try:
            if self.is_exit_tool(tool_use.name):
                return self.invoke_exit_tool(tool_use.name, tool_input), None, time.time() - start, tool_input, True
            tool = self.tool_registry.get_tool(tool_use.name)
            if switch := getattr(tool, '_model_switch', None):
                tool_input = self.execute_model_switch(switch, tool_use)
            return self.tool_registry.execute(tool_use.name, tool_input), None, time.time() - start, tool_input, False
        except Exception as e:
            if isinstance(e, self.interrupt_exceptions):
                raise
            return None, e, time.time() - start, tool_input, False

    def run_one_tool_threaded(self, tool_use, tool_input):
        if self.tool_thread_hook:
            return self.tool_thread_hook(lambda: self.run_one_tool(tool_use, tool_input))
        return self.run_one_tool(tool_use, tool_input)

    def build_tool_result(self, tool_use, result, exc):
        if exc is None:
            content = self.as_tool_content(result)
            if self.tool_result_hook:
                content = self.tool_result_hook(tool_use.name, content) or content
            summary = '\n'.join(item.text if item.text is not None else self.content_label(item) for item in content)
            tool_result = ToolResult(tool_use_id=tool_use.tool_use_id, content=content, status="success")
            return tool_result, {"type": "tool_result", "tool_use_id": tool_use.tool_use_id, "name": tool_use.name, "result": summary, "status": "success"}
        logger.error(f'Failed to call tool {exc}', exc_info=exc)
        tool_result = ToolResult(tool_use_id=tool_use.tool_use_id, content=[ToolResultContent(text=str(exc))], status="error")
        return tool_result, {"type": "tool_result", "tool_use_id": tool_use.tool_use_id, "name": tool_use.name, "result": str(exc), "status": "error"}

    def as_tool_content(self, result):
        if isinstance(result, ToolResultContent):
            return [result]
        if isinstance(result, list) and result and all(isinstance(item, ToolResultContent) for item in result):
            return result
        if isinstance(result, (dict, list)):
            return [ToolResultContent(text=self.extract_refs(json.dumps(result, default=str)))]
        return [ToolResultContent(text=self.extract_refs(str(result)))]

    def content_label(self, item):
        if item.image:
            return f'[image/{item.image.format}]'
        if item.document:
            return f'[document: {item.document.name}.{item.document.format}]'
        return json.dumps(item.json) if item.json is not None else ''

    def is_exit_tool(self, tool_name):
        return bool(self.exit_tool) and tool_name == self.exit_tool.tool_spec.name

    def invoke_exit_tool(self, tool_name, tool_input):
        if self.structured_output:
            return self.structured_output.model_validate(tool_input)
        if tool_name == 'Finish':
            return Finish.model_validate(tool_input).final_response
        return self.tool_registry.execute(tool_name, tool_input)

    def fire_tool_end(self, tool_name, tool_input, tool_use_id, result, status, elapsed):
        for cb in self.callbacks:
            if hasattr(cb, 'on_tool_end'):
                cb.on_tool_end(tool_name, tool_input, tool_use_id, result, status, elapsed)

    def finalize_exit(self, exit_tool_results, first_tool_only):
        if first_tool_only:
            result = exit_tool_results[0]
            return result.items if self._list_wrapped and hasattr(result, 'items') else result
        if self._list_wrapped:
            return [r.items if hasattr(r, 'items') else r for r in exit_tool_results]
        return exit_tool_results

    def resolve_message_refs(self, text):
        def replace_ref(match):
            prefix, ref_key = match.group(1), match.group(2)
            if ref_key in self.ref_registry:
                return f'{prefix}:{self.ref_registry[ref_key]}'
            return match.group(0)
        return re.sub(r'(\w+):([\w-]+)', replace_ref, text)

    def stream_run(self, message: Message | str = None, max_iterations=None, first_tool_only=True):
        return self.run_loop(message, max_iterations=max_iterations, first_tool_only=first_tool_only, streaming=True)
