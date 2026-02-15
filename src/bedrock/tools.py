import inspect
from functools import wraps
from typing import get_type_hints, Any, Callable, Optional
import jsonschema


def tool(func: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None):
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__.strip() if func.__doc__ else f"Function {func.__name__}")
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)

        all_defs = {}
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            param_type = type_hints.get(param_name, Any)
            schema = python_type_to_json_schema(param_type)

            if isinstance(schema, dict) and '$defs' in schema:
                all_defs.update(schema.pop('$defs'))

            if isinstance(schema, dict):
                property_def = schema
            else:
                property_def = {"type": schema}

            if param.default is not param.empty and isinstance(param.default, str):
                property_def["description"] = param.default
            properties[param_name] = property_def
            if param.default is param.empty:
                required.append(param_name)

        input_schema = {
            "type": "object",
            "properties": properties,
            "required": required
        }

        if all_defs:
            input_schema["$defs"] = all_defs

        tool_spec = {
            "name": tool_name,
            "description": tool_description,
            "input_schema": {"json": input_schema}
        }

        if not validate_tool_schema(tool_spec):
            raise ValueError(f"Invalid JSON schema generated for tool {tool_name}")

        func._tool_spec = tool_spec
        func._original_function = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def python_type_to_json_schema(python_type) -> dict:
    from typing import Union, get_origin, get_args
    import types

    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null"
    }

    if python_type is type(None) or python_type is types.NoneType:
        return {"type": "null"}

    if hasattr(python_type, '__bases__'):
        for base in python_type.__mro__:
            if base.__name__ == 'BaseModel' and hasattr(python_type, 'model_json_schema'):
                return python_type.model_json_schema()

    origin = get_origin(python_type)

    if origin is Union:
        args = get_args(python_type)
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            schema = python_type_to_json_schema(non_none_type)
            if isinstance(schema, dict):
                return schema
            return {"type": schema}

    if origin is list:
        args = get_args(python_type)
        if args:
            item_schema = python_type_to_json_schema(args[0])
            return {
                "type": "array",
                "items": item_schema if isinstance(item_schema, dict) else {"type": item_schema}
            }
        return {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    if origin and origin in type_mapping:
        return {"type": type_mapping[origin]}

    if python_type in type_mapping:
        return {"type": type_mapping[python_type]}

    return {"type": "object"}


def validate_tool_schema(tool_spec: dict) -> bool:
    try:
        schema = tool_spec["input_schema"]["json"]
        jsonschema.validators.Draft202012Validator.check_schema(schema)
        return True
    except jsonschema.SchemaError as e:
        print(f"Schema validation failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during schema validation: {e}")
        return False


class Tools:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._tool_methods = {}
        cls._discover_tools()

    @classmethod
    def _discover_tools(cls):
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith('_') or name in ['__init__', '__new__', 'get_tools']:
                continue

            tool_spec = cls._generate_tool_spec(name, method)
            cls._tool_methods[name] = {
                'spec': tool_spec,
                'method': method
            }

    @classmethod
    def _generate_tool_spec(cls, method_name: str, method: Callable) -> dict:
        tool_name = f"{cls.__name__}_{method_name}"
        tool_description = (method.__doc__.strip() if method.__doc__ else f"Method {method_name}")

        type_hints = get_type_hints(method)
        sig = inspect.signature(method)

        all_defs = {}
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            param_type = type_hints.get(param_name, Any)
            schema = python_type_to_json_schema(param_type)

            if isinstance(schema, dict) and '$defs' in schema:
                all_defs.update(schema.pop('$defs'))

            if isinstance(schema, dict):
                property_def = schema
            else:
                property_def = {"type": schema}

            if param.default is not param.empty and isinstance(param.default, str):
                property_def["description"] = param.default
            properties[param_name] = property_def
            if param.default is param.empty:
                required.append(param_name)

        input_schema = {
            "type": "object",
            "properties": properties,
            "required": required
        }

        if all_defs:
            input_schema["$defs"] = all_defs

        tool_spec = {
            "name": tool_name,
            "description": tool_description,
            "input_schema": {"json": input_schema}
        }

        if not validate_tool_schema(tool_spec):
            raise ValueError(f"Invalid JSON schema generated for tool {tool_name}")

        return tool_spec

    def get_tools(self):
        tools = []
        for method_name, tool_info in self._tool_methods.items():
            bound_method = getattr(self, method_name)

            @wraps(bound_method)
            def wrapper(*args, **kwargs):
                return bound_method(*args, **kwargs)

            wrapper._tool_spec = tool_info['spec']
            wrapper._original_function = bound_method
            wrapper._execute = bound_method

            tools.append(wrapper)

        return tools
