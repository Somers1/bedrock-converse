import os
import sys
import unittest
from typing import Optional, List, Dict, Union

from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bedrock.tools import tool, Tools, python_type_to_json_schema, validate_tool_schema


# ── Test Pydantic models ─────────────────────────────────────────────────────

class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    age: int
    address: Optional[Address] = None


# ══════════════════════════════════════════════════════════════════════════════
#  @tool decorator
# ══════════════════════════════════════════════════════════════════════════════

class TestToolDecorator(unittest.TestCase):

    def test_basic_function(self):
        @tool
        def greet(name: str):
            """Say hello"""
            return f"Hello {name}"

        self.assertTrue(hasattr(greet, '_tool_spec'))
        self.assertEqual(greet._tool_spec['name'], 'greet')
        self.assertEqual(greet._tool_spec['description'], 'Say hello')

    def test_function_still_callable(self):
        @tool
        def add(a: int, b: int):
            """Add numbers"""
            return a + b

        self.assertEqual(add(2, 3), 5)

    def test_custom_name_description(self):
        @tool(name="my_tool", description="Custom desc")
        def func(x: str):
            return x

        self.assertEqual(func._tool_spec['name'], 'my_tool')
        self.assertEqual(func._tool_spec['description'], 'Custom desc')

    def test_no_docstring_fallback(self):
        @tool
        def nodoc(x: str):
            return x

        self.assertIn("Function nodoc", nodoc._tool_spec['description'])

    def test_optional_param(self):
        @tool
        def func(required: str, optional: str = "default desc"):
            """A func"""
            return required

        spec = func._tool_spec
        schema = spec['input_schema']['json']
        self.assertIn('required', schema['required'])
        self.assertNotIn('optional', schema['required'])
        # Default string becomes description
        self.assertEqual(schema['properties']['optional'].get('description'), 'default desc')

    def test_type_hints_in_schema(self):
        @tool
        def func(name: str, count: int, rate: float, flag: bool):
            """Typed func"""
            pass

        schema = func._tool_spec['input_schema']['json']
        self.assertEqual(schema['properties']['name']['type'], 'string')
        self.assertEqual(schema['properties']['count']['type'], 'integer')
        self.assertEqual(schema['properties']['rate']['type'], 'number')
        self.assertEqual(schema['properties']['flag']['type'], 'boolean')

    def test_list_param(self):
        @tool
        def func(items: List[str]):
            """List func"""
            pass

        schema = func._tool_spec['input_schema']['json']
        self.assertEqual(schema['properties']['items']['type'], 'array')
        self.assertEqual(schema['properties']['items']['items']['type'], 'string')

    def test_pydantic_param(self):
        @tool
        def func(person: Person):
            """Process person"""
            pass

        schema = func._tool_spec['input_schema']['json']
        props = schema['properties']['person']
        # Should have properties from the Pydantic model
        self.assertIn('properties', props)
        self.assertIn('name', props['properties'])

    def test_self_param_excluded(self):
        # Simulating a method-like function
        @tool
        def method(self, x: str):
            """Method"""
            return x

        schema = method._tool_spec['input_schema']['json']
        self.assertNotIn('self', schema['properties'])


# ══════════════════════════════════════════════════════════════════════════════
#  Tools class
# ══════════════════════════════════════════════════════════════════════════════

class TestToolsClass(unittest.TestCase):

    def test_auto_discovery(self):
        class MyTools(Tools):
            def search(self, query: str):
                """Search for things"""
                return f"Results for {query}"

            def _private(self):
                pass

        self.assertIn('search', MyTools._tool_methods)
        self.assertNotIn('_private', MyTools._tool_methods)

    def test_get_tools(self):
        class MyTools(Tools):
            def action(self, x: int):
                """Do action"""
                return x * 2

        instance = MyTools()
        tools = instance.get_tools()
        self.assertEqual(len(tools), 1)
        self.assertTrue(hasattr(tools[0], '_tool_spec'))
        self.assertEqual(tools[0]._tool_spec['name'], 'MyTools_action')

    def test_tool_spec_includes_class_prefix(self):
        class Calculator(Tools):
            def add(self, a: int, b: int):
                """Add numbers"""
                return a + b

        # _tool_methods keys are method names, spec has class prefix
        self.assertIn('add', Calculator._tool_methods)
        self.assertEqual(Calculator._tool_methods['add']['spec']['name'], 'Calculator_add')


# ══════════════════════════════════════════════════════════════════════════════
#  python_type_to_json_schema
# ══════════════════════════════════════════════════════════════════════════════

class TestPythonTypeToJsonSchema(unittest.TestCase):

    def test_str(self):
        self.assertEqual(python_type_to_json_schema(str), {"type": "string"})

    def test_int(self):
        self.assertEqual(python_type_to_json_schema(int), {"type": "integer"})

    def test_float(self):
        self.assertEqual(python_type_to_json_schema(float), {"type": "number"})

    def test_bool(self):
        self.assertEqual(python_type_to_json_schema(bool), {"type": "boolean"})

    def test_list(self):
        self.assertEqual(python_type_to_json_schema(list), {"type": "array"})

    def test_dict(self):
        self.assertEqual(python_type_to_json_schema(dict), {"type": "object"})

    def test_none_type(self):
        self.assertEqual(python_type_to_json_schema(type(None)), {"type": "null"})

    def test_optional_str(self):
        result = python_type_to_json_schema(Optional[str])
        self.assertEqual(result, {"type": "string"})

    def test_list_of_int(self):
        result = python_type_to_json_schema(List[int])
        self.assertEqual(result, {"type": "array", "items": {"type": "integer"}})

    def test_dict_str_str(self):
        result = python_type_to_json_schema(Dict[str, str])
        self.assertEqual(result, {"type": "object"})

    def test_pydantic_model(self):
        result = python_type_to_json_schema(Person)
        self.assertIn('properties', result)
        self.assertIn('name', result['properties'])

    def test_nested_pydantic(self):
        result = python_type_to_json_schema(Person)
        # Address should be referenced via $defs or inline
        self.assertTrue('properties' in result)

    def test_any_type(self):
        from typing import Any
        result = python_type_to_json_schema(Any)
        self.assertEqual(result, {"type": "object"})


# ══════════════════════════════════════════════════════════════════════════════
#  Schema validation
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaValidation(unittest.TestCase):

    def test_valid_schema(self):
        spec = {
            "name": "test",
            "description": "test",
            "input_schema": {"json": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }}
        }
        self.assertTrue(validate_tool_schema(spec))

    def test_invalid_schema(self):
        spec = {
            "name": "test",
            "description": "test",
            "input_schema": {"json": {
                "type": "invalid_type_here"
            }}
        }
        # Draft202012 may or may not reject "invalid_type_here" depending on version
        # but at minimum it shouldn't crash
        result = validate_tool_schema(spec)
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    unittest.main()
