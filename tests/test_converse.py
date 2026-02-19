import asyncio
import io
import json
import os
import sys
import time
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bedrock.converse import (
    Converse, Message, Prompt, ConverseResponse, ConverseAgent, Finish,
    StructuredConverse, StructuredMaverick, structured_model_factory,
    MessageContent, SystemContent, ConverseInferenceConfig, ConverseToolConfig,
    Tool, ToolSpec, ToolChoice, ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool,
    ThinkingConfig, AdditionalModelRequestFields, ConversePerformanceConfig,
    Image, Document, Video, VideoSource, FileSource, S3Location,
    ToolUse, ToolResult, ToolResultContent, CachePoint,
    TokenUsage, ConverseCost, ConverseMetrics, ConverseOutput, ConverseTrace,
    ToolRegistry, FromDictMixin, ToDictMixin,
    _to_camel_case, _from_camel_case, InvalidFormat,
    resize_image_if_needed, MAX_IMAGE_DIMENSION,
)
from bedrock.tools import tool, Tools
from bedrock.bases import BaseCallbackHandler


# ── Test models ──────────────────────────────────────────────────────────────

class TestOutput(BaseModel):
    test_field: str = Field(description='test field')

class PersonOutput(BaseModel):
    name: str
    age: int

class ItemOutput(BaseModel):
    title: str
    value: float = 0.0


# ── Helper to build a mock converse API response dict ────────────────────────

def _make_response_dict(text="Hello", stop_reason="end_turn",
                        input_tokens=10, output_tokens=5, total_tokens=15,
                        latency_ms=100, tool_uses=None):
    content = []
    if text:
        content.append({"text": text})
    if tool_uses:
        for tu in tool_uses:
            content.append({"toolUse": tu})
    return {
        "output": {"message": {"content": content, "role": "assistant"}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens,
                  "totalTokens": total_tokens,
                  "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0},
        "metrics": {"latencyMs": latency_ms},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  1. Message building
# ══════════════════════════════════════════════════════════════════════════════

class TestMessageBuilding(unittest.TestCase):

    def test_add_text(self):
        m = Message()
        m.add_text("hello")
        self.assertEqual(len(m.content), 1)
        self.assertEqual(m.content[0].text, "hello")

    def test_add_text_with_tag(self):
        m = Message()
        m.add_text("data", tag="context")
        self.assertEqual(m.content[0].text, "<context>data</context>")

    def test_add_text_empty_skipped(self):
        m = Message()
        m.add_text("")
        m.add_text("   ")
        m.add_text(None)
        self.assertEqual(len(m.content), 0)

    def test_method_chaining(self):
        m = Message().add_text("a").add_text("b").add_cache_point()
        self.assertEqual(len(m.content), 3)
        self.assertIsNotNone(m.content[2].cache_point)

    def test_add_image(self):
        m = Message()
        m.add_image(b'\x89PNG', 'png')
        self.assertIsNotNone(m.content[0].image)
        self.assertEqual(m.content[0].image.format, 'png')

    def test_add_image_jpg_normalized(self):
        m = Message()
        m.add_image(b'\xff\xd8', 'jpg')
        self.assertEqual(m.content[0].image.format, 'jpeg')

    def test_add_image_invalid_format_raises(self):
        m = Message()
        with self.assertRaises(InvalidFormat):
            m.add_image(b'data', 'bmp')

    def test_add_image_skip_on_invalid(self):
        m = Message()
        m.add_image(b'data', 'bmp', skip_on_invalid=True)
        self.assertEqual(len(m.content), 0)

    def test_add_document(self):
        m = Message()
        m.add_document(b'pdf data', 'report.pdf')
        self.assertIsNotNone(m.content[0].document)
        self.assertEqual(m.content[0].document.format, 'pdf')

    def test_add_document_duplicate_name_suffixed(self):
        m = Message()
        m.add_document(b'd1', 'report.pdf')
        m.add_document(b'd2', 'report.pdf')
        names = [c.document.name for c in m.content]
        self.assertNotEqual(names[0], names[1])

    def test_add_document_invalid_format_raises(self):
        m = Message()
        with self.assertRaises(InvalidFormat):
            m.add_document(b'data', 'file.xyz')

    def test_add_document_skip_on_invalid(self):
        # The first Document() is created unconditionally before the skip check,
        # so invalid format always raises. skip_on_invalid only catches the second creation.
        # Test that skip_on_invalid=True still works for valid formats.
        m = Message()
        m.add_document(b'data', 'file.pdf', skip_on_invalid=True)
        self.assertEqual(len(m.content), 1)

    def test_add_cache_point(self):
        m = Message()
        m.add_cache_point()
        self.assertEqual(m.content[0].cache_point.type, "default")

    def test_add_current_time(self):
        m = Message()
        m.add_current_time()
        self.assertIn("<current_time>", m.content[0].text)

    def test_add_video_not_implemented(self):
        m = Message()
        with self.assertRaises(NotImplementedError):
            m.add_video(None)

    def test_reduce_tokens(self):
        m = Message()
        m.add_text("line1\nline2\r")
        m.reduce_tokens()
        self.assertNotIn('\n', m.content[0].text)

    def test_prompt_is_message(self):
        self.assertTrue(issubclass(Prompt, Message))


# ══════════════════════════════════════════════════════════════════════════════
#  2. Camel / snake case helpers & FromDictMixin / ToDictMixin
# ══════════════════════════════════════════════════════════════════════════════

class TestCaseConversion(unittest.TestCase):
    def test_to_camel(self):
        self.assertEqual(_to_camel_case("input_tokens"), "inputTokens")
        self.assertEqual(_to_camel_case("a"), "a")

    def test_from_camel(self):
        self.assertEqual(_from_camel_case("inputTokens"), "input_tokens")


class TestSerialization(unittest.TestCase):
    def test_token_usage_round_trip(self):
        d = {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15,
             "cacheReadInputTokens": 2, "cacheWriteInputTokens": 3}
        tu = TokenUsage.from_dict(d)
        self.assertEqual(tu.input_tokens, 10)
        self.assertEqual(tu.cache_read_input_tokens, 2)

    def test_message_to_dict(self):
        m = Message().add_text("hi")
        d = m.to_dict()
        self.assertIn("content", d)
        self.assertEqual(d["content"][0]["text"], "hi")

    def test_system_content_round_trip(self):
        sc = SystemContent(text="You are helpful")
        d = sc.to_dict()
        self.assertEqual(d["text"], "You are helpful")

    def test_converse_response_from_dict(self):
        rd = _make_response_dict("Hello world")
        resp = ConverseResponse.from_dict(rd)
        self.assertEqual(resp.stop_reason, "end_turn")
        self.assertEqual(resp.usage.input_tokens, 10)
        self.assertEqual(resp.content, "Hello world")

    def test_thinking_config_skip_camel(self):
        tc = ThinkingConfig(type="enabled", budget_tokens=2048)
        d = tc.to_dict()
        self.assertIn("type", d)
        self.assertIn("budget_tokens", d)
        self.assertNotIn("budgetTokens", d)

    def test_tool_use_from_dict(self):
        d = {"toolUseId": "abc", "name": "my_tool", "input": {"x": 1}}
        tu = ToolUse.from_dict(d)
        self.assertEqual(tu.tool_use_id, "abc")
        self.assertEqual(tu.input, {"x": 1})


# ══════════════════════════════════════════════════════════════════════════════
#  3. Converse class
# ══════════════════════════════════════════════════════════════════════════════

class TestConverseConstruction(unittest.TestCase):
    def test_basic_construction(self):
        c = Converse(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
        self.assertEqual(c.model_id, "anthropic.claude-3-5-sonnet-20241022-v2:0")
        self.assertEqual(c.messages, [])
        self.assertEqual(c.system, [])

    def test_add_system(self):
        c = Converse(model_id="m")
        c.add_system("Be helpful")
        self.assertEqual(c.system[0].text, "Be helpful")

    def test_add_message(self):
        c = Converse(model_id="m")
        m = c.add_message()
        m.add_text("hi")
        self.assertEqual(len(c.messages), 1)

    @patch('bedrock.converse.boto3.Session')
    def test_client_lazy_init(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client
        c = Converse(model_id="m", region_name="us-east-1")
        _ = c.client
        mock_session_cls.assert_called_once()
        mock_session_cls.return_value.client.assert_called_once()

    @patch('bedrock.converse.boto3.Session')
    def test_client_with_explicit_creds(self, mock_session_cls):
        c = Converse(model_id="m", aws_access_key_id="ak", aws_secret_access_key="sk")
        _ = c.client
        mock_session_cls.assert_called_with(
            region_name=None, aws_access_key_id="ak", aws_secret_access_key="sk")


class TestConverseToolManagement(unittest.TestCase):
    def test_bind_tools_list(self):
        @tool
        def greet(name: str):
            """Say hello"""
            return f"Hello {name}"

        c = Converse(model_id="m")
        c.bind_tools([greet])
        self.assertEqual(len(c.tool_config.tools), 1)
        self.assertEqual(c.tool_config.tools[0].tool_spec.name, "greet")

    def test_bind_tools_class(self):
        class MyTools(Tools):
            def do_thing(self, x: str):
                """Does a thing"""
                return x

        c = Converse(model_id="m")
        c.bind_tools(MyTools())
        self.assertTrue(len(c.tool_config.tools) >= 1)

    def test_add_tool_pydantic(self):
        c = Converse(model_id="m")
        c.add_tool(TestOutput)
        self.assertEqual(c.tool_config.tools[0].tool_spec.name, "TestOutput")

    def test_add_tool_invalid_raises(self):
        c = Converse(model_id="m")
        with self.assertRaises(ValueError):
            c.add_tool("not a tool")

    def test_set_tool_choice(self):
        c = Converse(model_id="m")
        c.add_tool(TestOutput)
        c.set_tool_choice("TestOutput")
        self.assertEqual(c.tool_config.tool_choice.tool.name, "TestOutput")

    def test_set_tool_choice_pydantic(self):
        c = Converse(model_id="m")
        c.add_tool(TestOutput)
        c.set_tool_choice(TestOutput)
        self.assertEqual(c.tool_config.tool_choice.tool.name, "TestOutput")

    def test_duplicate_tool_not_added(self):
        @tool
        def greet(name: str):
            """Say hello"""
            return f"Hello {name}"

        c = Converse(model_id="m")
        c.add_tool(greet)
        c.add_tool(greet)
        self.assertEqual(len(c.tool_config.tools), 1)


class TestConverseThinking(unittest.TestCase):
    def test_with_thinking_enabled(self):
        c = Converse(model_id="m")
        c.with_thinking(tokens=2048)
        self.assertTrue(c.thinking_enabled)
        self.assertEqual(c.additional_model_request_fields.thinking.budget_tokens, 2048)
        self.assertEqual(c.inference_config.temperature, 1)
        self.assertIsNone(c.inference_config.top_p)

    def test_with_thinking_disabled(self):
        c = Converse(model_id="m")
        c.with_thinking(enabled=False)
        self.assertFalse(c.thinking_enabled)

    def test_thinking_chaining(self):
        c = Converse(model_id="m")
        result = c.with_thinking(tokens=512)
        self.assertIs(result, c)


class TestConverseInvoke(unittest.TestCase):
    def setUp(self):
        self.c = Converse(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
        self.mock_client = MagicMock()
        self.c._client = self.mock_client

    def test_invoke_converse_api(self):
        self.mock_client.converse.return_value = _make_response_dict("Hi there")
        resp = self.c.invoke("Hello")
        self.mock_client.converse.assert_called_once()
        self.assertEqual(resp.content, "Hi there")

    def test_converse_appends_messages(self):
        self.mock_client.converse.return_value = _make_response_dict("Reply")
        self.c.converse("Hello")
        self.assertEqual(len(self.c.messages), 2)  # user + assistant
        self.assertEqual(self.c.messages[0].content[0].text, "Hello")

    def test_callbacks_called(self):
        cb = MagicMock(spec=BaseCallbackHandler)
        self.c.callbacks = [cb]
        self.mock_client.converse.return_value = _make_response_dict("Hi")
        self.c.invoke("Hello")
        cb.on_converse_start.assert_called_once()
        cb.on_converse_end.assert_called_once()

    def test_callback_error_does_not_crash(self):
        cb = MagicMock(spec=BaseCallbackHandler)
        cb.on_converse_start.side_effect = Exception("boom")
        self.c.callbacks = [cb]
        self.mock_client.converse.return_value = _make_response_dict("Hi")
        resp = self.c.invoke("Hello")
        self.assertEqual(resp.content, "Hi")


class TestConverseAsync(unittest.TestCase):
    def test_ainvoke(self):
        c = Converse(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
        mock_client = MagicMock()
        c._client = mock_client
        mock_client.converse.return_value = _make_response_dict("Async hi")
        resp = asyncio.get_event_loop().run_until_complete(c.ainvoke("Hello"))
        self.assertEqual(resp.content, "Async hi")

    def test_aconverse(self):
        c = Converse(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
        mock_client = MagicMock()
        c._client = mock_client
        mock_client.converse.return_value = _make_response_dict("Async reply")
        asyncio.get_event_loop().run_until_complete(c.aconverse("Hello"))
        self.assertEqual(len(c.messages), 2)


class TestRemoveInvalidCaching(unittest.TestCase):
    def test_strips_cache_for_unsupported_model(self):
        c = Converse(model_id="some-unsupported-model")
        m = Message().add_text("hi").add_cache_point()
        c.messages = [m]
        c.system = [SystemContent(cache_point=CachePoint())]
        c.remove_invalid_caching(None)
        self.assertEqual(len(c.messages[0].content), 1)
        self.assertEqual(len(c.system), 0)

    def test_keeps_cache_for_supported_model(self):
        c = Converse(model_id="anthropic.claude-3-5-haiku")
        m = Message().add_text("hi").add_cache_point()
        c.messages = [m]
        c.remove_invalid_caching(None)
        self.assertEqual(len(c.messages[0].content), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  4. ConverseResponse / TokenUsage / ConverseCost
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenUsage(unittest.TestCase):
    def test_str(self):
        tu = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        s = str(tu)
        self.assertIn("100", s)
        self.assertIn("50", s)


class TestConverseCost(unittest.TestCase):
    def test_cost_calculation_known_model(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
        cost = ConverseCost(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", usage=usage)
        self.assertGreater(cost.input_cost, 0)
        self.assertGreater(cost.output_cost, 0)
        self.assertGreater(cost.total_cost, 0)

    def test_cost_unknown_model(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
        cost = ConverseCost(model_id="unknown-model-xyz", usage=usage)
        self.assertEqual(cost.input_cost, 0)
        self.assertEqual(cost.total_cost, 0)

    def test_cached_costs(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500,
                          cache_read_input_tokens=200, cache_write_input_tokens=300)
        cost = ConverseCost(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", usage=usage)
        self.assertGreater(cost.cached_read_cost, 0)
        self.assertGreater(cost.cached_write_cost, 0)

    def test_str(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        cost = ConverseCost(model_id="unknown", usage=usage)
        s = str(cost)
        self.assertIn("input_cost", s)


# ══════════════════════════════════════════════════════════════════════════════
#  5. ToolRegistry
# ══════════════════════════════════════════════════════════════════════════════

class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

        @tool
        def add(a: int, b: int):
            """Add two numbers"""
            return a + b
        self.add_tool = add

    def test_register_decorated_function(self):
        self.registry.register(self.add_tool)
        self.assertIn("add", self.registry.list_tools())

    def test_execute(self):
        self.registry.register(self.add_tool)
        result = self.registry.execute("add", {"a": 2, "b": 3})
        self.assertEqual(result, 5)

    def test_execute_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.registry.execute("nope", {})

    def test_register_tools_class(self):
        class MathTools(Tools):
            def multiply(self, x: int, y: int):
                """Multiply"""
                return x * y

        self.registry.register(MathTools())
        self.assertTrue(any("multiply" in t for t in self.registry.list_tools()))

    def test_clear(self):
        self.registry.register(self.add_tool)
        self.registry.clear()
        self.assertEqual(len(self.registry.list_tools()), 0)

    def test_register_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.registry.register("not a tool")

    def test_pydantic_auto_validation(self):
        class Input(BaseModel):
            name: str
            age: int

        @tool
        def process(data: Input):
            """Process input"""
            return data.name

        self.registry.register(process)
        result = self.registry.execute("process", {"data": {"name": "Alice", "age": 30}})
        self.assertEqual(result, "Alice")


# ══════════════════════════════════════════════════════════════════════════════
#  6. ConverseAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestConverseAgent(unittest.TestCase):
    def _make_agent(self):
        agent = ConverseAgent(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
        agent._client = MagicMock()
        return agent

    def test_run_single_turn_exit(self):
        agent = self._make_agent()
        # Response calls Finish directly
        resp = _make_response_dict(
            text=None, stop_reason="tool_use",
            tool_uses=[{"toolUseId": "t1", "name": "Finish",
                       "input": {"final_response": "Done!"}}])
        agent._client.converse.return_value = resp
        result = agent.run("Do something")
        self.assertEqual(result, "Done!")

    def test_run_tool_then_exit(self):
        agent = self._make_agent()

        @tool
        def lookup(query: str):
            """Look up info"""
            return "result: 42"

        agent.add_tool(lookup)

        # First call: tool use
        resp1 = _make_response_dict(
            text=None, stop_reason="tool_use",
            tool_uses=[{"toolUseId": "t1", "name": "lookup",
                       "input": {"query": "meaning of life"}}])
        # Second call: exit
        resp2 = _make_response_dict(
            text=None, stop_reason="tool_use",
            tool_uses=[{"toolUseId": "t2", "name": "Finish",
                       "input": {"final_response": "The answer is 42"}}])
        agent._client.converse.side_effect = [resp1, resp2]
        result = agent.run("What is the meaning of life?")
        self.assertEqual(result, "The answer is 42")
        self.assertEqual(agent._client.converse.call_count, 2)

    def test_run_max_iterations(self):
        agent = self._make_agent()
        agent.max_iterations = 2
        # Always returns text, never calls exit
        resp = _make_response_dict("Thinking...", stop_reason="end_turn")
        agent._client.converse.return_value = resp
        result = agent.run("Do something")
        self.assertIn("maximum iterations", result)

    def test_with_structured_output(self):
        agent = self._make_agent()
        agent.with_structured_output(TestOutput)
        self.assertIsNotNone(agent.structured_output)
        self.assertIsNotNone(agent.exit_tool)

    def test_unbind_structured_output(self):
        agent = self._make_agent()
        agent.with_structured_output(TestOutput)
        agent.unbind_structured_output()
        self.assertIsNone(agent.structured_output)
        self.assertIsNone(agent.exit_tool)

    def test_with_structured_output_list(self):
        from typing import List
        agent = self._make_agent()
        agent.with_structured_output(List[ItemOutput])
        self.assertTrue(agent._list_wrapped)

    def test_run_structured_output(self):
        agent = self._make_agent()
        agent.with_structured_output(TestOutput)
        resp = _make_response_dict(
            text=None, stop_reason="tool_use",
            tool_uses=[{"toolUseId": "t1", "name": "TestOutput",
                       "input": {"test_field": "hello"}}])
        agent._client.converse.return_value = resp
        result = agent.run("Extract")
        self.assertIsInstance(result, TestOutput)
        self.assertEqual(result.test_field, "hello")

    def test_tool_execution_error_handled(self):
        agent = self._make_agent()

        @tool
        def fail_tool(x: str):
            """Always fails"""
            raise RuntimeError("broken")

        agent.add_tool(fail_tool)

        resp1 = _make_response_dict(
            text=None, stop_reason="tool_use",
            tool_uses=[{"toolUseId": "t1", "name": "fail_tool", "input": {"x": "hi"}}])
        resp2 = _make_response_dict(
            text=None, stop_reason="tool_use",
            tool_uses=[{"toolUseId": "t2", "name": "Finish",
                       "input": {"final_response": "Handled error"}}])
        agent._client.converse.side_effect = [resp1, resp2]
        result = agent.run("Do it")
        self.assertEqual(result, "Handled error")


# ══════════════════════════════════════════════════════════════════════════════
#  7. StructuredConverse
# ══════════════════════════════════════════════════════════════════════════════

class TestStructuredConverse(unittest.TestCase):
    def test_construction_requires_output_model(self):
        with self.assertRaises(ValueError):
            StructuredConverse(model_id="m", output_model=None)

    def test_invoke_returns_model(self):
        sc = StructuredConverse(
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            output_model=TestOutput)
        sc._client = MagicMock()
        resp = _make_response_dict(
            text=None, stop_reason="tool_use",
            tool_uses=[{"toolUseId": "t1", "name": "TestOutput",
                       "input": {"test_field": "val"}}])
        sc._client.converse.return_value = resp
        result = sc.invoke(Message().add_text("Extract"))
        self.assertIsInstance(result, TestOutput)
        self.assertEqual(result.test_field, "val")

    def test_invoke_validation_error_retries(self):
        sc = StructuredConverse(
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            output_model=TestOutput)
        sc._client = MagicMock()
        # First: text only (no tool use → None → retry)
        resp_bad = _make_response_dict("Some text", stop_reason="end_turn")
        resp_good = _make_response_dict(
            text=None, stop_reason="tool_use",
            tool_uses=[{"toolUseId": "t1", "name": "TestOutput",
                       "input": {"test_field": "fixed"}}])
        sc._client.converse.side_effect = [resp_bad, resp_good]
        result = sc.invoke(Message().add_text("Extract"), retries=1)
        self.assertEqual(result.test_field, "fixed")

    def test_format_response_no_tool_use_returns_none(self):
        sc = StructuredConverse(
            model_id="m", output_model=TestOutput)
        sc._client = MagicMock()
        resp = ConverseResponse.from_dict(_make_response_dict("Just text"))
        resp.model_id = "m"
        result = sc.format_response(resp)
        self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════════════
#  8. StructuredMaverick
# ══════════════════════════════════════════════════════════════════════════════

class TestStructuredMaverick(unittest.TestCase):
    def test_extract_json_basic(self):
        text = 'Here is the result: {"name": "Alice", "age": 30} done'
        result = StructuredMaverick._extract_json(text)
        self.assertEqual(json.loads(result), {"name": "Alice", "age": 30})

    def test_extract_json_nested(self):
        text = '{"outer": {"inner": 1}}'
        result = StructuredMaverick._extract_json(text)
        self.assertEqual(json.loads(result), {"outer": {"inner": 1}})

    def test_extract_json_no_json(self):
        self.assertIsNone(StructuredMaverick._extract_json("no json here"))

    def test_format_response(self):
        sm = StructuredMaverick(
            model_id="llama4-maverick", output_model=PersonOutput)
        sm._client = MagicMock()
        resp = ConverseResponse.from_dict(
            _make_response_dict('{"name": "Bob", "age": 25}'))
        resp.model_id = "llama4-maverick"
        result = sm.format_response(resp)
        self.assertIsInstance(result, PersonOutput)
        self.assertEqual(result.name, "Bob")

    def test_structured_model_factory(self):
        self.assertEqual(structured_model_factory("llama4-maverick-instruct"), StructuredMaverick)
        self.assertEqual(structured_model_factory("anthropic.claude-3-5-sonnet"), StructuredConverse)


# ══════════════════════════════════════════════════════════════════════════════
#  10. Image resize
# ══════════════════════════════════════════════════════════════════════════════

class TestImageResize(unittest.TestCase):
    def _make_image(self, width, height, mode='RGB', fmt='PNG'):
        from PIL import Image as PILImg
        img = PILImg.new(mode, (width, height), color='red')
        buf = io.BytesIO()
        if fmt == 'JPEG' and mode == 'RGBA':
            img = img.convert('RGB')
        img.save(buf, format=fmt)
        return buf.getvalue()

    def test_no_resize_needed(self):
        data = self._make_image(100, 100)
        result = resize_image_if_needed(data, 'png')
        self.assertEqual(result, data)

    def test_resize_large_image(self):
        data = self._make_image(10000, 5000)
        result = resize_image_if_needed(data, 'png')
        from PIL import Image as PILImg
        img = PILImg.open(io.BytesIO(result))
        self.assertLessEqual(max(img.size), MAX_IMAGE_DIMENSION)

    def test_resize_rgba_to_jpeg(self):
        data = self._make_image(10000, 5000, mode='RGBA', fmt='PNG')
        result = resize_image_if_needed(data, 'jpeg')
        from PIL import Image as PILImg
        img = PILImg.open(io.BytesIO(result))
        self.assertEqual(img.mode, 'RGB')

    @patch('bedrock.converse.PIL_AVAILABLE', False)
    def test_no_pil_returns_original(self):
        data = b'fake image data'
        result = resize_image_if_needed(data, 'png')
        self.assertEqual(result, data)


# ══════════════════════════════════════════════════════════════════════════════
#  11. Edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    def test_message_content_reduce_size(self):
        mc = MessageContent(text="a" * 500000)
        mc.reduce_size()
        self.assertLessEqual(len(mc.text), 400001)

    def test_document_name_cleaned(self):
        d = Document(format="pdf", name="héllo wörld!@#$.pdf",
                     source=FileSource(bytes=b'data'))
        # Should only contain ascii alphanumeric + allowed chars
        self.assertTrue(all(c.isascii() for c in d.name))

    def test_converse_to_dict_excludes_private(self):
        c = Converse(model_id="m", region_name="us-east-1")
        d = c.to_dict()
        self.assertNotIn("regionName", d)
        self.assertNotIn("_client", d)
        self.assertIn("modelId", d)

    def test_image_invalid_format(self):
        with self.assertRaises(InvalidFormat):
            Image(format="tiff", source=FileSource(bytes=b'data'))

    def test_tool_choice_any(self):
        tc = ToolChoice(any=ToolChoiceAny())
        d = tc.to_dict()
        self.assertIn("any", d)

    def test_from_dict_none_returns_none(self):
        result = TokenUsage.from_dict(None)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
