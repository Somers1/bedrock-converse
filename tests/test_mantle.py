import asyncio
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, AsyncMock

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bedrock.mantle import Mantle, MantleAgent, StructuredMantle
from bedrock.converse import (
    Message, MessageContent, ToolUse, ToolResult, ToolResultContent,
    SystemContent, CachePoint, ConverseInferenceConfig, Finish,
)
from bedrock.tools import tool
from bedrock.bases import BaseCallbackHandler


class TestOutput(BaseModel):
    test_field: str = Field(description='test field')


def _mock_completion(content="Hello!", tool_calls=None, finish_reason="stop",
                     prompt_tokens=10, completion_tokens=5, total_tokens=15):
    """Build a mock ChatCompletion object matching openai SDK structure."""
    message = MagicMock()
    message.content = content if not tool_calls else None
    if tool_calls:
        mock_tcs = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc['id']
            mock_tc.function.name = tc['function']['name']
            mock_tc.function.arguments = tc['function']['arguments']
            mock_tcs.append(mock_tc)
        message.tool_calls = mock_tcs
    else:
        message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    return completion


def _tool_call(name, arguments, call_id="call_123"):
    return {'id': call_id, 'function': {'name': name, 'arguments': json.dumps(arguments)}}


def _make_mock_session():
    session = MagicMock()
    creds = MagicMock()
    session.get_credentials.return_value.get_frozen_credentials.return_value = creds
    session.region_name = "us-east-1"
    return session


class TestMantleBuildParams(unittest.TestCase):
    def test_basic_params(self):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        m.add_system("Be helpful")
        m.inference_config = ConverseInferenceConfig(max_tokens=100, temperature=0.5)
        msgs = [Message().add_text("Hello")]
        params = m._build_params(msgs)
        self.assertEqual(params['model'], 'anthropic.claude-3-5-sonnet')
        self.assertEqual(params['messages'][0]['role'], 'system')
        self.assertEqual(params['messages'][1]['role'], 'user')
        self.assertEqual(params['max_tokens'], 100)
        self.assertEqual(params['temperature'], 0.5)

    def test_thinking_maps_to_reasoning_effort(self):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        m.with_thinking(tokens=1024)
        params = m._build_params([Message().add_text("Hi")])
        self.assertEqual(params['reasoning_effort'], 'low')

    def test_thinking_medium(self):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        m.with_thinking(tokens=4096)
        params = m._build_params([Message().add_text("Hi")])
        self.assertEqual(params['reasoning_effort'], 'medium')

    def test_thinking_high(self):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        m.with_thinking(tokens=16000)
        params = m._build_params([Message().add_text("Hi")])
        self.assertEqual(params['reasoning_effort'], 'high')


class TestMantleInvoke(unittest.TestCase):
    def _make_mantle(self, **kwargs):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1", **kwargs)
        mock_client = MagicMock()
        # Bypass cached_property by setting on instance __dict__
        m.__dict__['openai_client'] = mock_client
        return m, mock_client

    def test_invoke_basic(self):
        m, client = self._make_mantle()
        client.chat.completions.create.return_value = _mock_completion("Hi there")
        resp = m.invoke("Hello")
        self.assertEqual(resp.content, "Hi there")
        client.chat.completions.create.assert_called_once()
        call_kwargs = client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs['model'], 'anthropic.claude-3-5-sonnet')

    def test_invoke_with_tool_calls(self):
        m, client = self._make_mantle()
        client.chat.completions.create.return_value = _mock_completion(
            tool_calls=[_tool_call("my_tool", {"key": "value"})], finish_reason="tool_calls")
        resp = m.invoke("Use tool")
        self.assertEqual(resp.stop_reason, "tool_use")
        self.assertEqual(resp.output.message.content[0].tool_use.name, "my_tool")

    def test_system_prompts_in_params(self):
        m, client = self._make_mantle()
        m.add_system("Be helpful")
        m.add_system("Be concise")
        client.chat.completions.create.return_value = _mock_completion()
        m.invoke("Hi")
        call_kwargs = client.chat.completions.create.call_args[1]
        system_msg = call_kwargs['messages'][0]
        self.assertEqual(system_msg['role'], 'system')
        self.assertIn('Be helpful', system_msg['content'])
        self.assertIn('Be concise', system_msg['content'])


class TestMantleCallbacks(unittest.TestCase):
    def test_callbacks_fire(self):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        mock_client = MagicMock()
        m.__dict__['openai_client'] = mock_client
        cb = MagicMock(spec=BaseCallbackHandler)
        m.callbacks = [cb]
        mock_client.chat.completions.create.return_value = _mock_completion()
        m.invoke("Hi")
        cb.on_converse_start.assert_called_once()
        cb.on_converse_end.assert_called_once()

    def test_callback_error_does_not_crash(self):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        mock_client = MagicMock()
        m.__dict__['openai_client'] = mock_client
        cb = MagicMock(spec=BaseCallbackHandler)
        cb.on_converse_start.side_effect = Exception("boom")
        m.callbacks = [cb]
        mock_client.chat.completions.create.return_value = _mock_completion()
        resp = m.invoke("Hi")
        self.assertEqual(resp.content, "Hello!")


class TestMantleAuth(unittest.TestCase):
    def test_api_key_creates_client(self):
        m = Mantle(model_id="m", region_name="us-east-1", api_key="sk-test")
        with patch('bedrock.mantle.OpenAI') as mock_openai:
            mock_openai.return_value = MagicMock()
            client = m.openai_client
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            self.assertEqual(call_kwargs['api_key'], 'sk-test')

    def test_sigv4_creates_client_with_httpx(self):
        m = Mantle(model_id="m", region_name="us-east-1")
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            with patch('bedrock.mantle.OpenAI') as mock_openai:
                with patch('bedrock.mantle.httpx.Client') as mock_httpx:
                    mock_openai.return_value = MagicMock()
                    # Clear cached property
                    m.__dict__.pop('openai_client', None)
                    client = m.openai_client
                    mock_httpx.assert_called_once()
                    mock_openai.assert_called_once()


class TestMantleCacheRemoval(unittest.TestCase):
    def test_removes_cache_for_unsupported_model(self):
        m = Mantle(model_id="unsupported-model", region_name="us-east-1")
        mock_client = MagicMock()
        m.__dict__['openai_client'] = mock_client
        m.system = [SystemContent(text="hi"), SystemContent(cache_point=CachePoint())]
        mock_client.chat.completions.create.return_value = _mock_completion()
        m.invoke("Hi")
        self.assertEqual(len(m.system), 1)


class TestMantleAgent(unittest.TestCase):
    def _make_agent(self):
        agent = MantleAgent(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        mock_client = MagicMock()
        agent.__dict__['openai_client'] = mock_client
        return agent, mock_client

    def test_run_exit(self):
        agent, client = self._make_agent()
        client.chat.completions.create.return_value = _mock_completion(
            tool_calls=[_tool_call("Finish", {"final_response": "Done!"}, "t1")], finish_reason="tool_calls")
        result = agent.run("Do something")
        self.assertEqual(result, "Done!")

    def test_run_tool_then_exit(self):
        agent, client = self._make_agent()

        @tool
        def lookup(query: str):
            """Look up info"""
            return "result: 42"
        agent.add_tool(lookup)

        client.chat.completions.create.side_effect = [
            _mock_completion(tool_calls=[_tool_call("lookup", {"query": "meaning"}, "t1")], finish_reason="tool_calls"),
            _mock_completion(tool_calls=[_tool_call("Finish", {"final_response": "42"}, "t2")], finish_reason="tool_calls"),
        ]
        result = agent.run("What is the meaning?")
        self.assertEqual(result, "42")
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_max_iterations(self):
        agent, client = self._make_agent()
        agent.max_iterations = 2
        client.chat.completions.create.return_value = _mock_completion("Thinking...")
        result = agent.run("Do something")
        self.assertIn("maximum iterations", result)

    def test_on_text_hook(self):
        agent, client = self._make_agent()
        agent.on_text(lambda text: f"hooked: {text}")
        client.chat.completions.create.return_value = _mock_completion("Some text")
        result = agent.run("Hi")
        self.assertEqual(result, "hooked: Some text")


class TestStructuredMantle(unittest.TestCase):
    def test_invoke_returns_model(self):
        sm = StructuredMantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1", output_model=TestOutput)
        mock_client = MagicMock()
        sm.__dict__['openai_client'] = mock_client
        mock_client.chat.completions.create.return_value = _mock_completion(
            tool_calls=[_tool_call("TestOutput", {"test_field": "val"}, "t1")], finish_reason="tool_calls")
        result = sm.invoke(Message().add_text("Extract"))
        self.assertIsInstance(result, TestOutput)
        self.assertEqual(result.test_field, "val")


class TestMantleAsync(unittest.TestCase):
    def test_ainvoke(self):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        mock_client = AsyncMock()
        m.__dict__['async_openai_client'] = mock_client
        mock_client.chat.completions.create.return_value = _mock_completion("Async hi")
        resp = asyncio.new_event_loop().run_until_complete(m.ainvoke("Hello"))
        self.assertEqual(resp.content, "Async hi")

    def test_aconverse(self):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        mock_client = AsyncMock()
        m.__dict__['async_openai_client'] = mock_client
        mock_client.chat.completions.create.return_value = _mock_completion("Async reply")
        asyncio.new_event_loop().run_until_complete(m.aconverse("Hello"))
        self.assertEqual(len(m.messages), 2)


class TestMantleEndpoint(unittest.TestCase):
    def test_endpoint_url_uses_region(self):
        m = Mantle(model_id="m", region_name="us-west-2")
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            self.assertIn("us-west-2", m._mantle_base_url)

    def test_endpoint_url_format(self):
        m = Mantle(model_id="m", region_name="eu-west-1")
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            self.assertEqual(m._mantle_base_url, "https://bedrock-mantle.eu-west-1.api.aws/v1")


if __name__ == '__main__':
    unittest.main()
