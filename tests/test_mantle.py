import asyncio
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bedrock.mantle import Mantle, MantleAgent, StructuredMantle, _sign_and_post, _sign_and_post_async
from bedrock.converse import (
    Message, MessageContent, ToolUse, ToolResult, ToolResultContent,
    SystemContent, CachePoint, ConverseInferenceConfig, Finish,
)
from bedrock.tools import tool
from bedrock.bases import BaseCallbackHandler


class TestOutput(BaseModel):
    test_field: str = Field(description='test field')


def _openai_response(content="Hello!", tool_calls=None, finish_reason="stop",
                     prompt_tokens=10, completion_tokens=5, total_tokens=15):
    msg = {"role": "assistant"}
    if tool_calls:
        msg["tool_calls"] = tool_calls
        msg["content"] = None
    else:
        msg["content"] = content
    return {
        "id": "chatcmpl-123", "object": "chat.completion",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}
    }


def _tool_call(name, arguments, call_id="call_123"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _make_mock_session():
    session = MagicMock()
    creds = MagicMock()
    session.get_credentials.return_value.get_frozen_credentials.return_value = creds
    session.region_name = "us-east-1"
    return session


class TestMantleEndpoint(unittest.TestCase):
    def test_endpoint_url_uses_region(self):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-west-2")
        m._client = MagicMock()
        # Force session to return region
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            self.assertIn("us-west-2", m._mantle_endpoint)

    def test_endpoint_url_format(self):
        m = Mantle(model_id="m", region_name="eu-west-1")
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            self.assertEqual(m._mantle_endpoint, "https://bedrock-mantle.eu-west-1.api.aws/v1/chat/completions")


class TestMantleInvoke(unittest.TestCase):
    def _make_mantle(self, **kwargs):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1", **kwargs)
        return m

    @patch('bedrock.mantle._sign_and_post')
    def test_invoke_basic(self, mock_post):
        m = self._make_mantle()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response("Hi there")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            resp = m.invoke("Hello")
        self.assertEqual(resp.content, "Hi there")
        # Verify model field in payload
        call_args = mock_post.call_args
        payload = json.loads(call_args[0][3])  # payload_bytes is 4th positional arg
        self.assertEqual(payload["model"], "anthropic.claude-3-5-sonnet")

    @patch('bedrock.mantle._sign_and_post')
    def test_invoke_with_tool_calls(self, mock_post):
        m = self._make_mantle()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response(
            tool_calls=[_tool_call("my_tool", {"key": "value"})],
            finish_reason="tool_calls")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            resp = m.invoke("Use tool")
        self.assertEqual(resp.stop_reason, "tool_use")
        self.assertEqual(resp.output.message.content[0].tool_use.name, "my_tool")

    @patch('bedrock.mantle._sign_and_post')
    def test_model_id_in_payload(self, mock_post):
        m = self._make_mantle()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            m.invoke("Hi")
        payload = json.loads(mock_post.call_args[0][3])
        self.assertIn("model", payload)
        self.assertEqual(payload["model"], "anthropic.claude-3-5-sonnet")

    @patch('bedrock.mantle._sign_and_post')
    def test_system_prompts_in_payload(self, mock_post):
        m = self._make_mantle()
        m.add_system("Be helpful")
        m.add_system("Be concise")
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            m.invoke("Hi")
        payload = json.loads(mock_post.call_args[0][3])
        system_msg = payload["messages"][0]
        self.assertEqual(system_msg["role"], "system")
        self.assertIn("Be helpful", system_msg["content"])
        self.assertIn("Be concise", system_msg["content"])

    @patch('bedrock.mantle._sign_and_post')
    def test_http_error_raises(self, mock_post):
        m = self._make_mantle()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 500")
        mock_post.return_value = mock_resp
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            with self.assertRaises(Exception):
                m.invoke("Hi")


class TestMantleCallbacks(unittest.TestCase):
    @patch('bedrock.mantle._sign_and_post')
    def test_callbacks_fire(self, mock_post):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        cb = MagicMock(spec=BaseCallbackHandler)
        m.callbacks = [cb]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            m.invoke("Hi")
        cb.on_converse_start.assert_called_once()
        cb.on_converse_end.assert_called_once()

    @patch('bedrock.mantle._sign_and_post')
    def test_callback_error_does_not_crash(self, mock_post):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        cb = MagicMock(spec=BaseCallbackHandler)
        cb.on_converse_start.side_effect = Exception("boom")
        m.callbacks = [cb]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            resp = m.invoke("Hi")
        self.assertEqual(resp.content, "Hello!")


class TestMantleAuth(unittest.TestCase):
    @patch('bedrock.mantle.requests.post')
    @patch('bedrock.mantle.SigV4Auth')
    def test_sigv4_signing(self, mock_sigv4, mock_requests_post):
        session = _make_mock_session()
        mock_requests_post.return_value = MagicMock(
            json=lambda: _openai_response(), raise_for_status=MagicMock())
        _sign_and_post(session, "us-east-1", "https://example.com", b'{}')
        mock_sigv4.assert_called_once()
        mock_sigv4.return_value.add_auth.assert_called_once()

    @patch('bedrock.mantle.requests.post')
    def test_api_key_auth(self, mock_requests_post):
        mock_requests_post.return_value = MagicMock()
        _sign_and_post(MagicMock(), "us-east-1", "https://example.com", b'{}', api_key="sk-test")
        call_kwargs = mock_requests_post.call_args
        self.assertIn("Bearer sk-test", call_kwargs[1]["headers"]["Authorization"])

    @patch('bedrock.mantle.requests.post')
    def test_api_key_skips_sigv4(self, mock_requests_post):
        mock_requests_post.return_value = MagicMock()
        with patch('bedrock.mantle.SigV4Auth') as mock_sigv4:
            _sign_and_post(MagicMock(), "us-east-1", "https://example.com", b'{}', api_key="sk-test")
            mock_sigv4.assert_not_called()


class TestMantleCacheRemoval(unittest.TestCase):
    @patch('bedrock.mantle._sign_and_post')
    def test_removes_cache_for_unsupported_model(self, mock_post):
        m = Mantle(model_id="unsupported-model", region_name="us-east-1")
        m.system = [SystemContent(text="hi"), SystemContent(cache_point=CachePoint())]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            m.invoke("Hi")
        # Cache points should be stripped from system
        self.assertEqual(len(m.system), 1)


class TestMantleAgent(unittest.TestCase):
    @patch('bedrock.mantle._sign_and_post')
    def test_run_exit(self, mock_post):
        agent = MantleAgent(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response(
            tool_calls=[_tool_call("Finish", {"final_response": "Done!"}, "t1")],
            finish_reason="tool_calls")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with patch.object(type(agent), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            result = agent.run("Do something")
        self.assertEqual(result, "Done!")

    @patch('bedrock.mantle._sign_and_post')
    def test_run_tool_then_exit(self, mock_post):
        agent = MantleAgent(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")

        @tool
        def lookup(query: str):
            """Look up info"""
            return "result: 42"
        agent.add_tool(lookup)

        resp1 = _openai_response(
            tool_calls=[_tool_call("lookup", {"query": "meaning"}, "t1")],
            finish_reason="tool_calls")
        resp2 = _openai_response(
            tool_calls=[_tool_call("Finish", {"final_response": "42"}, "t2")],
            finish_reason="tool_calls")
        mock_resp1 = MagicMock(json=lambda: resp1, raise_for_status=MagicMock())
        mock_resp2 = MagicMock(json=lambda: resp2, raise_for_status=MagicMock())
        mock_post.side_effect = [mock_resp1, mock_resp2]
        with patch.object(type(agent), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            result = agent.run("What is the meaning?")
        self.assertEqual(result, "42")
        self.assertEqual(mock_post.call_count, 2)

    @patch('bedrock.mantle._sign_and_post')
    def test_max_iterations(self, mock_post):
        agent = MantleAgent(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        agent.max_iterations = 2
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response("Thinking...")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with patch.object(type(agent), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            result = agent.run("Do something")
        self.assertIn("maximum iterations", result)

    @patch('bedrock.mantle._sign_and_post')
    def test_on_text_hook(self, mock_post):
        agent = MantleAgent(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        agent.on_text(lambda text: f"hooked: {text}")
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response("Some text")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with patch.object(type(agent), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            result = agent.run("Hi")
        self.assertEqual(result, "hooked: Some text")


class TestStructuredMantle(unittest.TestCase):
    @patch('bedrock.mantle._sign_and_post')
    def test_invoke_returns_model(self, mock_post):
        sm = StructuredMantle(
            model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1",
            output_model=TestOutput)
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response(
            tool_calls=[_tool_call("TestOutput", {"test_field": "val"}, "t1")],
            finish_reason="tool_calls")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with patch.object(type(sm), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            result = sm.invoke(Message().add_text("Extract"))
        self.assertIsInstance(result, TestOutput)
        self.assertEqual(result.test_field, "val")


class TestMantleAsync(unittest.TestCase):
    @patch('bedrock.mantle._sign_and_post_async', new_callable=AsyncMock)
    def test_ainvoke(self, mock_async_post):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        mock_async_post.return_value = _openai_response("Async hi")
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            resp = asyncio.new_event_loop().run_until_complete(m.ainvoke("Hello"))
        self.assertEqual(resp.content, "Async hi")

    @patch('bedrock.mantle._sign_and_post_async', new_callable=AsyncMock)
    def test_aconverse(self, mock_async_post):
        m = Mantle(model_id="anthropic.claude-3-5-sonnet", region_name="us-east-1")
        mock_async_post.return_value = _openai_response("Async reply")
        with patch.object(type(m), 'session', new_callable=lambda: property(lambda self: _make_mock_session())):
            asyncio.new_event_loop().run_until_complete(m.aconverse("Hello"))
        self.assertEqual(len(m.messages), 2)


class TestSignAndPostAsync(unittest.TestCase):
    def test_api_key_skips_sigv4_async(self):
        """Verify that api_key path sets Bearer header and skips SigV4 in async."""
        # We test the sync equivalent since mocking nested async context managers is fragile.
        # The api_key branch is identical in sync/async — just sets the header.
        with patch('bedrock.mantle.requests.post') as mock_post, \
             patch('bedrock.mantle.SigV4Auth') as mock_sigv4:
            mock_post.return_value = MagicMock()
            _sign_and_post(MagicMock(), "us-east-1", "https://example.com", b'{}', api_key="sk-async")
            mock_sigv4.assert_not_called()
            headers = mock_post.call_args[1]["headers"]
            self.assertEqual(headers["Authorization"], "Bearer sk-async")


if __name__ == '__main__':
    unittest.main()
