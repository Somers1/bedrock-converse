"""Tests for LangfuseCallback and agent lifecycle hooks."""

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from bedrock.bases import BaseCallbackHandler
from bedrock.langfuse_callback import LangfuseCallback, _get_langfuse


# --- Test _get_langfuse ---

def test_get_langfuse_returns_none_when_not_installed():
    with patch.dict('sys.modules', {'langfuse': None}):
        result = _get_langfuse()
        assert result is None


def test_get_langfuse_returns_none_on_init_error():
    mock_module = MagicMock()
    mock_module.Langfuse.side_effect = Exception("bad config")
    with patch.dict('sys.modules', {'langfuse': mock_module}):
        result = _get_langfuse()
        assert result is None


# --- Test LangfuseCallback with no langfuse installed ---

class TestLangfuseDisabled:
    def setup_method(self):
        self.cb = LangfuseCallback(user_id="u1", session_id="s1", tags=["test"])
        self.cb._langfuse = None  # force disabled

    def test_enabled_is_false(self):
        assert self.cb.enabled is False

    def test_on_run_start_noop(self):
        self.cb.on_run_start(MagicMock())
        assert self.cb._trace is None

    def test_on_run_end_noop(self):
        self.cb.on_run_end(MagicMock(), "result")

    def test_on_converse_start_noop(self):
        self.cb.on_converse_start(MagicMock())
        assert self.cb._generation is None

    def test_on_converse_end_noop(self):
        self.cb.on_converse_end(MagicMock())

    def test_on_tool_start_noop(self):
        self.cb.on_tool_start("tool", {}, "id1")
        assert self.cb._tool_spans == {}

    def test_on_tool_end_noop(self):
        self.cb.on_tool_end("tool", {}, "id1", "result", "success", 0.1)


# --- Test LangfuseCallback with mocked langfuse ---

class TestLangfuseEnabled:
    def setup_method(self):
        self.cb = LangfuseCallback(user_id="u1", session_id="s1", tags=["heartbeat"], metadata={"env": "test"})
        self.mock_langfuse = MagicMock()
        self.cb._langfuse = self.mock_langfuse
        self.mock_trace = MagicMock()
        self.mock_langfuse.trace.return_value = self.mock_trace
        self.mock_generation = MagicMock()
        self.mock_trace.generation.return_value = self.mock_generation
        self.mock_span = MagicMock()
        self.mock_trace.span.return_value = self.mock_span

    def test_enabled_is_true(self):
        assert self.cb.enabled is True

    def test_on_run_start_creates_trace(self):
        agent = MagicMock()
        agent.model_id = "test-model"
        self.cb.on_run_start(agent)
        self.mock_langfuse.trace.assert_called_once_with(
            name="agent.run",
            user_id="u1",
            session_id="s1",
            tags=["heartbeat"],
            metadata={"env": "test", "model_id": "test-model"},
        )
        assert self.cb._trace is self.mock_trace

    def test_on_run_end_updates_trace_and_flushes(self):
        self.cb._trace = self.mock_trace
        self.cb.on_run_end(MagicMock(), "final result")
        self.mock_trace.update.assert_called_once_with(output="final result")
        self.mock_langfuse.flush.assert_called_once()

    def test_on_run_end_truncates_long_result(self):
        self.cb._trace = self.mock_trace
        long_result = "x" * 3000
        self.cb.on_run_end(MagicMock(), long_result)
        output = self.mock_trace.update.call_args[1]['output']
        assert len(output) == 2000

    def test_on_run_end_handles_none_result(self):
        self.cb._trace = self.mock_trace
        self.cb.on_run_end(MagicMock(), None)
        self.mock_trace.update.assert_called_once_with(output=None)

    def test_on_run_end_handles_flush_error(self):
        self.cb._trace = self.mock_trace
        self.mock_langfuse.flush.side_effect = Exception("network error")
        self.cb.on_run_end(MagicMock(), "result")  # should not raise

    def test_on_converse_start_creates_generation(self):
        self.cb._trace = self.mock_trace
        converse = MagicMock()
        converse.model_id = "kimi-k2.5"
        self.cb.on_converse_start(converse)
        self.mock_trace.generation.assert_called_once()
        assert self.cb._generation is self.mock_generation
        assert self.cb._generation_start is not None

    def test_on_converse_start_without_trace_noops(self):
        self.cb._trace = None
        self.cb.on_converse_start(MagicMock())
        assert self.cb._generation is None

    def test_on_converse_end_ends_generation(self):
        self.cb._trace = self.mock_trace
        self.cb._generation = self.mock_generation
        self.cb._generation_start = time.time()

        response = MagicMock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50
        response.usage.cache_read_input_tokens = 20
        response.cost.total = 0.001
        response.metrics.latency_ms = 500
        response.stop_reason = "end_turn"

        self.cb.on_converse_end(response)
        self.mock_generation.end.assert_called_once()
        call_kwargs = self.mock_generation.end.call_args[1]
        assert call_kwargs['usage']['input'] == 100
        assert call_kwargs['usage']['output'] == 50
        assert call_kwargs['usage']['input_cached'] == 20
        assert call_kwargs['metadata']['stop_reason'] == "end_turn"
        assert self.cb._generation is None

    def test_on_converse_end_without_cache(self):
        self.cb._trace = self.mock_trace
        self.cb._generation = self.mock_generation
        self.cb._generation_start = time.time()

        response = MagicMock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50
        response.usage.cache_read_input_tokens = 0
        response.cost.total = 0.001
        response.metrics.latency_ms = 200
        response.stop_reason = "end_turn"

        self.cb.on_converse_end(response)
        call_kwargs = self.mock_generation.end.call_args[1]
        assert 'input_cached' not in call_kwargs['usage']

    def test_on_converse_end_without_cost(self):
        self.cb._trace = self.mock_trace
        self.cb._generation = self.mock_generation
        self.cb._generation_start = time.time()

        response = MagicMock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50
        response.usage.cache_read_input_tokens = 0
        response.cost = None
        response.metrics.latency_ms = 200
        response.stop_reason = "end_turn"

        self.cb.on_converse_end(response)
        call_kwargs = self.mock_generation.end.call_args[1]
        assert call_kwargs['metadata']['cost_usd'] is None

    def test_on_tool_start_creates_span(self):
        self.cb._trace = self.mock_trace
        self.cb.on_tool_start("create_state", {"content": "test"}, "tool-123")
        self.mock_trace.span.assert_called_once_with(
            name="create_state", input={"content": "test"}, metadata={"tool_use_id": "tool-123"}
        )
        assert "tool-123" in self.cb._tool_spans

    def test_on_tool_end_ends_span(self):
        self.cb._trace = self.mock_trace
        self.cb._tool_spans["tool-123"] = (self.mock_span, time.time())
        self.cb.on_tool_end("create_state", {"content": "test"}, "tool-123", "ok", "success", 0.05)
        self.mock_span.end.assert_called_once()
        call_kwargs = self.mock_span.end.call_args[1]
        assert call_kwargs['output'] == "ok"
        assert call_kwargs['metadata']['status'] == "success"
        assert "tool-123" not in self.cb._tool_spans

    def test_on_tool_end_missing_span_noops(self):
        self.cb.on_tool_end("create_state", {}, "nonexistent", "ok", "success", 0.1)

    def test_on_tool_end_truncates_long_result(self):
        self.cb._trace = self.mock_trace
        self.cb._tool_spans["t1"] = (self.mock_span, time.time())
        self.cb.on_tool_end("tool", {}, "t1", "x" * 3000, "success", 0.1)
        output = self.mock_span.end.call_args[1]['output']
        assert len(output) == 2000

    def test_full_lifecycle(self):
        """Simulates a complete agent run with LLM call + tool."""
        agent = MagicMock()
        agent.model_id = "test-model"

        response = MagicMock()
        response.usage.input_tokens = 1000
        response.usage.output_tokens = 200
        response.usage.cache_read_input_tokens = 0
        response.cost.total = 0.01
        response.metrics.latency_ms = 800
        response.stop_reason = "tool_use"

        # Run start
        self.cb.on_run_start(agent)
        assert self.cb._trace is not None

        # LLM call
        self.cb.on_converse_start(agent)
        self.cb.on_converse_end(response)

        # Tool execution
        self.cb.on_tool_start("send_message", {"text": "hi"}, "t1")
        self.cb.on_tool_end("send_message", {"text": "hi"}, "t1", "sent", "success", 0.02)

        # Second LLM call
        response.stop_reason = "end_turn"
        self.cb.on_converse_start(agent)
        self.cb.on_converse_end(response)

        # Run end
        self.cb.on_run_end(agent, "Done")

        assert self.mock_langfuse.trace.call_count == 1
        assert self.mock_trace.generation.call_count == 2
        assert self.mock_trace.span.call_count == 1
        self.mock_langfuse.flush.assert_called_once()


# --- Test BaseCallbackHandler new hooks have defaults ---

class TestBaseCallbackHooks:
    def test_new_hooks_have_default_implementations(self):
        """on_tool_start, on_tool_end, on_run_start, on_run_end should be callable without override."""

        class MinimalCallback(BaseCallbackHandler):
            def on_converse_start(self, converse): pass
            def on_converse_end(self, response): pass

        cb = MinimalCallback()
        # These should not raise
        cb.on_tool_start("tool", {}, "id")
        cb.on_tool_end("tool", {}, "id", "result", "success", 0.1)
        cb.on_run_start(MagicMock())
        cb.on_run_end(MagicMock(), "result")


# --- Test agent run() fires lifecycle hooks ---

class TestAgentRunHooks:
    """Test that ConverseAgent.run() fires the new callback hooks."""

    def test_run_fires_on_run_start_and_end(self):
        """Verify on_run_start and on_run_end are called during agent.run()."""
        from bedrock.converse import ConverseAgent, Message, MessageContent, ConverseResponse

        mock_cb = MagicMock(spec=BaseCallbackHandler)
        mock_cb.on_run_start = MagicMock()
        mock_cb.on_run_end = MagicMock()
        mock_cb.on_converse_start = MagicMock()
        mock_cb.on_converse_end = MagicMock()
        mock_cb.on_tool_start = MagicMock()
        mock_cb.on_tool_end = MagicMock()

        agent = ConverseAgent(model_id="test", region_name="us-east-1")
        agent.callbacks.append(mock_cb)

        # Mock _get_response to return a text-only response
        mock_response = MagicMock()
        mock_message = MagicMock()
        text_content = MagicMock()
        text_content.text = "Hello"
        text_content.tool_use = None
        mock_message.content = [text_content]
        mock_response.output.message = mock_message
        mock_response.output.message.content = [text_content]
        agent._get_response = MagicMock(return_value=mock_response)

        result = agent.run("test input")

        mock_cb.on_run_start.assert_called_once_with(agent)
        mock_cb.on_run_end.assert_called_once()
        assert mock_cb.on_run_end.call_args[0][1] == "Hello"
