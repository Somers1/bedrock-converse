"""Tests for LangfuseCallback and agent lifecycle hooks."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bedrock.bases import BaseCallbackHandler
from bedrock.langfuse_callback import LangfuseCallback, _get_langfuse


def _mk_content(**kwargs):
    defaults = {
        "text": None,
        "tool_use": None,
        "tool_result": None,
        "reasoning_content": None,
        "image": None,
        "document": None,
        "video": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _mk_message(role="user", content=None):
    return SimpleNamespace(role=role, content=content or [])


# --- Test _get_langfuse ---


def test_get_langfuse_returns_none_when_not_installed():
    with patch.dict("sys.modules", {"langfuse": None}):
        result = _get_langfuse()
        assert result is None


def test_get_langfuse_returns_none_on_init_error():
    mock_module = MagicMock()
    mock_module.Langfuse.side_effect = Exception("bad config")
    with patch.dict("sys.modules", {"langfuse": mock_module}):
        result = _get_langfuse()
        assert result is None


# --- Test LangfuseCallback with no langfuse installed ---


class TestLangfuseDisabled:
    def setup_method(self):
        self.cb = LangfuseCallback(user_id="u1", session_id="s1", tags=["test"])
        self.cb._langfuse = None

    def test_enabled_is_false(self):
        assert self.cb.enabled is False

    def test_hooks_noop_when_disabled(self):
        self.cb.on_run_start(MagicMock())
        self.cb.on_converse_start(MagicMock())
        self.cb.on_converse_end(MagicMock())
        self.cb.on_tool_start("tool", {}, "id")
        self.cb.on_tool_end("tool", {}, "id", "result", "success", 0.1)
        self.cb.on_run_end(MagicMock(), "result")

        assert self.cb._trace is None
        assert self.cb._generation is None
        assert self.cb._tool_spans == {}


# --- Test LangfuseCallback with mocked langfuse (v3 API) ---


class TestLangfuseEnabled:
    def setup_method(self):
        self.cb = LangfuseCallback(user_id="u1", session_id="s1", tags=["heartbeat"], metadata={"env": "test"})

        self.mock_langfuse = MagicMock()
        self.mock_trace = MagicMock()
        self.mock_generation = MagicMock()
        self.mock_tool = MagicMock()

        self.mock_scope = MagicMock()
        self.mock_scope.__enter__ = MagicMock(return_value=self.mock_scope)
        self.mock_scope.__exit__ = MagicMock(return_value=None)
        self.mock_langfuse.propagate_attributes.return_value = self.mock_scope

        def root_start_observation(*args, **kwargs):
            as_type = kwargs.get("as_type")
            if as_type == "agent":
                return self.mock_trace
            return MagicMock()

        def child_start_observation(*args, **kwargs):
            as_type = kwargs.get("as_type")
            if as_type == "generation":
                return self.mock_generation
            if as_type == "tool":
                return self.mock_tool
            return MagicMock()

        self.mock_langfuse.start_observation.side_effect = root_start_observation
        self.mock_trace.start_observation.side_effect = child_start_observation

        self.cb._langfuse = self.mock_langfuse

    def test_enabled_is_true(self):
        assert self.cb.enabled is True

    def test_on_run_start_creates_agent_observation_and_trace_attrs(self):
        long_text = "x" * 10000
        system_text = "You are a precise assistant."
        agent = SimpleNamespace(
            model_id="test-model",
            max_iterations=5,
            messages=[_mk_message(role="user", content=[_mk_content(text=long_text)])],
            system=[SimpleNamespace(text=system_text, guard_content=None, cache_point=None)],
        )

        self.cb.on_run_start(agent)

        self.mock_langfuse.start_observation.assert_called_once()
        call_kwargs = self.mock_langfuse.start_observation.call_args.kwargs
        assert call_kwargs["as_type"] == "agent"
        assert call_kwargs["name"] == "agent.run"

        messages = call_kwargs["input"]["messages"]["messages"]
        assert messages[0]["content"][0]["text"] == long_text
        assert call_kwargs["input"]["system"]["prompts"][0]["text"] == system_text

        self.mock_trace.update_trace.assert_called_once()
        self.mock_langfuse.propagate_attributes.assert_called_once()
        assert self.cb._trace is self.mock_trace

    def test_on_converse_start_and_end_capture_full_generation_payload(self):
        self.cb._trace = self.mock_trace
        self.cb._trace_started_by_run = True

        user_text = "u" * 8000
        system_text = "System prompt with full text visibility."
        converse = SimpleNamespace(
            model_id="kimi-k2.5",
            messages=[_mk_message(role="user", content=[_mk_content(text=user_text)])],
            system=[SimpleNamespace(text=system_text, guard_content=None, cache_point=None)],
            inference_config=SimpleNamespace(max_tokens=123, temperature=0.2, top_p=0.9, stop_sequences=["stop"]),
            request_metadata={"request_id": "abc"},
            performance_config=SimpleNamespace(latency="standard"),
            tool_config=SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        tool_spec=SimpleNamespace(
                            name="get_weather",
                            input_schema={"json": {"type": "object", "properties": {"city": {"type": "string"}}}},
                        ),
                        cache_point=SimpleNamespace(to_dict=lambda: {"type": "default", "ttl": "5m"}),
                    )
                ]
            ),
        )

        self.cb.on_converse_start(converse)
        self.mock_trace.start_observation.assert_called_once()
        start_kwargs = self.mock_trace.start_observation.call_args.kwargs
        assert start_kwargs["as_type"] == "generation"
        assert start_kwargs["input"]["messages"]["messages"][0]["content"][0]["text"] == user_text
        assert start_kwargs["input"]["system"]["prompts"][0]["text"] == system_text
        assert start_kwargs["metadata"]["tool_names"] == ["get_weather"]
        assert "get_weather" in start_kwargs["metadata"]["tool_schemas"]
        assert start_kwargs["metadata"]["tool_cache_point_count"] == 1

        assistant_text = "a" * 9000
        response = SimpleNamespace(
            output=SimpleNamespace(message=_mk_message(role="assistant", content=[_mk_content(text=assistant_text)])),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cache_read_input_tokens=20,
                cache_write_input_tokens=10,
            ),
            cost=SimpleNamespace(
                input_cost=0.001,
                output_cost=0.002,
                cached_read_cost=0.0001,
                cached_write_cost=0.0002,
                total_cost=0.0033,
            ),
            metrics=SimpleNamespace(latency_ms=500),
            stop_reason="end_turn",
            model_id="kimi-k2.5",
            response_metadata={"request_id": "abc"},
        )

        self.cb.on_converse_end(response)

        self.mock_generation.update.assert_called_once()
        update_kwargs = self.mock_generation.update.call_args.kwargs

        assert update_kwargs["output"]["message"]["content"][0]["text"] == assistant_text
        assert update_kwargs["usage_details"]["input_tokens"] == 100
        assert update_kwargs["usage_details"]["input"] == 100
        assert update_kwargs["usage_details"]["output_tokens"] == 50
        assert update_kwargs["usage_details"]["output"] == 50
        assert update_kwargs["usage_details"]["cache_read_input_tokens"] == 20
        assert update_kwargs["usage_details"]["cache_write_input_tokens"] == 10
        assert update_kwargs["cost_details"]["total_usd"] == 0.0033
        assert update_kwargs["cost_details"]["usd"] == 0.0033

        self.mock_generation.end.assert_called_once()
        assert self.cb._generation is None

    def test_tool_tracing_captures_full_input_and_output(self):
        self.cb._trace = self.mock_trace
        self.cb._trace_started_by_run = True

        tool_input = {"text": "hello" * 1000}
        tool_output = {"result": "ok" * 1000}

        self.cb.on_tool_start("send_message", tool_input, "tool-123")
        self.mock_trace.start_observation.assert_called_once()
        start_kwargs = self.mock_trace.start_observation.call_args.kwargs
        assert start_kwargs["as_type"] == "tool"
        assert start_kwargs["input"]["text"] == tool_input["text"]

        self.cb.on_tool_end("send_message", tool_input, "tool-123", tool_output, "success", 0.05)
        self.mock_tool.update.assert_called_once()
        update_kwargs = self.mock_tool.update.call_args.kwargs
        assert update_kwargs["output"]["result"] == tool_output["result"]
        assert update_kwargs["metadata"]["tool_input"]["text"] == tool_input["text"]
        self.mock_tool.end.assert_called_once()

    def test_on_run_end_flushes_and_closes_scopes(self):
        self.cb._trace = self.mock_trace
        self.cb._generation = self.mock_generation
        self.cb._tool_spans["t1"] = self.mock_tool
        self.cb._trace_attribute_scope = self.mock_scope

        result_text = "final" * 3000
        self.cb.on_run_end(MagicMock(), result_text)

        # Root should get the full output without truncation.
        trace_update_calls = self.mock_trace.update.call_args_list
        assert trace_update_calls
        final_update_kwargs = trace_update_calls[-1].kwargs
        assert final_update_kwargs["output"] == result_text

        self.mock_trace.end.assert_called_once()
        self.mock_langfuse.flush.assert_called_once()
        self.mock_scope.__exit__.assert_called_once()

    def test_standalone_converse_creates_and_closes_root_trace(self):
        user_text = "hello"
        converse = SimpleNamespace(
            model_id="test-model",
            messages=[_mk_message(role="user", content=[_mk_content(text=user_text)])],
            system=[SimpleNamespace(text="sys", guard_content=None, cache_point=None)],
            inference_config=None,
            request_metadata=None,
            performance_config=None,
            tool_config=None,
            guardrail_config=None,
            additional_model_request_fields=None,
            prompt_variables=None,
            additional_model_response_field_paths=None,
            region_name=None,
            max_iterations=None,
        )

        response = SimpleNamespace(
            output=SimpleNamespace(message=_mk_message(role="assistant", content=[_mk_content(text="done")])),
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=3,
                total_tokens=13,
                cache_read_input_tokens=0,
                cache_write_input_tokens=0,
            ),
            cost=SimpleNamespace(
                input_cost=0.001,
                output_cost=0.002,
                cached_read_cost=0.0,
                cached_write_cost=0.0,
                total_cost=0.003,
            ),
            metrics=SimpleNamespace(latency_ms=123),
            stop_reason="end_turn",
            model_id="test-model",
            response_metadata={},
            additional_model_response_fields=None,
            trace=None,
            performance_config=None,
        )

        self.cb.on_converse_start(converse)
        assert self.mock_langfuse.start_observation.call_count == 1
        assert self.mock_trace.start_observation.call_count == 1
        self.cb.on_converse_end(response)

        # Standalone converse should close the root trace immediately.
        self.mock_trace.end.assert_called()
        self.mock_langfuse.flush.assert_called()

    def test_on_converse_error_closes_generation_and_trace(self):
        self.cb._trace = self.mock_trace
        self.cb._generation = self.mock_generation
        self.cb._trace_started_by_run = True

        converse = SimpleNamespace(
            model_id="test-model",
            messages=[_mk_message(role="user", content=[_mk_content(text="hi")])],
            system=[SimpleNamespace(text="sys", guard_content=None, cache_point=None)],
            inference_config=None,
            tool_config=None,
            guardrail_config=None,
            additional_model_request_fields=None,
            prompt_variables=None,
            additional_model_response_field_paths=None,
            request_metadata=None,
            performance_config=None,
            region_name=None,
        )

        self.cb.on_converse_error(converse, RuntimeError("provider timeout"))

        self.mock_generation.update.assert_called_once()
        self.mock_generation.end.assert_called_once()
        self.mock_trace.update.assert_called()
        self.mock_trace.end.assert_called()
        self.mock_langfuse.flush.assert_called()

    def test_fallback_to_start_generation_when_start_observation_missing(self):
        owner = SimpleNamespace(
            start_observation=None,
            start_generation=MagicMock(return_value="gen"),
            start_span=MagicMock(),
        )

        result = self.cb._start_observation(owner, name="llm", as_type="generation", input={"x": 1})
        assert result == "gen"
        owner.start_generation.assert_called_once()

    def test_fallback_to_start_span_when_only_legacy_span_exists(self):
        owner = SimpleNamespace(
            start_observation=None,
            start_generation=None,
            start_span=MagicMock(return_value="span"),
        )

        result = self.cb._start_observation(owner, name="tool.call", as_type="tool", input={"x": 1})
        assert result == "span"

        kwargs = owner.start_span.call_args.kwargs
        assert kwargs["metadata"]["observation_type"] == "tool"


# --- Test BaseCallbackHandler new hooks have defaults ---


class TestBaseCallbackHooks:
    def test_new_hooks_have_default_implementations(self):
        class MinimalCallback(BaseCallbackHandler):
            def on_converse_start(self, converse):
                pass

            def on_converse_end(self, response):
                pass

        cb = MinimalCallback()
        cb.on_tool_start("tool", {}, "id")
        cb.on_tool_end("tool", {}, "id", "result", "success", 0.1)
        cb.on_run_start(MagicMock())
        cb.on_run_end(MagicMock(), "result")


# --- Test agent run() fires lifecycle hooks ---


class TestAgentRunHooks:
    def test_run_fires_on_run_start_and_end(self):
        from bedrock.converse import ConverseAgent

        mock_cb = MagicMock(spec=BaseCallbackHandler)
        mock_cb.on_run_start = MagicMock()
        mock_cb.on_run_end = MagicMock()
        mock_cb.on_converse_start = MagicMock()
        mock_cb.on_converse_end = MagicMock()
        mock_cb.on_tool_start = MagicMock()
        mock_cb.on_tool_end = MagicMock()

        agent = ConverseAgent(model_id="test", region_name="us-east-1")
        agent.callbacks.append(mock_cb)

        mock_response = MagicMock()
        text_content = MagicMock()
        text_content.text = "Hello"
        text_content.tool_use = None
        mock_response.output.message.content = [text_content]
        agent._get_response = MagicMock(return_value=mock_response)

        result = agent.run("test input")

        mock_cb.on_run_start.assert_called_once_with(agent)
        mock_cb.on_run_end.assert_called_once()
        assert mock_cb.on_run_end.call_args[0][1] == "Hello"
