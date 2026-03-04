"""Langfuse tracing callback for Bedrock SDK (v3 SDK).

Reads config from env vars:
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST (optional, defaults to cloud)

Usage:
    from bedrock.langfuse_callback import LangfuseCallback

    cb = LangfuseCallback(user_id="123", session_id="abc", tags=["heartbeat"])
    agent.callbacks.append(cb)
    agent.run(prompt)
"""

import logging
import time

from .bases import BaseCallbackHandler

logger = logging.getLogger(__name__)


def _get_langfuse():
    try:
        from langfuse import Langfuse
        return Langfuse()
    except ImportError:
        logger.warning("langfuse package not installed — tracing disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to init Langfuse: {e}")
        return None


class LangfuseCallback(BaseCallbackHandler):
    """Traces agent runs, LLM calls, and tool executions to Langfuse v3."""

    def __init__(self, user_id: str = None, session_id: str = None, tags: list = None, metadata: dict = None):
        self.user_id = user_id
        self.session_id = session_id
        self.tags = tags or []
        self.metadata = metadata or {}
        self._langfuse = _get_langfuse()
        self._trace = None
        self._generation = None
        self._generation_start = None
        self._tool_spans = {}

    @property
    def enabled(self):
        return self._langfuse is not None

    # --- Run lifecycle ---

    def on_run_start(self, agent):
        if not self.enabled:
            return
        self._trace = self._langfuse.start_span(
            name="agent.run",
            input=self.metadata,
            metadata={**self.metadata, 'model_id': getattr(agent, 'model_id', None)},
        )
        self._trace.update_trace(
            user_id=self.user_id,
            session_id=self.session_id,
            tags=self.tags,
        )

    def on_run_end(self, agent, result):
        if not self.enabled or not self._trace:
            return
        self._trace.update(output=str(result)[:2000] if result else None)
        self._trace.end()
        try:
            self._langfuse.flush()
        except Exception as e:
            logger.warning(f"Langfuse flush failed: {e}")

    # --- LLM call lifecycle ---

    def on_converse_start(self, converse):
        if not self.enabled or not self._trace:
            return
        self._generation_start = time.time()
        self._generation = self._trace.start_generation(
            name="llm",
            model=getattr(converse, 'model_id', None),
        )

    def on_converse_end(self, response):
        if not self.enabled or not self._generation:
            return
        usage = {}
        if response.usage:
            usage = {
                'input': response.usage.input_tokens,
                'output': response.usage.output_tokens,
                'total': response.usage.input_tokens + response.usage.output_tokens,
            }
            if response.usage.cache_read_input_tokens:
                usage['cache_read'] = response.usage.cache_read_input_tokens
        cost = {}
        if response.cost:
            cost = {'total_usd': round(response.cost.total, 6)}
        self._generation.update(
            usage_details=usage or None,
            cost_details=cost or None,
            metadata={
                'latency_ms': response.metrics.latency_ms if response.metrics else None,
                'stop_reason': response.stop_reason,
            },
        )
        self._generation.end()
        self._generation = None

    # --- Tool lifecycle ---

    def on_tool_start(self, tool_name: str, tool_input: dict, tool_use_id: str):
        if not self.enabled or not self._trace:
            return
        span = self._trace.start_span(
            name=tool_name,
            input=tool_input,
            metadata={'tool_use_id': tool_use_id},
        )
        self._tool_spans[tool_use_id] = span

    def on_tool_end(self, tool_name: str, tool_input: dict, tool_use_id: str, result, status: str, duration: float):
        if not self.enabled:
            return
        span = self._tool_spans.pop(tool_use_id, None)
        if span:
            span.update(
                output=str(result)[:2000],
                metadata={'status': status, 'duration_s': round(duration, 3)},
            )
            span.end()
