"""Langfuse tracing callback for Bedrock SDK.

Reads config from env vars:
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_BASE_URL / LANGFUSE_HOST (optional, defaults to cloud)

Usage:
    from bedrock.langfuse_callback import LangfuseCallback

    cb = LangfuseCallback(user_id="123", session_id="abc", tags=["heartbeat"])
    agent.callbacks.append(cb)
    agent.run(prompt)
"""

import inspect
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .bases import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Full traceability defaults: capture complete text and full message/tool payloads.
_MAX_TEXT: Optional[int] = None
_MAX_MESSAGES: Optional[int] = None
_MAX_ITEMS: Optional[int] = None


def _get_langfuse():
    try:
        from langfuse import Langfuse

        return Langfuse()
    except ImportError:
        logger.warning("langfuse package not installed - tracing disabled")
        return None
    except Exception as exc:
        logger.warning("Failed to init Langfuse: %s", exc)
        return None


def _truncate_text(value: Any, limit: Optional[int] = _MAX_TEXT) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    if limit is None:
        return text
    if len(text) <= limit:
        return text
    return text[:limit]


def _take(values: Any, limit: Optional[int]):
    if values is None:
        return []
    items = list(values)
    if limit is None:
        return items
    return items[:limit]


def _safe_call_with_supported_kwargs(func, **kwargs):
    payload = {k: v for k, v in kwargs.items() if v is not None}

    try:
        signature = inspect.signature(func)
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            payload = {k: v for k, v in payload.items() if k in signature.parameters}
    except (TypeError, ValueError):
        # Builtins and some C-extension callables may not have signatures.
        pass

    return func(**payload)


class LangfuseCallback(BaseCallbackHandler):
    """Traces agent runs, LLM calls, and tool executions to Langfuse."""

    def __init__(self, user_id: str = None, session_id: str = None, tags: list = None, metadata: dict = None):
        self.user_id = user_id
        self.session_id = session_id
        self.tags = tags or []
        self.metadata = metadata or {}

        self._langfuse = _get_langfuse()

        self._trace = None
        self._generation = None
        self._generation_start_time = None
        self._generation_completion_start_time = None
        self._tool_spans = {}
        self._trace_attribute_scope = None
        self._trace_started_by_run = False

    @property
    def enabled(self):
        return self._langfuse is not None

    @staticmethod
    def _to_primitive(value: Any, depth: int = 0):
        if value is None:
            return None
        if depth > 8:
            return _truncate_text(value, None)
        if isinstance(value, (str, int, float, bool)):
            return _truncate_text(value, None) if isinstance(value, str) else value
        if isinstance(value, bytes):
            return f"<bytes:{len(value)}>"
        if isinstance(value, list):
            return [LangfuseCallback._to_primitive(v, depth + 1) for v in _take(value, _MAX_ITEMS)]
        if isinstance(value, tuple):
            return [LangfuseCallback._to_primitive(v, depth + 1) for v in _take(value, _MAX_ITEMS)]
        if isinstance(value, dict):
            compact = {}
            for k, v in _take(value.items(), _MAX_ITEMS):
                compact[_truncate_text(k, 120) if isinstance(k, str) else str(k)] = LangfuseCallback._to_primitive(
                    v, depth + 1
                )
            return compact
        if hasattr(value, "model_dump") and callable(value.model_dump):
            return LangfuseCallback._to_primitive(value.model_dump(), depth + 1)
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return LangfuseCallback._to_primitive(value.to_dict(), depth + 1)
        return _truncate_text(value, None)

    @classmethod
    def _summarize_tool_result_content(cls, content: Any) -> Dict[str, Any]:
        summary = {}

        text = getattr(content, "text", None)
        if text is not None:
            summary["text"] = _truncate_text(text, None)

        json_data = getattr(content, "json", None)
        if json_data is not None:
            summary["json"] = cls._to_primitive(json_data)

        if getattr(content, "image", None) is not None:
            image = content.image
            source = getattr(image, "source", None)
            image_bytes = getattr(source, "bytes", b"") if source else b""
            summary["image"] = {
                "format": getattr(image, "format", None),
                "bytes": len(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else None,
            }

        if getattr(content, "document", None) is not None:
            document = content.document
            source = getattr(document, "source", None)
            doc_bytes = getattr(source, "bytes", b"") if source else b""
            summary["document"] = {
                "name": getattr(document, "name", None),
                "format": getattr(document, "format", None),
                "bytes": len(doc_bytes) if isinstance(doc_bytes, (bytes, bytearray)) else None,
            }

        if getattr(content, "video", None) is not None:
            video = content.video
            source = getattr(video, "source", None)
            video_bytes = getattr(source, "bytes", b"") if source else b""
            summary["video"] = {
                "format": getattr(video, "format", None),
                "bytes": len(video_bytes) if isinstance(video_bytes, (bytes, bytearray)) else None,
            }

        return summary

    @classmethod
    def _summarize_message_content(cls, content: Any) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        text = getattr(content, "text", None)
        if text is not None:
            summary["text"] = _truncate_text(text, None)

        tool_use = getattr(content, "tool_use", None)
        if tool_use is not None:
            summary["tool_use"] = {
                "tool_use_id": getattr(tool_use, "tool_use_id", None),
                "name": getattr(tool_use, "name", None),
                "input": cls._to_primitive(getattr(tool_use, "input", None)),
            }

        tool_result = getattr(content, "tool_result", None)
        if tool_result is not None:
            result_content = []
            for item in _take(getattr(tool_result, "content", []), _MAX_ITEMS):
                result_content.append(cls._summarize_tool_result_content(item))
            summary["tool_result"] = {
                "tool_use_id": getattr(tool_result, "tool_use_id", None),
                "status": getattr(tool_result, "status", None),
                "content": result_content,
            }

        reasoning_content = getattr(content, "reasoning_content", None)
        if reasoning_content is not None:
            reasoning_text = getattr(reasoning_content, "reasoning_text", None)
            text_value = getattr(reasoning_text, "text", None) if reasoning_text else None
            summary["reasoning"] = _truncate_text(text_value, None)

        if getattr(content, "image", None) is not None and "image" not in summary:
            image = content.image
            source = getattr(image, "source", None)
            image_bytes = getattr(source, "bytes", b"") if source else b""
            summary["image"] = {
                "format": getattr(image, "format", None),
                "bytes": len(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else None,
            }

        if getattr(content, "document", None) is not None and "document" not in summary:
            document = content.document
            source = getattr(document, "source", None)
            doc_bytes = getattr(source, "bytes", b"") if source else b""
            summary["document"] = {
                "name": getattr(document, "name", None),
                "format": getattr(document, "format", None),
                "bytes": len(doc_bytes) if isinstance(doc_bytes, (bytes, bytearray)) else None,
            }

        if getattr(content, "video", None) is not None and "video" not in summary:
            video = content.video
            source = getattr(video, "source", None)
            video_bytes = getattr(source, "bytes", b"") if source else b""
            summary["video"] = {
                "format": getattr(video, "format", None),
                "bytes": len(video_bytes) if isinstance(video_bytes, (bytes, bytearray)) else None,
            }

        if not summary:
            return {"type": "empty"}

        return summary

    @classmethod
    def _summarize_messages(cls, messages: List[Any]) -> Dict[str, Any]:
        count = len(messages)
        tail = messages if _MAX_MESSAGES is None else messages[-_MAX_MESSAGES:]
        serialized = []

        for message in tail:
            contents = []
            for content in _take(getattr(message, "content", []), _MAX_ITEMS):
                contents.append(cls._summarize_message_content(content))

            serialized.append(
                {
                    "role": getattr(message, "role", None),
                    "content": contents,
                }
            )

        return {
            "message_count": count,
            "truncated": False if _MAX_MESSAGES is None else count > _MAX_MESSAGES,
            "messages": serialized,
        }

    @classmethod
    def _summarize_system_prompts(cls, system_prompts: List[Any]) -> Dict[str, Any]:
        count = len(system_prompts)
        serialized = []

        for item in _take(system_prompts, _MAX_ITEMS):
            entry: Dict[str, Any] = {}

            text = getattr(item, "text", None)
            if text is not None:
                entry["text"] = _truncate_text(text, None)

            guard_content = getattr(item, "guard_content", None)
            if guard_content is not None:
                if hasattr(guard_content, "to_dict") and callable(guard_content.to_dict):
                    entry["guard_content"] = cls._to_primitive(guard_content.to_dict())
                else:
                    entry["guard_content"] = cls._to_primitive(guard_content)

            cache_point = getattr(item, "cache_point", None)
            if cache_point is not None:
                if hasattr(cache_point, "to_dict") and callable(cache_point.to_dict):
                    entry["cache_point"] = cls._to_primitive(cache_point.to_dict())
                else:
                    entry["cache_point"] = cls._to_primitive(cache_point)

            if not entry:
                if hasattr(item, "to_dict") and callable(item.to_dict):
                    entry = cls._to_primitive(item.to_dict())
                else:
                    entry = {"value": cls._to_primitive(item)}

            serialized.append(entry)

        return {
            "count": count,
            "truncated": False if _MAX_ITEMS is None else count > _MAX_ITEMS,
            "prompts": serialized,
        }

    @classmethod
    def _serialize_obj(cls, value: Any):
        if value is None:
            return None
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return cls._to_primitive(value.to_dict())
        if hasattr(value, "model_dump") and callable(value.model_dump):
            return cls._to_primitive(value.model_dump())
        return cls._to_primitive(value)

    @staticmethod
    def _build_trace_attribute_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        result: Dict[str, str] = {}

        for key, value in metadata.items():
            key_text = str(key)
            if len(key_text) > 200:
                continue
            value_text = _truncate_text(value, 200)
            if value_text is None:
                continue
            result[key_text] = value_text

        return result

    def _build_run_input(self, agent: Any) -> Dict[str, Any]:
        messages = getattr(agent, "messages", None) or []
        system_prompts = getattr(agent, "system", None) or []
        return {
            "messages": self._summarize_messages(messages),
            "system": self._summarize_system_prompts(system_prompts),
        }

    def _build_run_metadata(self, agent: Any) -> Dict[str, Any]:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "model_id": getattr(agent, "model_id", None),
                "max_iterations": getattr(agent, "max_iterations", None),
                "callback": "LangfuseCallback",
                "sdk": "bedrock-converse",
            }
        )
        return metadata

    def _build_generation_input(self, converse: Any) -> Dict[str, Any]:
        messages = getattr(converse, "messages", None) or []
        system_prompts = getattr(converse, "system", None) or []
        input_payload = {
            "model_id": getattr(converse, "model_id", None),
            "messages": self._summarize_messages(messages),
            "system": self._summarize_system_prompts(system_prompts),
            "system_prompt_count": len(system_prompts),
            "tool_config": self._serialize_obj(getattr(converse, "tool_config", None)),
            "guardrail_config": self._serialize_obj(getattr(converse, "guardrail_config", None)),
            "additional_model_request_fields": self._serialize_obj(
                getattr(converse, "additional_model_request_fields", None)
            ),
            "prompt_variables": self._serialize_obj(getattr(converse, "prompt_variables", None)),
            "additional_model_response_field_paths": self._to_primitive(
                getattr(converse, "additional_model_response_field_paths", None)
            ),
            "request_metadata": self._to_primitive(getattr(converse, "request_metadata", None)),
            "performance_config": self._serialize_obj(getattr(converse, "performance_config", None)),
            "region_name": getattr(converse, "region_name", None),
        }

        inference_config = getattr(converse, "inference_config", None)
        if inference_config is not None:
            input_payload["inference_config"] = self._to_primitive(
                {
                    "max_tokens": getattr(inference_config, "max_tokens", None),
                    "temperature": getattr(inference_config, "temperature", None),
                    "top_p": getattr(inference_config, "top_p", None),
                    "stop_sequences": getattr(inference_config, "stop_sequences", None),
                }
            )

        return input_payload

    def _build_model_parameters(self, converse: Any) -> Optional[Dict[str, Any]]:
        inference_config = getattr(converse, "inference_config", None)
        if inference_config is None:
            return None

        params = {
            "max_tokens": getattr(inference_config, "max_tokens", None),
            "temperature": getattr(inference_config, "temperature", None),
            "top_p": getattr(inference_config, "top_p", None),
            "stop_sequences": getattr(inference_config, "stop_sequences", None),
        }

        filtered = {k: v for k, v in params.items() if v is not None}
        return filtered or None

    def _build_generation_metadata(self, converse: Any) -> Dict[str, Any]:
        metadata = {
            "request_metadata": self._to_primitive(getattr(converse, "request_metadata", None)),
            "performance_config": self._to_primitive(getattr(converse, "performance_config", None)),
            "tool_count": len(getattr(getattr(converse, "tool_config", None), "tools", None) or []),
            "region_name": getattr(converse, "region_name", None),
        }
        return {k: v for k, v in metadata.items() if v is not None}

    def _build_generation_output(self, response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        metrics = getattr(response, "metrics", None)
        output = {
            "model_id": getattr(response, "model_id", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "message": {
                "role": getattr(getattr(response, "output", None), "message", None)
                and getattr(response.output.message, "role", None),
                "content": [],
            },
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_write_input_tokens": getattr(usage, "cache_write_input_tokens", None),
            }
            if usage is not None
            else None,
            "metrics": {"latency_ms": getattr(metrics, "latency_ms", None)} if metrics is not None else None,
            "additional_model_response_fields": self._to_primitive(
                getattr(response, "additional_model_response_fields", None)
            ),
            "trace": self._serialize_obj(getattr(response, "trace", None)),
            "performance_config": self._serialize_obj(getattr(response, "performance_config", None)),
            "response_metadata": self._to_primitive(getattr(response, "response_metadata", None)),
        }

        message = getattr(getattr(response, "output", None), "message", None)
        for content in _take(getattr(message, "content", []), _MAX_ITEMS):
            output["message"]["content"].append(self._summarize_message_content(content))

        return {k: v for k, v in output.items() if v is not None}

    @staticmethod
    def _build_usage_details(response: Any) -> Optional[Dict[str, int]]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        cache_write = getattr(usage, "cache_write_input_tokens", None)

        usage_details: Dict[str, int] = {}

        if input_tokens is not None:
            usage_details["input_tokens"] = int(input_tokens)
            usage_details["input"] = int(input_tokens)
            usage_details["prompt_tokens"] = int(input_tokens)
        if output_tokens is not None:
            usage_details["output_tokens"] = int(output_tokens)
            usage_details["output"] = int(output_tokens)
            usage_details["completion_tokens"] = int(output_tokens)
        if total_tokens is not None:
            usage_details["total_tokens"] = int(total_tokens)
            usage_details["total"] = int(total_tokens)
        if cache_read is not None:
            usage_details["cache_read_input_tokens"] = int(cache_read)
            usage_details["cache_read"] = int(cache_read)
        if cache_write is not None:
            usage_details["cache_write_input_tokens"] = int(cache_write)
            usage_details["cache_write"] = int(cache_write)

        return usage_details or None

    @staticmethod
    def _build_cost_details(response: Any) -> Optional[Dict[str, float]]:
        cost = getattr(response, "cost", None)
        if cost is None:
            return None

        cost_map = {
            "input_cost_usd": "input_cost",
            "output_cost_usd": "output_cost",
            "cached_read_cost_usd": "cached_read_cost",
            "cached_write_cost_usd": "cached_write_cost",
            "total_usd": "total_cost",
        }

        details: Dict[str, float] = {}
        for output_key, attr in cost_map.items():
            try:
                value = getattr(cost, attr)
            except Exception:
                continue

            if value is None:
                continue

            try:
                details[output_key] = round(float(value), 8)
            except (TypeError, ValueError):
                continue

        if not details:
            total = getattr(cost, "total", None)
            if total is not None:
                try:
                    details["total_usd"] = round(float(total), 8)
                except (TypeError, ValueError):
                    pass
        if "total_usd" in details:
            details["usd"] = details["total_usd"]

        return details or None

    @staticmethod
    def _safe_end(observation: Any):
        if observation is None:
            return
        end_func = getattr(observation, "end", None)
        if callable(end_func):
            try:
                end_func()
            except Exception as exc:
                logger.warning("Langfuse end failed: %s", exc)

    def _update_observation(self, observation: Any, **kwargs):
        if observation is None:
            return

        update = getattr(observation, "update", None)
        if not callable(update):
            return

        try:
            _safe_call_with_supported_kwargs(update, **kwargs)
        except Exception as exc:
            logger.warning("Langfuse update failed: %s", exc)

    def _start_observation(self, owner: Any, name: str, as_type: str, **kwargs):
        if owner is None:
            return None

        start_observation = getattr(owner, "start_observation", None)
        if callable(start_observation):
            try:
                return _safe_call_with_supported_kwargs(start_observation, name=name, as_type=as_type, **kwargs)
            except Exception as exc:
                logger.warning("Langfuse start_observation failed: %s", exc)

        # Fallback to pre-v3 helpers when available.
        if as_type == "generation":
            start_generation = getattr(owner, "start_generation", None)
            if callable(start_generation):
                try:
                    return _safe_call_with_supported_kwargs(start_generation, name=name, **kwargs)
                except Exception as exc:
                    logger.warning("Langfuse start_generation failed: %s", exc)

        start_span = getattr(owner, "start_span", None)
        if callable(start_span):
            metadata = kwargs.get("metadata") or {}
            if as_type != "span":
                metadata = {**metadata, "observation_type": as_type}

            try:
                return _safe_call_with_supported_kwargs(
                    start_span,
                    name=name,
                    input=kwargs.get("input"),
                    output=kwargs.get("output"),
                    metadata=metadata,
                    version=kwargs.get("version"),
                    level=kwargs.get("level"),
                    status_message=kwargs.get("status_message"),
                )
            except Exception as exc:
                logger.warning("Langfuse start_span failed: %s", exc)

        return None

    def _start_trace_attribute_propagation(self, run_metadata: Dict[str, Any]):
        if not self.enabled:
            return

        propagate = getattr(self._langfuse, "propagate_attributes", None)
        if not callable(propagate):
            return

        kwargs: Dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tags": self.tags or None,
        }

        trace_metadata = self._build_trace_attribute_metadata(run_metadata)
        if trace_metadata:
            kwargs["metadata"] = trace_metadata

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not kwargs:
            return

        try:
            scope = _safe_call_with_supported_kwargs(propagate, **kwargs)
            enter = getattr(scope, "__enter__", None)
            if callable(enter):
                enter()
                self._trace_attribute_scope = scope
        except Exception as exc:
            logger.warning("Langfuse propagate_attributes failed: %s", exc)

    def _close_trace_attribute_propagation(self):
        if self._trace_attribute_scope is None:
            return

        exit_func = getattr(self._trace_attribute_scope, "__exit__", None)
        if callable(exit_func):
            try:
                exit_func(None, None, None)
            except Exception as exc:
                logger.warning("Langfuse propagate_attributes scope close failed: %s", exc)

        self._trace_attribute_scope = None

    def _finalize_dangling_observations(self):
        if self._generation is not None:
            self._update_observation(
                self._generation,
                level="WARNING",
                status_message="Generation closed during run finalization",
            )
            self._safe_end(self._generation)
            self._generation = None

        for tool_use_id, span in list(self._tool_spans.items()):
            self._update_observation(
                span,
                metadata={"status": "aborted", "tool_use_id": tool_use_id},
                level="WARNING",
                status_message="Tool span closed during run finalization",
            )
            self._safe_end(span)
            self._tool_spans.pop(tool_use_id, None)

    # --- Run lifecycle ---

    def on_run_start(self, agent):
        if not self.enabled:
            return

        self._close_trace_attribute_propagation()
        self._trace_started_by_run = True

        run_input = self._build_run_input(agent)
        run_metadata = self._build_run_metadata(agent)
        self._start_trace_attribute_propagation(run_metadata)

        self._trace = self._start_observation(
            self._langfuse,
            name="agent.run",
            as_type="agent",
            input=run_input,
            metadata=run_metadata,
        )
        if not self._trace:
            self._close_trace_attribute_propagation()
            return

        if self._trace and hasattr(self._trace, "update_trace"):
            trace_kwargs: Dict[str, Any] = {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "tags": self.tags or None,
            }

            trace_metadata = self._build_trace_attribute_metadata(run_metadata)
            if trace_metadata:
                trace_kwargs["metadata"] = trace_metadata

            try:
                _safe_call_with_supported_kwargs(self._trace.update_trace, **trace_kwargs)
            except Exception as exc:
                logger.warning("Langfuse trace attribute update failed: %s", exc)

    def on_run_end(self, agent, result):
        if not self.enabled:
            return

        if not self._trace:
            self._close_trace_attribute_propagation()
            self._trace_started_by_run = False
            return

        self._finalize_dangling_observations()
        run_output = {
            "result": self._to_primitive(result),
            "messages": self._summarize_messages(getattr(agent, "messages", None) or []),
            "system": self._summarize_system_prompts(getattr(agent, "system", None) or []),
        }

        self._update_observation(
            self._trace,
            output=run_output,
            metadata={"result_type": type(result).__name__ if result is not None else None},
        )
        self._safe_end(self._trace)

        try:
            self._langfuse.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)
        finally:
            self._close_trace_attribute_propagation()
            self._trace = None
            self._generation = None
            self._generation_start_time = None
            self._generation_completion_start_time = None
            self._tool_spans.clear()
            self._trace_started_by_run = False

    # --- LLM call lifecycle ---

    def on_converse_start(self, converse):
        if not self.enabled:
            return

        if not self._trace:
            self._close_trace_attribute_propagation()
            self._trace_started_by_run = False
            run_metadata = self._build_run_metadata(converse)
            self._start_trace_attribute_propagation(run_metadata)
            self._trace = self._start_observation(
                self._langfuse,
                name="converse.run",
                as_type="chain",
                input=self._build_generation_input(converse),
                metadata=run_metadata,
            )
            if self._trace and hasattr(self._trace, "update_trace"):
                trace_kwargs: Dict[str, Any] = {
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "tags": self.tags or None,
                }
                trace_metadata = self._build_trace_attribute_metadata(run_metadata)
                if trace_metadata:
                    trace_kwargs["metadata"] = trace_metadata
                try:
                    _safe_call_with_supported_kwargs(self._trace.update_trace, **trace_kwargs)
                except Exception as exc:
                    logger.warning("Langfuse trace attribute update failed: %s", exc)

        if not self._trace:
            return

        self._generation_start_time = time.time()
        self._generation_completion_start_time = datetime.now(timezone.utc)

        self._generation = self._start_observation(
            self._trace,
            name="llm.converse",
            as_type="generation",
            model=getattr(converse, "model_id", None),
            model_parameters=self._build_model_parameters(converse),
            input=self._build_generation_input(converse),
            metadata=self._build_generation_metadata(converse),
        )

    def on_converse_end(self, response):
        if not self.enabled or not self._generation:
            return

        elapsed_ms = None
        if self._generation_start_time is not None:
            elapsed_ms = int((time.time() - self._generation_start_time) * 1000)

        metadata = {
            "latency_ms": getattr(getattr(response, "metrics", None), "latency_ms", None),
            "duration_ms": elapsed_ms,
            "stop_reason": getattr(response, "stop_reason", None),
            "model_id": getattr(response, "model_id", None),
            "response_metadata": self._to_primitive(getattr(response, "response_metadata", None)),
        }

        level = None
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "error":
            level = "ERROR"

        self._update_observation(
            self._generation,
            output=self._build_generation_output(response),
            usage_details=self._build_usage_details(response),
            cost_details=self._build_cost_details(response),
            completion_start_time=self._generation_completion_start_time,
            metadata={k: v for k, v in metadata.items() if v is not None},
            level=level,
        )

        self._safe_end(self._generation)
        self._generation = None
        self._generation_start_time = None
        self._generation_completion_start_time = None

        if self._trace and not self._trace_started_by_run:
            self._update_observation(
                self._trace,
                output={"response": self._build_generation_output(response)},
            )
            self._safe_end(self._trace)
            try:
                self._langfuse.flush()
            except Exception as exc:
                logger.warning("Langfuse flush failed: %s", exc)
            finally:
                self._close_trace_attribute_propagation()
                self._trace = None

    def on_converse_error(self, converse, error: Exception):
        if not self.enabled:
            return

        error_text = _truncate_text(error, None)
        short_error_text = _truncate_text(error, 600)
        error_type = type(error).__name__

        if self._generation is not None:
            self._update_observation(
                self._generation,
                level="ERROR",
                status_message=short_error_text,
                metadata={"error_type": error_type, "error": error_text},
            )
            self._safe_end(self._generation)
            self._generation = None
            self._generation_start_time = None
            self._generation_completion_start_time = None

        if self._trace is not None:
            self._update_observation(
                self._trace,
                output={
                    "error": error_text,
                    "error_type": error_type,
                    "failed_request": self._build_generation_input(converse),
                },
                level="ERROR",
                status_message=short_error_text,
                metadata={"error_type": error_type},
            )
            self._safe_end(self._trace)

            try:
                self._langfuse.flush()
            except Exception as flush_exc:
                logger.warning("Langfuse flush failed: %s", flush_exc)
            finally:
                self._close_trace_attribute_propagation()
                self._trace = None
                self._tool_spans.clear()
                self._trace_started_by_run = False

    # --- Tool lifecycle ---

    def on_tool_start(self, tool_name: str, tool_input: dict, tool_use_id: str):
        if not self.enabled or not self._trace:
            return

        span = self._start_observation(
            self._trace,
            name=f"tool.{tool_name}",
            as_type="tool",
            input=self._to_primitive(tool_input),
            metadata={"tool_use_id": tool_use_id, "tool_name": tool_name},
        )

        if span is not None:
            self._tool_spans[tool_use_id] = span

    def on_tool_end(self, tool_name: str, tool_input: dict, tool_use_id: str, result, status: str, duration: float):
        if not self.enabled:
            return

        span = self._tool_spans.pop(tool_use_id, None)
        if span is None:
            return

        metadata = {
            "status": status,
            "duration_ms": int(duration * 1000),
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "tool_input": self._to_primitive(tool_input),
        }

        level = "ERROR" if status == "error" else None
        status_message = _truncate_text(result, None) if status == "error" else None

        self._update_observation(
            span,
            output=self._to_primitive(result),
            metadata=metadata,
            level=level,
            status_message=status_message,
        )
        self._safe_end(span)
