import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from openai import OpenAI, AsyncOpenAI

from .converse import (Converse, ConverseAgent, StructuredConverse, ConverseResponse, ConverseOutput,
                       Message, MessageContent, ToolUse, TokenUsage, ConverseMetrics,
                       ReasoningContent, ReasoningText, AdditionalModelRequestFields, ThinkingConfig)

logger = logging.getLogger(__name__)

STOP_REASON_MAP = {'stop': 'end_turn', 'length': 'max_tokens', 'tool_calls': 'tool_use'}


def new_tool_use_id():
    return f'tooluse_{uuid.uuid4().hex}'


def token_usage_from_openai(usage):
    if not usage:
        return TokenUsage()
    details = getattr(usage, 'prompt_tokens_details', None) or getattr(usage, 'input_tokens_details', None)
    cache_read = (getattr(details, 'cached_tokens', 0) or 0) if details else 0
    input_tokens = getattr(usage, 'prompt_tokens', None)
    input_tokens = getattr(usage, 'input_tokens', 0) if input_tokens is None else input_tokens
    output_tokens = getattr(usage, 'completion_tokens', None)
    output_tokens = getattr(usage, 'output_tokens', 0) if output_tokens is None else output_tokens
    return TokenUsage(input_tokens=(input_tokens or 0) - cache_read, output_tokens=output_tokens or 0,
                      total_tokens=getattr(usage, 'total_tokens', 0) or 0, cache_read_input_tokens=cache_read)


def strict_schema(schema):
    if isinstance(schema, list):
        return [strict_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: strict_schema(value) for key, value in schema.items()}
    if result.get('type') == 'object' and 'properties' in result:
        result['additionalProperties'] = False
        result['required'] = list(result['properties'])
    return result


class _MantleTransport:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_mode: str = 'chat_completions'
    extra_params: Optional[dict] = None
    session_affinity_header: Optional[str] = None

    def __post_init__(self):
        getattr(super(), '__post_init__', lambda: None)()
        self._TO_DICT_EXCLUSIONS.extend(['api_key', 'base_url', 'api_mode', 'extra_params', 'session_affinity_header'])

    @property
    def strict_tool_names(self):
        return set()

    @property
    def _mantle_base_url(self):
        if self.base_url:
            return self.base_url
        if endpoint := os.environ.get('MANTLE_ENDPOINT'):
            return endpoint
        region = self.region_name or self.session.region_name
        return f'https://bedrock-mantle.{region}.api.aws/v1'

    def _get_client(self, openai_class):
        if self.api_key:
            return openai_class(api_key=self.api_key, base_url=self._mantle_base_url)
        if api_key := os.environ.get('MANTLE_API_KEY'):
            return openai_class(api_key=api_key, base_url=self._mantle_base_url)
        return openai_class(base_url=self._mantle_base_url)

    @cached_property
    def openai_client(self) -> OpenAI:
        return self._get_client(OpenAI)

    @cached_property
    def async_openai_client(self) -> AsyncOpenAI:
        return self._get_client(AsyncOpenAI)

    def _build_tool_params(self, params):
        if not self.tool_config:
            return
        tools = []
        for t in self.tool_config.tools:
            if not t.tool_spec:
                continue
            function = {'name': t.tool_spec.name, 'description': t.tool_spec.description,
                        'parameters': t.tool_spec.input_schema.get('json', {})}
            if t.tool_spec.name in self.strict_tool_names:
                function.update(parameters=strict_schema(function['parameters']), strict=True)
            tools.append({'type': 'function', 'function': function})
        if tools:
            params['tools'] = tools
        if self.tool_config.tool_choice:
            tc = self.tool_config.tool_choice
            if tc.tool:
                params['tool_choice'] = {'type': 'function', 'function': {'name': tc.tool.name}}
            elif tc.any:
                params['tool_choice'] = 'required'
            elif tc.auto:
                params['tool_choice'] = 'auto'

    def _build_inference_params(self, params):
        if not self.inference_config:
            return
        ic = self.inference_config
        if ic.max_tokens is not None: params['max_tokens'] = ic.max_tokens
        if ic.temperature is not None: params['temperature'] = ic.temperature
        if ic.top_p is not None: params['top_p'] = ic.top_p
        if ic.stop_sequences: params['stop'] = ic.stop_sequences

    @property
    def uses_responses_api(self):
        return self.api_mode == 'responses'

    @property
    def reasoning_effort(self):
        if not (self.additional_model_request_fields and self.additional_model_request_fields.thinking
                and self.additional_model_request_fields.thinking.type == 'enabled'):
            return None
        budget = self.additional_model_request_fields.thinking.budget_tokens
        if isinstance(budget, str):
            return budget
        if budget <= 2048: return 'low'
        if budget <= 8192: return 'medium'
        return 'high'

    def _build_thinking_params(self, params):
        if effort := self.reasoning_effort:
            params['reasoning_effort'] = effort

    def payload_system_text(self, payload):
        return '\n'.join(s['text'] for s in payload.get('system', []) if s.get('text'))

    def payload_messages(self, payload):
        return [Message.from_dict(m) for m in payload.get('messages', [])]

    def _build_shared_params(self) -> dict:
        params = {'model': self.model_id}
        self._build_tool_params(params)
        self._build_inference_params(params)
        self._build_thinking_params(params)
        if self.cache_key:
            if self.session_affinity_header:
                params['extra_headers'] = {self.session_affinity_header: self.cache_key}
            else:
                params['prompt_cache_key'] = self.cache_key
        if self.extra_params:
            params['extra_body'] = dict(self.extra_params)
        return params

    def _build_params(self, messages=None) -> dict:
        payload = self.build_payload(messages)
        msgs = [{'role': 'system', 'content': text}] if (text := self.payload_system_text(payload)) else []
        for msg in self.payload_messages(payload):
            msgs.extend(self._convert_message(msg))
        return {**self._build_shared_params(), 'messages': msgs}

    def _build_responses_params(self, messages=None):
        payload = self.build_payload(messages)
        params = self._build_shared_params()
        response_params = {'model': params['model'], 'input': self._responses_input(payload), 'store': False,
                           'include': ['reasoning.encrypted_content']}
        if tools := params.get('tools'):
            response_params['tools'] = self._responses_tools(tools)
        if tool_choice := self._responses_tool_choice(params.get('tool_choice')):
            response_params['tool_choice'] = tool_choice
        if max_tokens := params.get('max_tokens'):
            response_params['max_output_tokens'] = max_tokens
        if params.get('temperature') is not None:
            response_params['temperature'] = params['temperature']
        if params.get('top_p') is not None:
            response_params['top_p'] = params['top_p']
        if effort := params.get('reasoning_effort'):
            response_params['reasoning'] = {'effort': effort, 'summary': 'auto'}
        if cache_key := params.get('prompt_cache_key'):
            response_params['prompt_cache_key'] = cache_key
        if extra_body := params.get('extra_body'):
            response_params['extra_body'] = extra_body
        if extra_headers := params.get('extra_headers'):
            response_params['extra_headers'] = extra_headers
        return response_params

    def _responses_tools(self, tools):
        return [{'type': 'function', 'name': tool['function']['name'],
                 'description': tool['function'].get('description'), 'parameters': tool['function'].get('parameters') or {},
                 'strict': tool['function'].get('strict', False)} for tool in tools]

    def _responses_tool_choice(self, tool_choice):
        if isinstance(tool_choice, str):
            return tool_choice
        if tool_choice and tool_choice.get('type') == 'function':
            return {'type': 'function', 'name': tool_choice['function']['name']}
        return None

    def _responses_input(self, payload):
        items = [{'type': 'message', 'role': 'system', 'content': text}] if (text := self.payload_system_text(payload)) else []
        for msg in self.payload_messages(payload):
            items.extend(self._responses_message_items(msg))
        return items

    def _responses_message_items(self, msg):
        items = [{'type': 'function_call_output', 'call_id': c.tool_result.tool_use_id, 'output': self._tool_result_text(c.tool_result)}
                 for c in msg.content if c.tool_result]
        items.extend(self._responses_user_message([MessageContent(image=item.image) for c in msg.content if c.tool_result
                                                   for item in c.tool_result.content if item.image]))
        body = [c for c in msg.content if not c.tool_result and not c.cache_point]
        if msg.role == 'assistant':
            return items + [item for c in body for item in self._responses_assistant_item(c)]
        return items + self._responses_user_message(body)

    def _responses_assistant_item(self, c):
        if c.reasoning_content:
            return [c.reasoning_content.responses_item] if c.reasoning_content.responses_item else []
        if c.tool_use:
            return [{'type': 'function_call', 'call_id': c.tool_use.tool_use_id, 'name': c.tool_use.name,
                     'arguments': self._tool_use_arguments(c.tool_use), 'status': 'completed'}]
        return [{'type': 'message', 'role': 'assistant', 'content': c.text}] if c.text else []

    def _responses_user_message(self, content_list):
        parts = [part for c in content_list for part in self._responses_user_parts(c)]
        return [{'type': 'message', 'role': 'user', 'content': parts}] if parts else []

    def _responses_user_parts(self, c):
        if c.text:
            return [{'type': 'input_text', 'text': c.text}]
        if c.image:
            return [{'type': 'input_image', 'image_url': self._data_url(c.image.format, c.image.source.bytes), 'detail': 'auto'}]
        if c.document:
            return [{'type': 'input_text', 'text': self._document_text(c.document)}]
        return []

    def _data_url(self, media_format, raw):
        return f'data:image/{media_format};base64,{base64.b64encode(raw).decode()}'

    def _document_text(self, document):
        return f'[Document: {document.name}.{document.format}]\n{base64.b64encode(document.source.bytes).decode()}'

    def _tool_use_arguments(self, tool_use):
        return json.dumps(tool_use.input) if isinstance(tool_use.input, dict) else str(tool_use.input)

    def _tool_result_text(self, tool_result):
        parts = []
        for trc in tool_result.content:
            if trc.text: parts.append(trc.text)
            elif trc.json is not None: parts.append(json.dumps(trc.json))
            elif trc.image: parts.append('[image provided in the following message]')
            elif trc.document: parts.append(f'[document: {trc.document.name}.{trc.document.format}]')
        return '\n'.join(parts)

    def _convert_tool_results(self, content_list):
        return [{'role': 'tool', 'tool_call_id': c.tool_result.tool_use_id, 'content': self._tool_result_text(c.tool_result)}
                for c in content_list if c.tool_result]

    def _convert_assistant(self, content_list):
        openai_msg = {'role': 'assistant'}
        texts, tool_calls, reasoning = [], [], []
        for c in content_list:
            if c.text: texts.append(c.text)
            elif c.tool_use:
                tool_calls.append({'id': c.tool_use.tool_use_id, 'type': 'function',
                    'function': {'name': c.tool_use.name, 'arguments': self._tool_use_arguments(c.tool_use)}})
            elif c.reasoning_content and c.reasoning_content.reasoning_text:
                reasoning.append(c.reasoning_content.reasoning_text.text)
        if texts: openai_msg['content'] = '\n'.join(texts)
        if tool_calls: openai_msg['tool_calls'] = tool_calls
        if reasoning: openai_msg['reasoning_content'] = '\n'.join(reasoning)
        return [openai_msg]

    def _convert_user(self, content_list):
        parts, has_multimodal = [], False
        for c in content_list:
            if c.text: parts.append({'type': 'text', 'text': c.text})
            elif c.image:
                has_multimodal = True
                parts.append({'type': 'image_url', 'image_url': {'url': self._data_url(c.image.format, c.image.source.bytes)}})
            elif c.document:
                parts.append({'type': 'text', 'text': self._document_text(c.document)})
        if not parts:
            return []
        if has_multimodal or len(parts) > 1:
            return [{'role': 'user', 'content': parts}]
        return [{'role': 'user', 'content': parts[0].get('text', '')}]

    def _convert_message(self, msg):
        images = [MessageContent(image=item.image) for c in msg.content if c.tool_result for item in c.tool_result.content if item.image]
        tool_results = self._convert_tool_results(msg.content) + (self._convert_user(images) if images else [])
        other = [c for c in msg.content if not c.tool_result and not c.cache_point]
        if not other:
            return tool_results
        if msg.role == 'assistant':
            return tool_results + self._convert_assistant(other)
        return tool_results + self._convert_user(other)

    def _parse_completion(self, completion, latency_ms) -> ConverseResponse:
        if not completion.choices:
            return ConverseResponse(
                output=ConverseOutput(message=Message(role='assistant', content=[MessageContent(text='')])),
                stop_reason='end_turn', usage=TokenUsage(), metrics=ConverseMetrics(latency_ms=latency_ms))
        choice = completion.choices[0]
        content = []
        reasoning = getattr(choice.message, 'reasoning', None) or getattr(choice.message, 'reasoning_content', None)
        if reasoning:
            content.append(MessageContent(reasoning_content=ReasoningContent(
                reasoning_text=ReasoningText(text=reasoning, signature=''), redacted_content=None)))
        if choice.message.content:
            content.append(MessageContent(text=choice.message.content))
        for tc in (choice.message.tool_calls or []):
            args = tc.function.arguments
            if isinstance(args, str):
                try: args = json.loads(args)
                except json.JSONDecodeError: args = {"raw_input": args}
            content.append(MessageContent(tool_use=ToolUse(tool_use_id=new_tool_use_id(), name=tc.function.name, input=args)))
        return ConverseResponse(
            output=ConverseOutput(message=Message(role='assistant', content=content)),
            stop_reason=STOP_REASON_MAP.get(choice.finish_reason or '', 'end_turn'),
            usage=token_usage_from_openai(completion.usage),
            metrics=ConverseMetrics(latency_ms=latency_ms))

    def _consume_stream(self, stream, start) -> ConverseResponse:
        for _event in self._stream_events(stream):
            pass
        return self._stream_response(start)

    def _consume_responses_stream(self, stream, start) -> ConverseResponse:
        for _event in self._responses_stream_events(stream):
            pass
        return self._stream_response(start)

    def _start_stream(self):
        self._stream_output = {}
        self._stream_started_blocks = set()
        self._stream_block_indexes = {}
        self._stream_block_order = []
        self._stream_finish_reason = None
        self._stream_usage = None

    def _stream_events(self, stream):
        self._start_stream()
        yield {"type": "message_start", "role": "assistant"}
        for chunk in stream:
            if chunk.usage:
                self._stream_usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                self._stream_finish_reason = choice.finish_reason
            yield from self._stream_reasoning_delta(delta)
            yield from self._stream_text_delta(delta)
            yield from self._stream_tool_deltas(delta)
        yield from self._stream_end_events()

    def _responses_stream_events(self, stream):
        self._start_stream()
        yield {"type": "message_start", "role": "assistant"}
        for event in stream:
            event_type = getattr(event, 'type', '')
            if event_type == 'response.output_text.delta':
                yield from self._responses_text_delta(event)
            elif event_type in ('response.reasoning_text.delta', 'response.reasoning_summary_text.delta'):
                yield from self._responses_reasoning_delta(event)
            elif event_type == 'response.output_item.added':
                yield from self._responses_output_item(event.item, event.output_index)
            elif event_type == 'response.output_item.done':
                yield from self._responses_output_item(event.item, event.output_index)
                self._capture_responses_reasoning(event.item, event.output_index)
            elif event_type == 'response.function_call_arguments.delta':
                yield from self._responses_function_call_arguments_delta(event)
            elif event_type == 'response.completed':
                self._responses_completed(event.response)
            elif event_type in ('response.failed', 'response.error'):
                raise RuntimeError(getattr(event, 'error', event))
        yield from self._stream_end_events()

    def _stream_end_events(self):
        for index in self._stream_block_order:
            yield {"type": "content_block_stop", "index": index}
        yield {"type": "message_stop", "stop_reason": STOP_REASON_MAP.get(self._stream_finish_reason or '', self._stream_finish_reason or 'end_turn')}
        yield {"type": "metadata", "usage": self._stream_usage_obj(), "metrics": None}

    def _stream_item(self, key, kind, **initial):
        self._stream_block_index(key)
        return self._stream_output.setdefault(key, {'kind': kind, **initial})

    def _stream_block_index(self, key):
        if key not in self._stream_block_indexes:
            self._stream_block_indexes[key] = len(self._stream_block_order)
            self._stream_block_order.append(self._stream_block_indexes[key])
        return self._stream_block_indexes[key]

    def _start_content_block(self, index, block_type, **kwargs):
        if index in self._stream_started_blocks:
            return []
        self._stream_started_blocks.add(index)
        return [{"type": "content_block_start", "index": index, "block_type": block_type, **kwargs}]

    def _emit_text(self, key, text):
        item = self._stream_item(key, 'text', parts=[])
        index = self._stream_block_indexes[key]
        yield from self._start_content_block(index, "text")
        item['parts'].append(text)
        yield {"type": "text_delta", "index": index, "text": text}

    def _emit_reasoning(self, key, text):
        item = self._stream_item(key, 'reasoning', parts=[], responses_item=None)
        index = self._stream_block_indexes[key]
        yield from self._start_content_block(index, "reasoning")
        item['parts'].append(text)
        yield {"type": "reasoning_delta", "index": index, "reasoning": {"text": text}}

    def _stream_text_delta(self, delta):
        if delta.content:
            yield from self._emit_text("text", delta.content)

    def _stream_reasoning_delta(self, delta):
        if reasoning := (getattr(delta, 'reasoning', None) or getattr(delta, 'reasoning_content', None)):
            yield from self._emit_reasoning("reasoning", reasoning)

    def _stream_tool_deltas(self, delta):
        for tc in (delta.tool_calls or []):
            tool_call = self._stream_item(f"tool:{tc.index}", 'tool', id=new_tool_use_id(), name='', arguments='')
            index = self._stream_block_indexes[f"tool:{tc.index}"]
            if tc.function:
                if tc.function.name:
                    tool_call['name'] = tc.function.name
                if tc.function.arguments:
                    tool_call['arguments'] += tc.function.arguments
            if tool_call['name']:
                yield from self._start_content_block(index, "tool_use", tool_use_id=tool_call['id'], name=tool_call['name'])
            if index in self._stream_started_blocks and tc.function and tc.function.arguments:
                yield {"type": "tool_use_input_delta", "index": index, "partial_json": tc.function.arguments}

    def _responses_text_delta(self, event):
        yield from self._emit_text(f"text:{event.output_index}:{event.content_index}", event.delta)

    def _responses_reasoning_delta(self, event):
        yield from self._emit_reasoning(f"reasoning:{event.output_index}", event.delta)

    def _capture_responses_reasoning(self, item, output_index):
        if getattr(item, 'type', None) != 'reasoning' or not getattr(item, 'encrypted_content', None):
            return
        entry = self._stream_item(f"reasoning:{output_index}", 'reasoning', parts=[], responses_item=None)
        entry['responses_item'] = item.model_dump(exclude_none=True)

    def _responses_output_item(self, item, output_index):
        if getattr(item, 'type', None) != 'function_call':
            return []
        tool_call = self._stream_item(f"tool:{output_index}", 'tool', id=new_tool_use_id(), name='', arguments='')
        index = self._stream_block_indexes[f"tool:{output_index}"]
        previous_arguments = tool_call['arguments']
        tool_call['name'] = item.name or tool_call['name']
        if item.arguments and not tool_call['arguments']:
            tool_call['arguments'] = item.arguments
        if tool_call['name']:
            yield from self._start_content_block(index, "tool_use", tool_use_id=tool_call['id'], name=tool_call['name'])
        if index in self._stream_started_blocks and tool_call['arguments'] and not previous_arguments:
            yield {"type": "tool_use_input_delta", "index": index, "partial_json": tool_call['arguments']}

    def _responses_function_call_arguments_delta(self, event):
        tool_call = self._stream_item(f"tool:{event.output_index}", 'tool', id=new_tool_use_id(), name='', arguments='')
        index = self._stream_block_indexes[f"tool:{event.output_index}"]
        tool_call['arguments'] += event.delta
        if index in self._stream_started_blocks:
            yield {"type": "tool_use_input_delta", "index": index, "partial_json": event.delta}

    def _responses_completed(self, response):
        for output_index, item in enumerate(response.output):
            self._capture_responses_reasoning(item, output_index)
        self._stream_usage = response.usage
        self._stream_finish_reason = self._responses_stop_reason(response)

    def _responses_stop_reason(self, response):
        if any(item['kind'] == 'tool' for item in self._stream_output.values()):
            return 'tool_use'
        if getattr(response, 'status', '') == 'incomplete':
            return 'max_tokens'
        return 'end_turn'

    def _stream_content_blocks(self, item):
        if item['kind'] == 'reasoning':
            text = ''.join(item['parts'])
            return [MessageContent(reasoning_content=ReasoningContent(
                reasoning_text=ReasoningText(text=text, signature='') if text else None, responses_item=item['responses_item']))]
        if item['kind'] == 'text':
            return [MessageContent(text=text)] if (text := ''.join(item['parts'])) else []
        try:
            args = json.loads(item['arguments']) if item['arguments'] else {}
        except (json.JSONDecodeError, ValueError):
            args = {"raw_input": item['arguments']}
        return [MessageContent(tool_use=ToolUse(tool_use_id=item['id'], name=item['name'], input=args))]

    def _stream_response(self, start) -> ConverseResponse:
        content = [block for item in self._stream_output.values() for block in self._stream_content_blocks(item)]
        return ConverseResponse(
            output=ConverseOutput(message=Message(role='assistant', content=content)),
            stop_reason=STOP_REASON_MAP.get(self._stream_finish_reason or '', self._stream_finish_reason or 'end_turn'),
            usage=self._stream_usage_obj(),
            metrics=ConverseMetrics(latency_ms=int((time.time() - start) * 1000)))

    def _stream_usage_obj(self):
        return token_usage_from_openai(self._stream_usage)

    def rate_limited(self, error):
        return getattr(error, 'status_code', None) == 429

    def _openai_stream(self, messages):
        if self.uses_responses_api:
            return self.openai_client.responses.create(**self._build_responses_params(messages), stream=True)
        params = self._build_params(messages)
        params['stream'] = True
        params['stream_options'] = {'include_usage': True}
        return self.openai_client.chat.completions.create(**params)

    def _consumed_response(self, messages):
        start = time.time()
        stream = self._openai_stream(messages)
        if self.uses_responses_api:
            return self._consume_responses_stream(stream, start)
        return self._consume_stream(stream, start)

    def stream(self, messages=None):
        for callback in self.callbacks:
            try:
                if hasattr(callback, 'on_converse_start'): callback.on_converse_start(self)
            except Exception as e: logger.warning(f"Callback error: {e}")
        start = time.time()
        try:
            stream = self._openai_stream(messages)
            if self.uses_responses_api:
                yield from self._responses_stream_events(stream)
            else:
                yield from self._stream_events(stream)
            response = self._stream_response(start)
        except Exception as error:
            for callback in self.callbacks:
                try:
                    if hasattr(callback, 'on_converse_error'): callback.on_converse_error(self, error)
                except Exception as callback_error: logger.warning(f"Callback error: {callback_error}")
            raise
        response.model_id = self.model_id
        for callback in self.callbacks:
            try:
                if hasattr(callback, 'on_converse_end'): callback.on_converse_end(response)
            except Exception as e: logger.warning(f"Callback error: {e}")
        return response

    def with_extra_params(self, params):
        self.extra_params = {**(self.extra_params or {}), **params}
        return self

    def with_thinking(self, tokens: int | str = 1024, enabled: bool = True):
        thinking_config = ThinkingConfig(
            type="enabled" if enabled else "disabled",
            budget_tokens=tokens
        )
        if self.additional_model_request_fields is None:
            self.additional_model_request_fields = AdditionalModelRequestFields()
        self.additional_model_request_fields.thinking = thinking_config
        return self

    def _get_response(self, messages=None):
        for callback in self.callbacks:
            try: callback.on_converse_start(self)
            except Exception as e: logger.warning(f"Callback error: {e}")
        try:
            response = self.retry_rate_limits(lambda: self._consumed_response(messages))
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

    async def _aget_response(self, messages=None):
        for callback in self.callbacks:
            try: callback.on_converse_start(self)
            except Exception as e: logger.warning(f"Callback error: {e}")
        params = self._build_params(messages)
        params['stream'] = True
        params['stream_options'] = {'include_usage': True}
        start = time.time()
        try:
            stream = await self.async_openai_client.chat.completions.create(**params)
            # Async stream — collect chunks
            text_parts, reasoning_parts, tool_calls_map, finish_reason, usage = [], [], {}, None, None
            async for chunk in stream:
                if chunk.usage: usage = chunk.usage
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                if chunk.choices[0].finish_reason: finish_reason = chunk.choices[0].finish_reason
                if delta.content: text_parts.append(delta.content)
                reasoning = getattr(delta, 'reasoning', None) or getattr(delta, 'reasoning_content', None)
                if reasoning: reasoning_parts.append(reasoning)
                for tc in (delta.tool_calls or []):
                    idx = tc.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {'id': new_tool_use_id(), 'name': '', 'arguments': ''}
                    if tc.function:
                        if tc.function.name: tool_calls_map[idx]['name'] = tc.function.name
                        if tc.function.arguments: tool_calls_map[idx]['arguments'] += tc.function.arguments

            content = []
            if reasoning_parts:
                content.append(MessageContent(reasoning_content=ReasoningContent(
                    reasoning_text=ReasoningText(text=''.join(reasoning_parts), signature=''), redacted_content=None)))
            full_text = ''.join(text_parts)
            if full_text: content.append(MessageContent(text=full_text))
            for idx in sorted(tool_calls_map):
                tc = tool_calls_map[idx]
                args = tc['arguments']
                try: args = json.loads(args)
                except (json.JSONDecodeError, ValueError): args = {"raw_input": args}
                content.append(MessageContent(tool_use=ToolUse(tool_use_id=tc['id'], name=tc['name'], input=args)))

            response = ConverseResponse(
                output=ConverseOutput(message=Message(role='assistant', content=content)),
                stop_reason=STOP_REASON_MAP.get(finish_reason or '', 'end_turn'),
                usage=token_usage_from_openai(usage),
                metrics=ConverseMetrics(latency_ms=int((time.time() - start) * 1000)))
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


@dataclass
class Mantle(_MantleTransport, Converse):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_mode: str = 'chat_completions'
    extra_params: Optional[dict] = None
    session_affinity_header: Optional[str] = None

    @property
    def structured_output_class(self):
        return StructuredMantle

    def with_structured_output(self, output_model, force_choice=True, skip_add_tool=False, first_tool_only=True):
        structured = super().with_structured_output(output_model, force_choice, skip_add_tool, first_tool_only)
        structured.api_key = self.api_key
        structured.base_url = self.base_url
        structured.api_mode = self.api_mode
        structured.extra_params = self.extra_params
        structured.session_affinity_header = self.session_affinity_header
        return structured


@dataclass
class MantleAgent(_MantleTransport, ConverseAgent):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_mode: str = 'chat_completions'
    extra_params: Optional[dict] = None
    session_affinity_header: Optional[str] = None

    def prune_dangling_reasoning(self):
        pass


@dataclass
class StructuredMantle(_MantleTransport, StructuredConverse):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_mode: str = 'chat_completions'
    extra_params: Optional[dict] = None
    session_affinity_header: Optional[str] = None

    @property
    def supports_forced_tool_choice(self):
        return True

    @property
    def strict_tool_names(self):
        return {self.output_model.__name__}
