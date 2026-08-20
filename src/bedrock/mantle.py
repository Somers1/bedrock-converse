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


class _MantleTransport:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_mode: str = 'chat_completions'

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
        tools = [{'type': 'function', 'function': {
            'name': t.tool_spec.name, 'description': t.tool_spec.description,
            'parameters': t.tool_spec.input_schema.get('json', {})
        }} for t in self.tool_config.tools if t.tool_spec]
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

    def _build_params(self, messages=None) -> dict:
        self.remove_invalid_caching(messages)
        msgs = []
        if self.system:
            system_text = '\n'.join(s.text for s in self.system if s.text)
            if system_text:
                msgs.append({'role': 'system', 'content': system_text})
        for msg in (messages or self.messages):
            msgs.extend(self._convert_message(msg))
        params = {'model': self.model_id, 'messages': msgs}
        self._build_tool_params(params)
        self._build_inference_params(params)
        self._build_thinking_params(params)
        if self.cache_key:
            params['prompt_cache_key'] = self.cache_key
        return params

    def _build_responses_params(self, messages=None):
        params = self._build_params(messages)
        response_params = {'model': params['model'], 'input': self._responses_input(params['messages']), 'store': False,
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
        return response_params

    def _responses_tools(self, tools):
        return [{'type': 'function', 'name': tool['function']['name'],
                 'description': tool['function'].get('description'), 'parameters': tool['function'].get('parameters') or {},
                 'strict': False} for tool in tools]

    def _responses_tool_choice(self, tool_choice):
        if isinstance(tool_choice, str):
            return tool_choice
        if tool_choice and tool_choice.get('type') == 'function':
            return {'type': 'function', 'name': tool_choice['function']['name']}
        return None

    def _responses_input(self, messages):
        items = []
        for message in messages:
            if message['role'] == 'tool':
                items.append({'type': 'function_call_output', 'call_id': message['tool_call_id'], 'output': message.get('content') or ''})
                continue
            if message['role'] == 'assistant':
                if reasoning_item := message.get('reasoning_item'):
                    items.append(reasoning_item)
                if message.get('content'):
                    items.append({'type': 'message', 'role': 'assistant', 'content': message['content']})
                items.extend(self._responses_function_calls(message.get('tool_calls') or []))
                continue
            items.append({'type': 'message', 'role': message['role'], 'content': self._responses_content(message.get('content') or '')})
        return items

    def _responses_function_calls(self, tool_calls):
        return [{'type': 'function_call', 'call_id': tool_call['id'], 'name': tool_call['function']['name'],
                 'arguments': tool_call['function']['arguments'], 'status': 'completed'} for tool_call in tool_calls]

    def _responses_content(self, content):
        if isinstance(content, str):
            return content
        parts = []
        for part in content:
            if part.get('type') == 'text':
                parts.append({'type': 'input_text', 'text': part.get('text') or ''})
            elif part.get('type') == 'image_url':
                parts.append({'type': 'input_image', 'image_url': part['image_url']['url'], 'detail': 'auto'})
        return parts

    def _convert_tool_results(self, content_list):
        results = []
        for c in content_list:
            if not c.tool_result:
                continue
            tr = c.tool_result
            parts = []
            for trc in tr.content:
                if trc.text: parts.append(trc.text)
                elif trc.json is not None: parts.append(json.dumps(trc.json))
                elif trc.image: parts.append('[image provided in the following message]')
                elif trc.document: parts.append(f'[document: {trc.document.name}.{trc.document.format}]')
            results.append({'role': 'tool', 'tool_call_id': tr.tool_use_id, 'content': '\n'.join(parts)})
        return results

    def _convert_assistant(self, content_list):
        openai_msg = {'role': 'assistant'}
        texts, tool_calls, reasoning, reasoning_item = [], [], [], None
        for c in content_list:
            if c.text: texts.append(c.text)
            elif c.tool_use:
                tool_calls.append({'id': c.tool_use.tool_use_id, 'type': 'function',
                    'function': {'name': c.tool_use.name, 'arguments': json.dumps(c.tool_use.input) if isinstance(c.tool_use.input, dict) else str(c.tool_use.input)}})
            elif c.reasoning_content:
                if c.reasoning_content.reasoning_text: reasoning.append(c.reasoning_content.reasoning_text.text)
                reasoning_item = c.reasoning_content.responses_item or reasoning_item
        if texts: openai_msg['content'] = '\n'.join(texts)
        if tool_calls: openai_msg['tool_calls'] = tool_calls
        if reasoning: openai_msg['reasoning_content'] = '\n'.join(reasoning)
        if reasoning_item and self.uses_responses_api: openai_msg['reasoning_item'] = reasoning_item
        return [openai_msg]

    def _convert_user(self, content_list):
        parts, has_multimodal = [], False
        for c in content_list:
            if c.text: parts.append({'type': 'text', 'text': c.text})
            elif c.image:
                has_multimodal = True
                b64 = base64.b64encode(c.image.source.bytes).decode()
                parts.append({'type': 'image_url', 'image_url': {'url': f'data:image/{c.image.format};base64,{b64}'}})
            elif c.document:
                b64 = base64.b64encode(c.document.source.bytes).decode()
                parts.append({'type': 'text', 'text': f'[Document: {c.document.name}.{c.document.format}]\n{b64}'})
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
        self._stream_text_parts = []
        self._stream_reasoning_parts = []
        self._stream_reasoning_item = None
        self._stream_tool_calls = {}
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
                yield from self._responses_output_item(event.item)
            elif event_type == 'response.output_item.done':
                yield from self._responses_output_item(event.item)
                self._capture_responses_reasoning(event.item)
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

    def _stream_text_delta(self, delta):
        if not delta.content:
            return
        index = self._stream_block_index("text")
        yield from self._start_content_block(index, "text")
        self._stream_text_parts.append(delta.content)
        yield {"type": "text_delta", "index": index, "text": delta.content}

    def _stream_reasoning_delta(self, delta):
        reasoning = getattr(delta, 'reasoning', None) or getattr(delta, 'reasoning_content', None)
        if not reasoning:
            return
        index = self._stream_block_index("reasoning")
        yield from self._start_content_block(index, "reasoning")
        self._stream_reasoning_parts.append(reasoning)
        yield {"type": "reasoning_delta", "index": index, "reasoning": {"text": reasoning}}

    def _stream_tool_deltas(self, delta):
        for tc in (delta.tool_calls or []):
            index = self._stream_block_index(f"tool:{tc.index}")
            tool_call = self._stream_tool_calls.setdefault(index, {'id': new_tool_use_id(), 'name': '', 'arguments': ''})
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
        index = self._stream_block_index(f"text:{event.output_index}:{event.content_index}")
        yield from self._start_content_block(index, "text")
        self._stream_text_parts.append(event.delta)
        yield {"type": "text_delta", "index": index, "text": event.delta}

    def _responses_reasoning_delta(self, event):
        part_index = getattr(event, 'content_index', getattr(event, 'summary_index', 0))
        index = self._stream_block_index(f"reasoning:{event.output_index}:{part_index}")
        yield from self._start_content_block(index, "reasoning")
        self._stream_reasoning_parts.append(event.delta)
        yield {"type": "reasoning_delta", "index": index, "reasoning": {"text": event.delta}}

    def _capture_responses_reasoning(self, item):
        if getattr(item, 'type', None) == 'reasoning' and getattr(item, 'encrypted_content', None):
            self._stream_reasoning_item = item.model_dump(exclude_none=True)

    def _responses_output_item(self, item):
        if getattr(item, 'type', None) != 'function_call':
            return []
        index = self._stream_block_index(f"tool:{item.id or item.call_id}")
        tool_call = self._stream_tool_calls.setdefault(index, {'id': new_tool_use_id(), 'name': '', 'arguments': ''})
        previous_arguments = tool_call['arguments']
        tool_call['name'] = item.name or tool_call['name']
        if item.arguments and not tool_call['arguments']:
            tool_call['arguments'] = item.arguments
        if tool_call['name']:
            yield from self._start_content_block(index, "tool_use", tool_use_id=tool_call['id'], name=tool_call['name'])
        if index in self._stream_started_blocks and tool_call['arguments'] and not previous_arguments:
            yield {"type": "tool_use_input_delta", "index": index, "partial_json": tool_call['arguments']}

    def _responses_function_call_arguments_delta(self, event):
        index = self._stream_block_index(f"tool:{event.item_id}")
        tool_call = self._stream_tool_calls.setdefault(index, {'id': '', 'name': '', 'arguments': ''})
        tool_call['arguments'] += event.delta
        if index in self._stream_started_blocks:
            yield {"type": "tool_use_input_delta", "index": index, "partial_json": event.delta}

    def _responses_completed(self, response):
        for item in response.output:
            self._capture_responses_reasoning(item)
        self._stream_usage = response.usage
        self._stream_finish_reason = self._responses_stop_reason(response)

    def _responses_stop_reason(self, response):
        if self._stream_tool_calls:
            return 'tool_use'
        if getattr(response, 'status', '') == 'incomplete':
            return 'max_tokens'
        return 'end_turn'

    def _stream_response(self, start) -> ConverseResponse:
        content = []
        if self._stream_reasoning_parts or self._stream_reasoning_item:
            reasoning_text = ReasoningText(text=''.join(self._stream_reasoning_parts), signature='') if self._stream_reasoning_parts else None
            content.append(MessageContent(reasoning_content=ReasoningContent(
                reasoning_text=reasoning_text, responses_item=self._stream_reasoning_item)))
        if text := ''.join(self._stream_text_parts):
            content.append(MessageContent(text=text))
        for index in sorted(self._stream_tool_calls):
            tool_call = self._stream_tool_calls[index]
            try:
                args = json.loads(tool_call['arguments']) if tool_call['arguments'] else {}
            except (json.JSONDecodeError, ValueError):
                args = {"raw_input": tool_call['arguments']}
            content.append(MessageContent(tool_use=ToolUse(tool_use_id=tool_call['id'], name=tool_call['name'], input=args)))
        return ConverseResponse(
            output=ConverseOutput(message=Message(role='assistant', content=content)),
            stop_reason=STOP_REASON_MAP.get(self._stream_finish_reason or '', self._stream_finish_reason or 'end_turn'),
            usage=self._stream_usage_obj(),
            metrics=ConverseMetrics(latency_ms=int((time.time() - start) * 1000)))

    def _stream_usage_obj(self):
        return token_usage_from_openai(self._stream_usage)

    def stream(self, messages=None):
        for callback in self.callbacks:
            try:
                if hasattr(callback, 'on_converse_start'): callback.on_converse_start(self)
            except Exception as e: logger.warning(f"Callback error: {e}")
        start = time.time()
        try:
            if self.uses_responses_api:
                stream = self.openai_client.responses.create(**self._build_responses_params(messages), stream=True)
                yield from self._responses_stream_events(stream)
            else:
                params = self._build_params(messages)
                params['stream'] = True
                params['stream_options'] = {'include_usage': True}
                stream = self.openai_client.chat.completions.create(**params)
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
        start = time.time()
        try:
            if self.uses_responses_api:
                stream = self.openai_client.responses.create(**self._build_responses_params(messages), stream=True)
                response = self._consume_responses_stream(stream, start)
            else:
                params = self._build_params(messages)
                params['stream'] = True
                params['stream_options'] = {'include_usage': True}
                stream = self.openai_client.chat.completions.create(**params)
                response = self._consume_stream(stream, start)
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

    @property
    def structured_output_class(self):
        return StructuredMantle

    def with_structured_output(self, output_model, force_choice=True, skip_add_tool=False, first_tool_only=True):
        structured = super().with_structured_output(output_model, force_choice, skip_add_tool, first_tool_only)
        structured.api_key = self.api_key
        structured.base_url = self.base_url
        structured.api_mode = self.api_mode
        return structured


@dataclass
class MantleAgent(_MantleTransport, ConverseAgent):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_mode: str = 'chat_completions'

    def prune_dangling_reasoning(self):
        pass


@dataclass
class StructuredMantle(_MantleTransport, StructuredConverse):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_mode: str = 'chat_completions'
