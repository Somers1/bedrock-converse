import base64
import json
import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, List

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from openai import OpenAI, AsyncOpenAI

from .converse import (Converse, ConverseAgent, StructuredConverse, ConverseResponse, ConverseOutput,
                       Message, MessageContent, ToolUse, TokenUsage, ConverseMetrics)
from .bases import BaseCallbackHandler

logger = logging.getLogger(__name__)

STOP_REASON_MAP = {'stop': 'end_turn', 'length': 'max_tokens', 'tool_calls': 'tool_use'}


class SigV4Auth_httpx(httpx.Auth):
    def __init__(self, credentials, region):
        self.credentials = credentials
        self.region = region

    def auth_flow(self, request):
        aws_request = AWSRequest(method=request.method, url=str(request.url), data=request.content, headers=dict(request.headers))
        SigV4Auth(self.credentials, 'bedrock', self.region).add_auth(aws_request)
        request.headers.update(aws_request.headers)
        yield request


class _MantleTransport:
    """Mixin that overrides transport to use Mantle OpenAI-compatible endpoint."""
    api_key: Optional[str] = None

    @property
    def _mantle_base_url(self):
        region = self.region_name or self.session.region_name
        return f'https://bedrock-mantle.{region}.api.aws/v1'

    @cached_property
    def openai_client(self) -> OpenAI:
        if self.api_key:
            return OpenAI(api_key=self.api_key, base_url=self._mantle_base_url)
        region = self.region_name or self.session.region_name
        credentials = self.session.get_credentials().get_frozen_credentials()
        transport = httpx.Client(auth=SigV4Auth_httpx(credentials, region))
        return OpenAI(api_key='unused', base_url=self._mantle_base_url, http_client=transport)

    @cached_property
    def async_openai_client(self) -> AsyncOpenAI:
        if self.api_key:
            return AsyncOpenAI(api_key=self.api_key, base_url=self._mantle_base_url)
        region = self.region_name or self.session.region_name
        credentials = self.session.get_credentials().get_frozen_credentials()
        transport = httpx.AsyncClient(auth=SigV4Auth_httpx(credentials, region))
        return AsyncOpenAI(api_key='unused', base_url=self._mantle_base_url, http_client=transport)

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
        if self.tool_config:
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
        if self.inference_config:
            ic = self.inference_config
            if ic.max_tokens is not None: params['max_tokens'] = ic.max_tokens
            if ic.temperature is not None: params['temperature'] = ic.temperature
            if ic.top_p is not None: params['top_p'] = ic.top_p
            if ic.stop_sequences: params['stop'] = ic.stop_sequences
        if (self.additional_model_request_fields and self.additional_model_request_fields.thinking
                and self.additional_model_request_fields.thinking.type == 'enabled'):
            budget = self.additional_model_request_fields.thinking.budget_tokens
            if budget <= 2048: params['reasoning_effort'] = 'low'
            elif budget <= 8192: params['reasoning_effort'] = 'medium'
            else: params['reasoning_effort'] = 'high'
        return params

    def _convert_message(self, msg):
        results = []
        tool_results = [c for c in msg.content if c.tool_result]
        other = [c for c in msg.content if not c.tool_result and not c.cache_point]
        for c in tool_results:
            tr = c.tool_result
            parts = []
            for trc in tr.content:
                if trc.text: parts.append(trc.text)
                elif trc.json is not None: parts.append(json.dumps(trc.json))
            results.append({'role': 'tool', 'tool_call_id': tr.tool_use_id, 'content': '\n'.join(parts)})
        if not other:
            return results
        if msg.role == 'assistant':
            openai_msg = {'role': 'assistant'}
            texts, tool_calls = [], []
            for c in other:
                if c.text: texts.append(c.text)
                elif c.tool_use:
                    tool_calls.append({'id': c.tool_use.tool_use_id, 'type': 'function',
                        'function': {'name': c.tool_use.name, 'arguments': json.dumps(c.tool_use.input) if isinstance(c.tool_use.input, dict) else str(c.tool_use.input)}})
            if texts: openai_msg['content'] = '\n'.join(texts)
            if tool_calls: openai_msg['tool_calls'] = tool_calls
            results.append(openai_msg)
        else:
            parts = []
            has_multimodal = False
            for c in other:
                if c.text: parts.append({'type': 'text', 'text': c.text})
                elif c.image:
                    has_multimodal = True
                    b64 = base64.b64encode(c.image.source.bytes).decode()
                    parts.append({'type': 'image_url', 'image_url': {'url': f'data:image/{c.image.format};base64,{b64}'}})
                elif c.document:
                    b64 = base64.b64encode(c.document.source.bytes).decode()
                    parts.append({'type': 'text', 'text': f'[Document: {c.document.name}.{c.document.format}]\n{b64}'})
            if has_multimodal or len(parts) > 1:
                results.append({'role': 'user', 'content': parts})
            elif parts:
                results.append({'role': 'user', 'content': parts[0].get('text', '')})
        return results

    def _parse_completion(self, completion, latency_ms) -> ConverseResponse:
        if not completion.choices:
            return ConverseResponse(
                output=ConverseOutput(message=Message(role='assistant', content=[MessageContent(text='')])),
                stop_reason='end_turn', usage=TokenUsage(), metrics=ConverseMetrics(latency_ms=latency_ms))
        choice = completion.choices[0]
        content = []
        if choice.message.content:
            content.append(MessageContent(text=choice.message.content))
        for tc in (choice.message.tool_calls or []):
            args = tc.function.arguments
            if isinstance(args, str):
                try: args = json.loads(args)
                except json.JSONDecodeError: args = {"raw_input": args}
            content.append(MessageContent(tool_use=ToolUse(tool_use_id=tc.id, name=tc.function.name, input=args)))
        usage = completion.usage
        return ConverseResponse(
            output=ConverseOutput(message=Message(role='assistant', content=content)),
            stop_reason=STOP_REASON_MAP.get(choice.finish_reason or '', 'end_turn'),
            usage=TokenUsage(input_tokens=usage.prompt_tokens if usage else 0,
                             output_tokens=usage.completion_tokens if usage else 0,
                             total_tokens=usage.total_tokens if usage else 0),
            metrics=ConverseMetrics(latency_ms=latency_ms))

    def _get_response(self, messages=None):
        for callback in self.callbacks:
            try: callback.on_converse_start(self)
            except Exception as e: logger.warning(f"Callback error: {e}")
        params = self._build_params(messages)
        start = time.time()
        completion = self.openai_client.chat.completions.create(**params)
        response = self._parse_completion(completion, int((time.time() - start) * 1000))
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
        start = time.time()
        completion = await self.async_openai_client.chat.completions.create(**params)
        response = self._parse_completion(completion, int((time.time() - start) * 1000))
        response.model_id = self.model_id
        for callback in self.callbacks:
            try: callback.on_converse_end(response)
            except Exception as e: logger.warning(f"Callback error: {e}")
        return response


@dataclass
class Mantle(_MantleTransport, Converse):
    api_key: Optional[str] = None


@dataclass
class MantleAgent(_MantleTransport, ConverseAgent):
    api_key: Optional[str] = None


@dataclass
class StructuredMantle(_MantleTransport, StructuredConverse):
    api_key: Optional[str] = None
