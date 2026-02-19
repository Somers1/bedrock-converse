import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List

import aiohttp
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from .converse import Converse, ConverseAgent, StructuredConverse, ConverseResponse, Message
from .bases import BaseCallbackHandler

logger = logging.getLogger(__name__)


def _sign_and_post(session, region, endpoint_url, payload_bytes, api_key=None):
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
        return requests.post(endpoint_url, headers=headers, data=payload_bytes)
    credentials = session.get_credentials().get_frozen_credentials()
    aws_request = AWSRequest(method='POST', url=endpoint_url, data=payload_bytes, headers=headers)
    SigV4Auth(credentials, 'bedrock', region).add_auth(aws_request)
    return requests.post(endpoint_url, headers=dict(aws_request.headers), data=aws_request.body)


async def _sign_and_post_async(session, region, endpoint_url, payload_bytes, api_key=None):
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    else:
        credentials = session.get_credentials().get_frozen_credentials()
        aws_request = AWSRequest(method='POST', url=endpoint_url, data=payload_bytes, headers=headers)
        SigV4Auth(credentials, 'bedrock', region).add_auth(aws_request)
        headers = dict(aws_request.headers)
    async with aiohttp.ClientSession() as http_session:
        async with http_session.post(endpoint_url, headers=headers, data=payload_bytes) as resp:
            resp.raise_for_status()
            return await resp.json()


class _MantleTransport:
    """Mixin that overrides transport to use Mantle HTTP endpoint instead of boto3."""
    api_key: Optional[str] = None

    @property
    def _mantle_endpoint(self):
        region = self.region_name or self.session.region_name
        return f'https://bedrock-mantle.{region}.api.aws/v1/chat/completions'

    def _get_response(self, messages=None):
        for callback in self.callbacks:
            try: callback.on_converse_start(self)
            except Exception as e: logger.warning(f"Callback error: {e}")
        self.remove_invalid_caching(messages)
        payload = self._to_openai_payload(messages or self.messages)
        payload['model'] = self.model_id
        payload_bytes = json.dumps(payload)
        start = time.time()
        resp = _sign_and_post(self.session, self.region_name or self.session.region_name, self._mantle_endpoint, payload_bytes, self.api_key)
        resp.raise_for_status()
        response = self._parse_openai_response(resp.json(), int((time.time() - start) * 1000))
        response.model_id = self.model_id
        for callback in self.callbacks:
            try: callback.on_converse_end(response)
            except Exception as e: logger.warning(f"Callback error: {e}")
        return response

    async def _aget_response(self, messages=None):
        for callback in self.callbacks:
            try: callback.on_converse_start(self)
            except Exception as e: logger.warning(f"Callback error: {e}")
        self.remove_invalid_caching(messages)
        payload = self._to_openai_payload(messages or self.messages)
        payload['model'] = self.model_id
        payload_bytes = json.dumps(payload)
        start = time.time()
        result = await _sign_and_post_async(self.session, self.region_name or self.session.region_name, self._mantle_endpoint, payload_bytes, self.api_key)
        response = self._parse_openai_response(result, int((time.time() - start) * 1000))
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
