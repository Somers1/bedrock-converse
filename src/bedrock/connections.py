from functools import lru_cache

import httpx
from openai import DefaultHttpxClient

KEEPALIVE = httpx.Limits(max_connections=1000, max_keepalive_connections=100, keepalive_expiry=300)


@lru_cache(maxsize=None)
def connection_pool(base_url):
    return DefaultHttpxClient(limits=KEEPALIVE)
