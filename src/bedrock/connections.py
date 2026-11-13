from contextlib import suppress
from functools import lru_cache

import httpx
from openai import DefaultHttpxClient

KEEPALIVE = httpx.Limits(max_connections=1000, max_keepalive_connections=100, keepalive_expiry=300)
DONE = b"[DONE]"
TAIL = 64


class Finished(httpx.SyncByteStream):
    def __init__(self, stream):
        self.stream = stream
        self.chunks = iter(stream)
        self.tail = b""

    def __iter__(self):
        for chunk in self.chunks:
            self.tail = (self.tail + chunk)[-TAIL:]
            yield chunk

    def close(self):
        if DONE in self.tail:
            with suppress(httpx.TransportError):
                for chunk in self.chunks:
                    pass
        self.stream.close()


def finish(response):
    response.stream = Finished(response.stream)


@lru_cache(maxsize=None)
def connection_pool(base_url):
    return DefaultHttpxClient(limits=KEEPALIVE, event_hooks={"response": [finish]})

