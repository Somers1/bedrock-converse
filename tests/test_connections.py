import unittest

import httpx

from bedrock.connections import KEEPALIVE, Finished, connection_pool
from bedrock.converse import Converse
from bedrock.embedding import OpenAIEmbedding
from bedrock.mantle import Mantle, MantleAgent


def mantle(cls=Mantle, base_url="http://herman.test/v1"):
    return cls(model_id="m", region_name="us-east-1", api_key="key", base_url=base_url)


class TestSharedConnections(unittest.TestCase):
    def test_agents_on_one_endpoint_share_a_pool(self):
        self.assertIs(mantle().openai_client._client, mantle(MantleAgent).openai_client._client)
        self.assertIs(mantle().openai_client._client, connection_pool("http://herman.test/v1"))

    def test_endpoints_keep_separate_pools(self):
        self.assertIsNot(mantle().openai_client._client, mantle(base_url="http://other.test/v1").openai_client._client)

    def test_each_agent_keeps_its_own_retries_and_timeout(self):
        patient, impatient = mantle(), mantle(MantleAgent)
        impatient.max_retries = 0
        impatient.read_timeout = 30.0
        self.assertEqual((patient.openai_client.max_retries, impatient.openai_client.max_retries), (2, 0))
        self.assertEqual((patient.openai_client.timeout.read, impatient.openai_client.timeout.read), (180.0, 30.0))

    def test_embeddings_use_the_same_pool(self):
        embedding = OpenAIEmbedding(model_id="e", api_key="key", base_url="http://herman.test/v1")
        self.assertIs(embedding.openai_client._client, mantle().openai_client._client)

    def test_idle_connections_are_kept_for_five_minutes(self):
        self.assertEqual(KEEPALIVE.keepalive_expiry, 300)
        self.assertEqual(mantle().openai_client.base_url.host, "herman.test")


class Body(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks, self.read, self.closed = chunks, [], False

    def __iter__(self):
        for chunk in self.chunks:
            self.read.append(chunk)
            yield chunk

    def close(self):
        self.closed = True


class TestStreamedResponsesReturnTheirConnection(unittest.TestCase):
    def stopped_at(self, chunks, marker):
        body = Body(chunks)
        finished = Finished(body)
        for chunk in finished:
            if marker in chunk:
                break
        finished.close()
        return body

    def test_a_stream_left_at_its_end_marker_is_read_to_the_end_before_closing(self):
        body = self.stopped_at([b"data: {}\n\n", b"data: [DONE]\n\n", b"\n"], b"[DONE]")
        self.assertEqual(body.read[-1], b"\n")
        self.assertTrue(body.closed)

    def test_an_end_marker_split_across_chunks_still_counts(self):
        body = self.stopped_at([b"data: {}\n\ndata: [DO", b"NE]\n\n", b"\n"], b"NE]")
        self.assertEqual(len(body.read), 3)

    def test_a_stream_cancelled_midway_is_closed_without_reading_on(self):
        body = self.stopped_at([b"data: {}\n\n", b"data: {}\n\n", b"data: [DONE]\n\n"], b"{}")
        self.assertEqual(len(body.read), 1)
        self.assertTrue(body.closed)

    def test_every_response_from_the_pool_is_wrapped(self):
        response = httpx.Response(200, stream=Body([b"data: [DONE]\n\n"]))
        for hook in connection_pool("http://herman.test/v1").event_hooks["response"]:
            hook(response)
        self.assertIsInstance(response.stream, Finished)


class TestSharedBedrockClient(unittest.TestCase):
    def converse(self, region="ap-southeast-2", key="a"):
        return Converse(model_id="m", region_name=region, aws_access_key_id=key, aws_secret_access_key="s")

    def test_requests_with_the_same_region_and_credentials_share_a_client(self):
        self.assertIs(self.converse().client, self.converse().client)

    def test_another_region_or_account_gets_its_own_client(self):
        self.assertIsNot(self.converse().client, self.converse(region="us-east-1").client)
        self.assertIsNot(self.converse().client, self.converse(key="b").client)
