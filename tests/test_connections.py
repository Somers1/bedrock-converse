import unittest

from bedrock.connections import KEEPALIVE, connection_pool
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
