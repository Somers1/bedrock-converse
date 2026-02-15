import asyncio
import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bedrock.embedding import (
    BedrockEmbedding, MultimodalInput, EmbeddingResponse,
    TextChunker, ChunkerConfig,
    S3VectorsStore, VectorItem, VectorResponse,
)


# ══════════════════════════════════════════════════════════════════════════════
#  MultimodalInput
# ══════════════════════════════════════════════════════════════════════════════

class TestMultimodalInput(unittest.TestCase):
    def test_add_text(self):
        mi = MultimodalInput()
        mi.add_text("hello")
        self.assertEqual(mi.content[0], {"type": "text", "text": "hello"})

    def test_add_image(self):
        mi = MultimodalInput()
        mi.add_image("base64data", "image/jpeg")
        self.assertIn("image_url", mi.content[0])

    def test_chaining(self):
        mi = MultimodalInput().add_text("hi").add_image("b64")
        self.assertEqual(len(mi.content), 2)

    def test_add_image_from_file(self):
        import tempfile, base64
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(b'\x89PNG fake data')
            path = f.name
        try:
            mi = MultimodalInput().add_image_from_file(path)
            self.assertEqual(len(mi.content), 1)
            self.assertIn("image_url", mi.content[0])
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
#  BedrockEmbedding
# ══════════════════════════════════════════════════════════════════════════════

class TestBedrockEmbedding(unittest.TestCase):
    def setUp(self):
        self.emb = BedrockEmbedding()
        self.mock_client = MagicMock()
        self.emb._client = self.mock_client

    def _mock_response(self, embeddings):
        body = json.dumps({
            "id": "resp1", "response_type": "embeddings_floats",
            "embeddings": embeddings
        }).encode()
        mock_body = MagicMock()
        mock_body.read.return_value = body
        self.mock_client.invoke_model.return_value = {"body": mock_body}

    def test_embed_texts(self):
        self._mock_response({"float": [[0.1, 0.2]]})
        resp = self.emb.embed_texts(["hello"])
        self.mock_client.invoke_model.assert_called_once()
        self.assertEqual(resp.embeddings, {"float": [[0.1, 0.2]]})

    def test_embed_texts_custom_input_type(self):
        self._mock_response({"float": [[0.1]]})
        self.emb.embed_texts(["hi"], input_type="search_query")
        call_body = json.loads(self.mock_client.invoke_model.call_args[1]['body'])
        self.assertEqual(call_body['input_type'], 'search_query')

    def test_embed_images(self):
        self._mock_response({"float": [[0.3, 0.4]]})
        resp = self.emb.embed_images(["base64img"])
        self.assertIsInstance(resp, EmbeddingResponse)

    def test_embed_multimodal(self):
        self._mock_response({"float": [[0.5]]})
        mi = MultimodalInput().add_text("hi")
        resp = self.emb.embed_multimodal([mi])
        self.assertIsInstance(resp, EmbeddingResponse)

    def test_embed_query(self):
        self._mock_response({"float": [[0.1, 0.2, 0.3]]})
        result = self.emb.embed_query("search term")
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_embed_documents(self):
        self._mock_response({"float": [[0.1], [0.2]]})
        resp = self.emb.embed_documents(["doc1", "doc2"])
        self.assertIsInstance(resp, EmbeddingResponse)

    def test_aembed_texts(self):
        self._mock_response({"float": [[0.1]]})
        resp = asyncio.get_event_loop().run_until_complete(
            self.emb.aembed_texts(["hello"]))
        self.assertIsInstance(resp, EmbeddingResponse)

    @patch('bedrock.embedding.boto3.Session')
    def test_client_lazy_init_no_creds(self, mock_session_cls):
        emb = BedrockEmbedding()
        mock_session_cls.return_value.client.return_value = MagicMock()
        _ = emb.client
        mock_session_cls.assert_called_with(region_name='ap-southeast-2')

    @patch('bedrock.embedding.boto3.Session')
    def test_client_lazy_init_with_creds(self, mock_session_cls):
        emb = BedrockEmbedding(aws_access_key_id="ak", aws_secret_access_key="sk")
        mock_session_cls.return_value.client.return_value = MagicMock()
        _ = emb.client
        mock_session_cls.assert_called_with(
            region_name='ap-southeast-2', aws_access_key_id="ak", aws_secret_access_key="sk")


# ══════════════════════════════════════════════════════════════════════════════
#  TextChunker
# ══════════════════════════════════════════════════════════════════════════════

class TestTextChunker(unittest.TestCase):
    def test_empty_text(self):
        tc = TextChunker()
        self.assertEqual(tc.chunk(""), [])

    def test_whitespace_only(self):
        tc = TextChunker()
        self.assertEqual(tc.chunk("   "), [])

    def test_single_chunk(self):
        tc = TextChunker(ChunkerConfig(max_tokens=100, overlap_tokens=0))
        text = " ".join(f"w{i}" for i in range(10))
        chunks = tc.chunk(text)
        self.assertEqual(len(chunks), 1)

    def test_multiple_chunks(self):
        tc = TextChunker(ChunkerConfig(max_tokens=5, overlap_tokens=2))
        text = " ".join(f"w{i}" for i in range(20))
        chunks = tc.chunk(text)
        self.assertGreater(len(chunks), 1)

    def test_overlap(self):
        tc = TextChunker(ChunkerConfig(max_tokens=4, overlap_tokens=2))
        text = "a b c d e f g h"
        chunks = tc.chunk(text)
        # With overlap=2, step=2, chunks should overlap
        self.assertGreater(len(chunks), 2)

    def test_preserve_newlines_true(self):
        tc = TextChunker(ChunkerConfig(max_tokens=100, overlap_tokens=0, preserve_newlines=True))
        text = "line1\nline2\nline3"
        chunks = tc.chunk(text)
        self.assertEqual(len(chunks), 1)
        # When preserve_newlines=True, text is not joined, so split() keeps tokens with newlines
        # Actually split() splits on all whitespace, so newlines appear as token boundaries

    def test_preserve_newlines_false(self):
        tc = TextChunker(ChunkerConfig(max_tokens=100, overlap_tokens=0, preserve_newlines=False))
        text = "line1\n\n\nline2"
        chunks = tc.chunk(text)
        self.assertEqual(len(chunks), 1)
        # preserve_newlines=False joins with spaces first
        self.assertNotIn('\n', chunks[0])

    def test_invalid_config(self):
        with self.assertRaises(AssertionError):
            TextChunker(ChunkerConfig(max_tokens=0))

    def test_overlap_ge_max_tokens_invalid(self):
        with self.assertRaises(AssertionError):
            TextChunker(ChunkerConfig(max_tokens=5, overlap_tokens=5))

    def test_exact_boundary(self):
        tc = TextChunker(ChunkerConfig(max_tokens=3, overlap_tokens=0))
        text = "a b c d e f"
        chunks = tc.chunk(text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "a b c")
        self.assertEqual(chunks[1], "d e f")


# ══════════════════════════════════════════════════════════════════════════════
#  S3VectorsStore
# ══════════════════════════════════════════════════════════════════════════════

class TestS3VectorsStore(unittest.TestCase):
    @patch('bedrock.embedding.boto3.Session')
    def setUp(self, mock_session_cls):
        self.mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = self.mock_client
        self.store = S3VectorsStore(
            vector_bucket="test-bucket", index_name="test-index")

    def test_put_vectors_empty(self):
        self.store.put_vectors([])
        self.mock_client.put_vectors.assert_not_called()

    def test_put_vectors_single_batch(self):
        items = [VectorItem(key=f"k{i}", vector=[0.1, 0.2], metadata={"id": str(i)})
                 for i in range(3)]
        self.store.put_vectors(items)
        self.mock_client.put_vectors.assert_called_once()

    def test_put_vectors_multi_batch(self):
        self.store.max_batch = 2
        items = [VectorItem(key=f"k{i}", vector=[0.1], metadata={}) for i in range(5)]
        self.store.put_vectors(items)
        self.assertEqual(self.mock_client.put_vectors.call_count, 3)

    def test_put_vectors_retry_on_throttle(self):
        exc_cls = type('TooManyRequestsException', (Exception,), {})
        self.mock_client.exceptions.TooManyRequestsException = exc_cls
        self.mock_client.put_vectors.side_effect = [exc_cls("throttled"), None]
        self.store.backoff_base = 0.001
        items = [VectorItem(key="k1", vector=[0.1], metadata={})]
        self.store.put_vectors(items)
        self.assertEqual(self.mock_client.put_vectors.call_count, 2)

    def test_query(self):
        self.mock_client.query_vectors.return_value = {
            "vectors": [{"key": "k1", "distance": 0.5, "metadata": {"doc": "d1"}}]}
        results = self.store.query([0.1, 0.2])
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], VectorResponse)
        self.assertEqual(results[0].key, "k1")

    def test_delete_vectors_empty(self):
        self.store.delete_vectors_document_id([])
        self.mock_client.delete_vectors.assert_not_called()

    def test_delete_vectors(self):
        self.store.delete_vectors_document_id(["k1", "k2"])
        self.mock_client.delete_vectors.assert_called_once()

    def test_delete_vectors_retry(self):
        exc_cls = type('TooManyRequestsException', (Exception,), {})
        self.mock_client.exceptions.TooManyRequestsException = exc_cls
        self.mock_client.delete_vectors.side_effect = [exc_cls("throttled"), None]
        self.store.backoff_base = 0.001
        self.store.delete_vectors_document_id(["k1"])
        self.assertEqual(self.mock_client.delete_vectors.call_count, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  Dataclass construction
# ══════════════════════════════════════════════════════════════════════════════

class TestDataclasses(unittest.TestCase):
    def test_vector_item(self):
        vi = VectorItem(key="k1", vector=[0.1, 0.2], metadata={"a": "b"})
        self.assertEqual(vi.key, "k1")

    def test_vector_response(self):
        vr = VectorResponse(key="k1", distance=0.5, metadata={"doc": "d1"})
        self.assertEqual(vr.distance, 0.5)

    def test_embedding_response(self):
        er = EmbeddingResponse(id="r1", response_type="float", embeddings=[[0.1]])
        self.assertEqual(er.id, "r1")


if __name__ == '__main__':
    unittest.main()
