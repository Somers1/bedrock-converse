import asyncio
import base64
import json
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Iterable
from typing import List, Literal, Optional, Union, Dict, Any

import boto3
from botocore.config import Config


@dataclass
class MultimodalInput:
    content: List[Dict[str, str]] = field(default_factory=list)

    def add_text(self, text: str):
        self.content.append({"type": "text", "text": text})
        return self

    def add_image(self, base64_data: str, mime_type: str = "image/png"):
        self.content.append({
            "type": "image_url",
            "image_url": f"data:{mime_type};base64,{base64_data}"
        })
        return self

    def add_image_from_file(self, file_path: str, mime_type: str = "image/png"):
        with open(file_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
        return self.add_image(base64_data, mime_type)


@dataclass
class EmbeddingResponse:
    id: str
    response_type: str
    embeddings: Union[List[List[float]], Dict[str, List[List]]]
    texts: Optional[List[str]] = None
    inputs: Optional[List[Dict]] = None


@dataclass
class BedrockEmbedding:
    model_id: str = "global.cohere.embed-v4:0"
    region_name: str = 'ap-southeast-2'
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    _client: boto3.client = None

    input_type: Literal["search_document", "search_query", "classification", "clustering"] = "search_document"
    embedding_types: List[Literal["float", "int8", "uint8", "binary", "ubinary"]] = field(
        default_factory=lambda: ["float"])
    output_dimension: Literal[256, 512, 1024, 1536] = 1536
    truncate: Literal["NONE", "LEFT", "RIGHT"] = "RIGHT"
    max_tokens: int = 128000

    @property
    def client(self):
        if self._client is None:
            if self.aws_access_key_id and self.aws_secret_access_key:
                session = boto3.Session(
                    region_name=self.region_name,
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key
                )
            else:
                session = boto3.Session(region_name=self.region_name)
            self._client = session.client('bedrock-runtime')
        return self._client

    def _invoke(self, body: dict) -> EmbeddingResponse:
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            accept='*/*',
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read())
        return EmbeddingResponse(**response_body)

    async def _ainvoke(self, body: dict) -> EmbeddingResponse:
        loop = asyncio.get_event_loop()
        response_dict = await loop.run_in_executor(
            None,
            lambda: self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                accept='*/*',
                contentType='application/json'
            )
        )
        response_body = json.loads(response_dict['body'].read())
        return EmbeddingResponse(**response_body)

    def embed_texts(self, texts: List[str], input_type: Optional[str] = None) -> EmbeddingResponse:
        body = {
            "texts": texts,
            "input_type": input_type or self.input_type,
            "embedding_types": self.embedding_types,
            "output_dimension": self.output_dimension,
            "truncate": self.truncate,
            "max_tokens": self.max_tokens
        }
        return self._invoke(body)

    async def aembed_texts(self, texts: List[str], input_type: Optional[str] = None) -> EmbeddingResponse:
        body = {
            "texts": texts,
            "input_type": input_type or self.input_type,
            "embedding_types": self.embedding_types,
            "output_dimension": self.output_dimension,
            "truncate": self.truncate,
            "max_tokens": self.max_tokens
        }
        return await self._ainvoke(body)

    def embed_images(self, images: List[str], input_type: Optional[str] = None) -> EmbeddingResponse:
        body = {
            "images": images,
            "input_type": input_type or self.input_type,
            "embedding_types": self.embedding_types,
            "output_dimension": self.output_dimension,
            "truncate": self.truncate,
            "max_tokens": self.max_tokens
        }
        return self._invoke(body)

    async def aembed_images(self, images: List[str], input_type: Optional[str] = None) -> EmbeddingResponse:
        body = {
            "images": images,
            "input_type": input_type or self.input_type,
            "embedding_types": self.embedding_types,
            "output_dimension": self.output_dimension,
            "truncate": self.truncate,
            "max_tokens": self.max_tokens
        }
        return await self._ainvoke(body)

    def embed_multimodal(self, inputs: List[MultimodalInput], input_type: Optional[str] = None) -> EmbeddingResponse:
        body = {
            "inputs": [{"content": inp.content} for inp in inputs],
            "input_type": input_type or self.input_type,
            "embedding_types": self.embedding_types,
            "output_dimension": self.output_dimension,
            "truncate": self.truncate,
            "max_tokens": self.max_tokens
        }
        return self._invoke(body)

    async def aembed_multimodal(self, inputs: List[MultimodalInput],
                                input_type: Optional[str] = None) -> EmbeddingResponse:
        body = {
            "inputs": [{"content": inp.content} for inp in inputs],
            "input_type": input_type or self.input_type,
            "embedding_types": self.embedding_types,
            "output_dimension": self.output_dimension,
            "truncate": self.truncate,
            "max_tokens": self.max_tokens
        }
        return await self._ainvoke(body)

    def embed_query(self, text: str):
        return self.embed_texts([text], input_type="search_query").embeddings['float'][0]

    async def aembed_query(self, text: str) -> EmbeddingResponse:
        return await self.aembed_texts([text], input_type="search_query")

    def embed_documents(self, texts: List[str]) -> EmbeddingResponse:
        return self.embed_texts(texts, input_type="search_document")

    async def aembed_documents(self, texts: List[str]) -> EmbeddingResponse:
        return await self.aembed_texts(texts, input_type="search_document")


@dataclass
class ChunkerConfig:
    max_tokens: int = 900
    overlap_tokens: int = 100
    preserve_newlines: bool = True


class TextChunker:
    def __init__(self, config: ChunkerConfig = ChunkerConfig()):
        assert config.max_tokens > 0
        assert 0 <= config.overlap_tokens < config.max_tokens
        self.cfg = config

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        t = text if self.cfg.preserve_newlines else " ".join(text.split())
        words = t.split()
        if not words:
            return []

        max_w = self.cfg.max_tokens
        ov_w = self.cfg.overlap_tokens
        step = max_w - ov_w

        chunks: List[str] = []
        i = 0
        while i < len(words):
            window = words[i:i + max_w]
            chunks.append(" ".join(window))
            if i + max_w >= len(words):
                break
            i += step
        return chunks


@dataclass
class VectorItem:
    key: str
    vector: Dict[str, List[float]]
    metadata: Dict[str, Any]


@dataclass
class VectorResponse:
    key: str
    distance: float
    metadata: Dict[str, Any]


class S3VectorsStore:
    def __init__(
            self,
            vector_bucket: str,
            index_name: str,
            region_name: str = "ap-southeast-2",
            max_batch: int = 500,
            retries: int = 4,
            backoff_base: float = 0.6,
            aws_access_key_id: Optional[str] = None,
            aws_secret_access_key: Optional[str] = None,
    ):
        self.vector_bucket = vector_bucket
        self.index_name = index_name
        self.max_batch = max_batch
        self.retries = retries
        self.backoff_base = backoff_base

        session = boto3.Session(
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self.client = session.client(
            "s3vectors",
            config=Config(retries={"max_attempts": 0})  # we do our own backoff
        )

    def _chunks(self, items: List[VectorItem], n: int) -> Iterable[List[VectorItem]]:
        for i in range(0, len(items), n):
            yield items[i:i + n]

    def put_vectors(self, items: List[VectorItem]) -> None:
        if not items:
            return
        for batch in self._chunks(items, self.max_batch):
            payload = {
                "vectorBucketName": self.vector_bucket,
                "indexName": self.index_name,
                "vectors": [
                    {
                        "key": it.key,
                        "data": {'float32': it.vector},
                        "metadata": it.metadata
                    } for it in batch
                ]
            }
            attempt = 0
            while True:
                try:
                    self.client.put_vectors(**payload)
                    break
                except self.client.exceptions.TooManyRequestsException as e:
                    if attempt >= self.retries:
                        raise
                    time.sleep(self.backoff_base * (2 ** attempt))
                    attempt += 1
                except Exception:
                    if attempt >= self.retries:
                        raise
                    time.sleep(self.backoff_base * (2 ** attempt))
                    attempt += 1

    def query(self, vector, query_filter=None):
        payload = {
            "vectorBucketName": self.vector_bucket,
            "indexName": self.index_name,
            'topK': 30,
            "queryVector": {'float32': vector},
            'filter': query_filter,
            'returnMetadata': True,
            'returnDistance': True,
        }
        return [VectorResponse(**v) for v in self.client.query_vectors(**payload)['vectors']]

    def query_text(self, text, query_filter=None):
        return self.query(BedrockEmbedding().embed_query(text), query_filter)

    def query_n(self, text: str, n: int, query_filter: Optional[Dict[str, Any]] = None, max_passes: int = 12):
        q = BedrockEmbedding().embed_query(text)
        best, seen = {}, set()
        for _ in range(max_passes):
            if len(best) >= n: break
            filters = {"$and": ([query_filter] if query_filter else []) + (
                [{"document_id": {"$nin": list(seen)}}] if seen else [])} or None
            hits = self.query(q, filters)
            if not hits: break
            for h in hits:
                doc = (h.metadata or {}).get("document_id")
                if not doc: continue
                seen.add(doc)
                d = float(h.distance)
                if doc not in best or d < best[doc]:
                    best[doc] = d
        return [{"documentId": k, "distance": best[k]} for k in sorted(best, key=best.get)][:n]

    def delete_vectors_document_id(self, keys: list) -> None:
        if not keys:
            return
        for i in range(0, len(keys), self.max_batch):
            batch = keys[i:i + self.max_batch]
            attempt = 0
            while True:
                try:
                    self.client.delete_vectors(
                        vectorBucketName=self.vector_bucket,
                        indexName=self.index_name,
                        keys=batch
                    )
                    break
                except self.client.exceptions.TooManyRequestsException:
                    if attempt >= self.retries:
                        raise
                    time.sleep(self.backoff_base * (2 ** attempt))
                    attempt += 1
                except Exception:
                    if attempt >= self.retries:
                        raise
                    time.sleep(self.backoff_base * (2 ** attempt))
                    attempt += 1

