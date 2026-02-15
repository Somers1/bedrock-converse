# bedrock-sdk

Python SDK for AWS Bedrock Converse API with tool calling, embeddings, and vector search.

## Install

```bash
pip install bedrock-sdk
# With image resizing support:
pip install bedrock-sdk[image]
```

## Quick Start

```python
from bedrock import Converse, Message

# Basic conversation
converse = Converse(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")
response = converse.invoke("Hello, how are you?")
print(response.content)

# Structured output
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    score: float

structured = converse.with_structured_output(Sentiment)
result = structured.invoke("I love this product!")
print(result.label, result.score)

# Agent with tools
from bedrock import ConverseAgent, tool

@tool
def search(query: str) -> str:
    """Search the web"""
    return f"Results for: {query}"

agent = ConverseAgent(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")
agent.add_tool(search)
result = agent.run("Find information about Python")

# Embeddings
from bedrock import BedrockEmbedding
emb = BedrockEmbedding(region_name="us-east-1")
response = emb.embed_texts(["Hello world"])

# InvokeModel (OpenAI-compatible models on Bedrock)
converse = Converse(model_id="some-model", region_name="us-east-1", use_invoke_model=True)
```

## Features

- **Converse API** — Full support for messages, images, documents, videos, caching, thinking
- **Structured Output** — Pydantic model validation with tool use or JSON extraction
- **Tool Calling** — `@tool` decorator, `Tools` class, `ToolRegistry` with auto Pydantic validation
- **Agent Loop** — `ConverseAgent` with automatic tool execution and exit conditions
- **InvokeModel** — OpenAI-compatible payload conversion for models using `invoke_model`
- **Embeddings** — Cohere embed-v4 support with text, image, and multimodal inputs
- **Vector Store** — S3 Vectors integration with chunking and similarity search
- **Callbacks** — Extensible callback system for logging, metrics (CloudWatch EMF)
- **Prompt Caching** — Automatic cache point management with model compatibility checks
