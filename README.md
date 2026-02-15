# Bedrock SDK

A Python SDK for AWS Bedrock that makes prompts readable, tools natural, and agents simple.

## Why?

I built this because the existing options frustrated me:

1. **Prompt formatting sucks.** Every SDK makes you build nested dicts or wrestle with message arrays. I wanted to build prompts the way I think about them — add text, add an image, add a document, send it.

2. **Tool definitions are painful.** Writing JSON schemas by hand or decorating functions with 15 lines of metadata is insane. I wanted `@tool` on a function and done — types inferred, schema generated.

3. **LangChain wasn't keeping up.** When new models dropped on Bedrock, I'd wait weeks for library updates. I needed something I controlled that worked directly with the Bedrock APIs.

4. **Security matters.** AWS Bedrock means my data stays in my AWS account. No third-party API proxies, no data leaving my infrastructure. The Converse API is universal across models — swap Claude for Llama for Nova without changing code.

## Install

```bash
pip install git+https://github.com/Somers1/bedrock-sdk.git
# With image resizing support:
pip install "bedrock-sdk[image]"
```

---

## Prompt Building

The `Message` class lets you build rich, multimodal prompts with chaining. No nested dicts, no format strings — just describe what you want to send.

```python
from bedrock import Converse, Message

converse = Converse(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1"
)

# Simple text
response = converse.invoke("What is the capital of France?")

# Rich prompt with chaining
prompt = Message()
prompt.add_text("Analyse this document and describe the image.")
prompt.add_document(open("report.pdf", "rb").read(), "report")
prompt.add_image(open("chart.png", "rb").read(), "png")
prompt.add_cache_point()  # Cache everything above for repeat calls

response = converse.invoke(prompt)
```

### Adding content

```python
message = Message()

# Text (with optional XML tags)
message.add_text("Summarise this for me")
message.add_text("Some context here", tag="context")  # Wraps in <context>...</context>

# Images (auto-resized if over 8000px when pillow installed)
message.add_image(image_bytes, "png")
message.add_image(image_bytes, "jpeg")

# Documents (PDF, Word, CSV, etc.)
message.add_document(pdf_bytes, "quarterly-report")  # Name auto-cleaned

# Video
from bedrock import Video, VideoSource, S3Location
video = Video(format="mp4", source=VideoSource(
    s3_location=S3Location(uri="s3://bucket/video.mp4", bucket_owner="123456")
))
message.add_video(video)

# Timestamp
from zoneinfo import ZoneInfo
message.add_current_time(tz=ZoneInfo("Australia/Sydney"))

# Cache points (for prompt caching on supported models)
message.add_cache_point()
```

### System prompts

```python
from bedrock import SystemContent

converse.add_system(SystemContent(text="You are a helpful assistant."))
converse.add_system(SystemContent(text="Always respond in JSON."))
converse.add_system_cache_point()  # Cache the system prompt
```

### Conversation history

```python
# Multi-turn conversations are just message lists
converse = Converse(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")

response = converse.converse("What is Python?")
# History is tracked automatically

response = converse.converse("What about its type system?")
# Model sees the full conversation
```

---

## Tool Calling

### The `@tool` decorator

Decorate any function. Types are inferred, schema is generated. That's it.

```python
from bedrock import tool

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city"""
    return f"22°C and sunny in {city}"

@tool
def search_database(query: str, limit: int = 10) -> list:
    """Search the knowledge base"""
    return [{"title": "Result 1", "score": 0.95}]
```

The decorator reads your type hints and docstring to generate the tool schema automatically. Optional parameters (those with defaults) are marked as optional in the schema. The docstring becomes the tool description.

### Pydantic models as parameters

For complex inputs, use Pydantic models — the schema is generated from the model definition:

```python
from pydantic import BaseModel

class SearchFilter(BaseModel):
    category: str
    min_price: float
    max_price: float

@tool
def filtered_search(query: str, filters: SearchFilter) -> list:
    """Search with filters"""
    return []
```

### The `Tools` class

Group related tools into a class. Methods are auto-discovered as tools:

```python
from bedrock import Tools

class DatabaseTools(Tools):
    def __init__(self, connection):
        self.db = connection

    def query_users(self, name: str, active: bool = True) -> list:
        """Find users by name"""
        return self.db.query(f"SELECT * FROM users WHERE name LIKE '%{name}%'")

    def get_user(self, user_id: int) -> dict:
        """Get a specific user by ID"""
        return self.db.get("users", user_id)

    def update_status(self, user_id: int, status: str) -> str:
        """Update a user's status"""
        self.db.update("users", user_id, {"status": status})
        return f"Updated user {user_id} to {status}"
```

Tool names are automatically prefixed with the class name (e.g. `DatabaseTools_query_users`). Instance state is preserved — use `self` for database connections, API clients, whatever.

### Binding tools to a conversation

```python
converse = Converse(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")

# Single tools
converse.add_tool(get_weather)
converse.add_tool(search_database)

# Tool class instances
db_tools = DatabaseTools(my_connection)
converse.add_tool(db_tools)

# Or bind a list
converse.bind_tools([get_weather, search_database, db_tools])

# Force a specific tool
converse.set_tool_choice("get_weather")
```

---

## Agent Loop

`ConverseAgent` runs a full tool-execution loop. The model calls tools, results feed back in, repeat until the model calls the exit tool or hits max iterations.

```python
from bedrock import ConverseAgent, tool, Tools

@tool
def search(query: str) -> str:
    """Search the web for information"""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression"""
    return str(eval(expression))

agent = ConverseAgent(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    max_iterations=15,
    debug=True  # Logs tool calls
)
agent.add_tool(search)
agent.add_tool(calculate)

# Run — agent loops until it has a final answer
result = agent.run("What is the population of Tokyo divided by the area in km²?")
print(result)  # "The population density of Tokyo is approximately..."
```

### Structured output from agents

Force the agent to return a specific Pydantic model:

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    sentiment: str
    confidence: float

agent.with_structured_output(Analysis)
result = agent.run("Analyse this customer review: 'Great product, fast shipping!'")
print(result.summary)     # "Positive review highlighting product quality..."
print(result.sentiment)   # "positive"
print(result.confidence)  # 0.95
```

Works with `List[Model]` too:

```python
from typing import List

class Entity(BaseModel):
    name: str
    type: str

agent.with_structured_output(List[Entity])
entities = agent.run("Extract entities from: 'John works at Google in Sydney'")
# [Entity(name="John", type="person"), Entity(name="Google", type="org"), ...]
```

### How the loop works

```
User message
    │
    ▼
┌──────────────┐
│  LLM Call    │ ──→ Text response? → Return it (or Finish tool)
│  with tools  │ ──→ Tool calls?  → Execute them ─┐
└──────────────┘                                    │
    ▲                                               │
    └── Tool results fed back as messages ──────────┘
    
Repeats until: exit tool called, text-only response, or max_iterations hit
```

---

## Structured Output (without agent)

For single-call structured extraction (no tool loop):

```python
from bedrock import Converse, StructuredConverse
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    score: float
    reasoning: str

converse = Converse(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")
structured = converse.with_structured_output(Sentiment)
result = structured.invoke("I absolutely love this product!")
print(result.label)  # "positive"
```

---

## Thinking (Extended Reasoning)

Enable Claude's extended thinking for complex problems:

```python
converse = Converse(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")
converse.with_thinking(tokens=2048)

response = converse.invoke("Solve this step by step: what is 127 * 843 + 291?")
```

---

## Embeddings

Cohere embed-v4 on Bedrock for text, image, and multimodal embeddings:

```python
from bedrock import BedrockEmbedding

emb = BedrockEmbedding(
    model_id="global.cohere.embed-v4:0",
    region_name="us-east-1",
    output_dimension=1536
)

# Text embeddings
response = emb.embed_texts(["Hello world", "Another document"])
vectors = response.embeddings  # List of float vectors

# Query embedding (for search)
query_vector = emb.embed_query("search term")

# Document embeddings (batch)
doc_vectors = emb.embed_documents(["doc 1 text", "doc 2 text"])

# Image embeddings
img_vectors = emb.embed_images([base64_image_string])

# Multimodal
from bedrock import MultimodalInput
inp = MultimodalInput().add_text("A cat").add_image(base64_data, "image/png")
response = emb.embed_multimodal([inp])
```

### Text Chunking

```python
from bedrock import TextChunker

chunker = TextChunker(ChunkerConfig(max_tokens=900, overlap_tokens=100))
chunks = chunker.chunk(long_document_text)
# Each chunk is ~900 tokens with 100-token overlap
```

### Vector Store (S3 Vectors)

```python
from bedrock import S3VectorsStore, VectorItem

store = S3VectorsStore(
    vector_bucket="my-vectors",
    index_name="documents",
    region_name="us-east-1"
)

# Index documents
items = [VectorItem(key="doc-1", vector=embedding, metadata={"title": "Doc 1"})]
store.put_vectors(items)

# Search
results = store.query_text("find similar documents")
for r in results:
    print(r.key, r.distance, r.metadata)
```

---

## Callbacks

Monitor cost, latency, and usage:

```python
from bedrock import Converse, PrintCallback

converse = Converse(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    callbacks=[PrintCallback()]
)
# Prints token usage and cost after each call
```

Custom callbacks:

```python
from bedrock import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_converse_start(self, converse):
        print(f"Starting call to {converse.model_id}")

    def on_converse_end(self, response):
        print(f"Used {response.usage.total_tokens} tokens, cost ${response.cost.total_cost:.4f}")
```

---

## InvokeModel (OpenAI-compatible)

Some Bedrock models use the InvokeModel API with OpenAI-format payloads instead of the Converse API. Set `use_invoke_model=True` and the SDK handles the conversion:

```python
converse = Converse(
    model_id="some-openai-compat-model",
    region_name="us-east-1",
    use_invoke_model=True
)
# Same API — prompts, tools, everything works the same
response = converse.invoke("Hello")
```

---

## Async Support

All main methods have async variants:

```python
response = await converse.ainvoke("Hello")
response = await converse.aconverse("Hello")
result = await emb.aembed_texts(["Hello"])
```

---

## Configuration

```python
from bedrock import Converse, ConverseInferenceConfig

converse = Converse(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    # Optional: explicit AWS credentials (otherwise uses default chain)
    aws_access_key_id="...",
    aws_secret_access_key="...",
    # Inference config
    inference_config=ConverseInferenceConfig(
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        stop_sequences=["\n\nHuman:"]
    ),
)
```

## License

MIT
