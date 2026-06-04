# Bedrock Converse

A Python SDK for AWS Bedrock that makes prompts readable, tools natural, and agents simple.

## Why?

AWS Bedrock's Converse API is a single unified interface across every model — Claude, Llama, Nova, Mistral, Cohere, Kimi. Same prompt format, same tool calling, same response structure. Write once, swap models with a string. Your data stays in your AWS account.

This SDK is a lightweight wrapper built specifically around the Converse API. If that's all you need, it's simpler and more direct than full frameworks like LangChain (which is great for broader orchestration, but carries a lot of weight if you just want to talk to Bedrock).

The goals were straightforward:

1. **Readable prompts.** Build messages by chaining `.add_text()`, `.add_image()`, `.add_document()` — not nested dicts.
2. **Natural tool definitions.** Put `@tool` on a function. Types are inferred, schema is generated. Done.
3. **Stay current.** When new models land on Bedrock, they work immediately through the Converse API — no waiting for library updates.
4. **Keep it small.** Minimal dependencies, no framework lock-in.

## Install

```bash
pip install bedrock-converse
# With image resizing support:
pip install "bedrock-converse[image]"
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

### Prompt caching across the loop

A single agent turn makes one Bedrock call per tool iteration, each re-sending the system prompt, tools, and the whole growing message history. `with_prompt_caching()` places **one rolling cache point on the last message before every call**, so the growing conversation is written once and read cheaply on each subsequent call:

```python
agent = ConverseAgent(model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0", region_name="us-east-1")
agent.bind_tools([search, calculate])
agent.with_prompt_caching(ttl="1h")   # single TTL; defaults to "5m"
```

Tool-loop iterations and new user turns are the same thing to the cache — both append to the message list — so one moving point covers both. The point is injected into the request payload only, so your `Message` objects are never mutated and persisted history stays clean. It's **non-destructive**: it only ever appends to the last message and skips even that if you've already placed a point there, so your own cache points are never moved or removed. Unsupported models are skipped automatically.

**Static / tiered caching is yours to place.** The SDK does not cache the system prompt or tools — add those points where you build the agent (`add_system_cache_point()`, a tools cache point, per-user prefixes), most-shared first, staying within Bedrock's 4-breakpoint limit. See [docs/prompt-caching.md](docs/prompt-caching.md) for the full design, the TTL-invariant rule behind it, and the experiments that established it.

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

### Controlling how the loop ends

By default the agent auto-binds a `Finish` tool, so the model ends the turn by calling it with a final message. You can change that:

```python
# Chat agents that end by replying in plain text, not via a tool call
agent.auto_exit_tool = False

# Designate an already-bound tool as the exit (matches by name suffix)
agent.bind_exit_tool("send_message")
```

Or mark a method on a `Tools` class with `@exit_tool`, and it becomes the exit automatically when bound:

```python
from bedrock import Tools, exit_tool

class ChatTools(Tools):
    @exit_tool
    def send_message(self, text: str) -> str:
        """Send the final reply to the user"""
        return text
```

### Agent hooks

```python
# Called when the agent replies with text instead of tools. Return a value to end the
# loop with it; return None to let the loop continue.
agent.on_text(lambda text: text if "done" in text else None)

# Called with (tool_name, content) for each successful tool result before it's sent to
# the model. Return replacement content to substitute (e.g. offload an oversize result
# to a file and swap in a short pointer), or None to keep the original. Runs in-loop, so
# the substituted content is what gets cached and persisted.
agent.on_tool_result(lambda name, content: offload(content) if too_big(content) else None)
```

### Parallel tool execution

When the model emits several tool calls in one turn, run them concurrently instead of one-by-one. The exit tool and model-switch tools stay serial, so the ref registry and history are never touched off-thread:

```python
agent.parallel_tools = True
# Optional wrapper around each pooled tool body for per-thread setup/teardown:
agent.tool_thread_hook = lambda fn: run_with_db_connection(fn)
```

### Switching models mid-conversation

Decorate a tool with `model_switch` and the agent swaps the conversation onto a different `Converse` before re-running the call — e.g. escalate to a stronger model when the task gets hard:

```python
from bedrock import model_switch, Converse

strong = Converse(model_id="us.anthropic.claude-opus-4-20250514-v1:0", region_name="us-east-1")

@model_switch(strong, message="Escalating to the larger model.")
def escalate() -> str:
    """Hand off to a more capable model for hard sub-problems"""
    return "switched"
```

---

## Streaming

`invoke`, `converse`, and the agent `run` all take `stream=True` and yield normalized events as they arrive. The stream returns the final `ConverseResponse` (via `StopIteration.value`) once exhausted, and history is updated automatically:

```python
for event in converse.invoke("Write a haiku about the sea", stream=True):
    if event["type"] == "text_delta":
        print(event["text"], end="", flush=True)
```

For the agent loop, `run(stream=True)` (or `stream_run(...)`) interleaves model-output events with loop events — `iteration_start`, `tool_call`, `tool_result`, and a final `done` carrying the result:

```python
for event in agent.run("Research and summarise X", stream=True):
    match event["type"]:
        case "text_delta":      print(event["text"], end="")
        case "tool_call":       print(f"\n→ {event['name']}({event['input']})")
        case "tool_result":     print(f"  ✓ {event['status']}")
        case "done":            final = event["result"]
```

Event types include `message_start`, `content_block_start`, `text_delta`, `reasoning_delta`, `tool_use_input_delta`, `content_block_stop`, `message_stop`, and `metadata`. If a streamed tool call arrives corrupt (a dropped Bedrock delta frame), the loop emits a `stream_reset` event and re-requests the stream up to `stream_retries` times (default 2) rather than guessing the arguments.

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

If a model returns output that won't validate against your schema, fall back to another model automatically:

```python
structured.with_backup_model("us.anthropic.claude-opus-4-20250514-v1:0")
# Pass a model id, or a fully-configured Converse instance to fall back to.
```

---

## Thinking (Extended Reasoning)

Enable Claude's extended thinking for complex problems:

```python
converse = Converse(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")
converse.with_thinking(tokens=2048)

response = converse.invoke("Solve this step by step: what is 127 * 843 + 291?")
```

On Claude 4.5+ models that support **adaptive thinking**, let the model decide its own budget by effort level instead of a fixed token count:

```python
converse.with_adaptive_thinking(effort="high")  # "low" | "medium" | "high"
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

OpenAI-compatible embedding endpoints such as Fireworks can use the same helper methods:

```python
from bedrock import OpenAIEmbedding

emb = OpenAIEmbedding(
    model_id="accounts/fireworks/models/qwen3-embedding-8b",
    base_url="https://api.fireworks.ai/inference/v1",
    api_key="fw_...",
)

query_vector = emb.embed_query("search term")
doc_vectors = emb.embed_documents(["doc 1 text", "doc 2 text"])
```

If you're using Bedrock Mantle's OpenAI-compatible endpoint, use `MantleEmbedding` instead.

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

Handlers can also hook the agent loop — `on_run_start`, `on_tool_start`, `on_tool_end` — for per-tool tracing.

### Langfuse tracing

`LangfuseCallback` traces every call, tool execution, and token/cost figure to [Langfuse](https://langfuse.com) out of the box:

```python
from bedrock import Converse
from bedrock.langfuse_callback import LangfuseCallback

converse = Converse(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    callbacks=[LangfuseCallback(user_id="u-123", session_id="s-456", tags=["prod"])],
)
```

---

## Mantle (OpenAI-compatible endpoint)

Some models reach Bedrock through [Bedrock Mantle](https://docs.aws.amazon.com/bedrock/) — an OpenAI-compatible endpoint — rather than the Converse API. `Mantle`, `MantleAgent`, and `StructuredMantle` are drop-in subclasses of `Converse`, `ConverseAgent`, and `StructuredConverse`: the entire API (prompt building, tools, structured output, the agent loop, thinking, streaming) is identical — only the transport changes.

```python
from bedrock import Mantle, MantleAgent

# Endpoint is derived from region (https://bedrock-mantle.{region}.api.aws/v1)
mantle = Mantle(model_id="openai.gpt-oss-120b-1:0", region_name="us-east-1")
response = mantle.invoke("Hello")

# Or point at any OpenAI-compatible endpoint explicitly
mantle = Mantle(
    model_id="moonshotai/kimi-k2",
    base_url="https://my-gateway/v1",
    api_key="sk-...",
)
```

`base_url` falls back to `MANTLE_ENDPOINT`, and `api_key` to `MANTLE_API_KEY`. Use `with_cache_key(key)` to set the OpenAI `prompt_cache_key` for prefix-cache routing. Set `api_mode="responses"` to drive the OpenAI **Responses** API instead of Chat Completions — useful for reasoning models that stream summarised thinking:

```python
agent = MantleAgent(model_id="openai.gpt-5", region_name="us-east-1", api_mode="responses")
agent.bind_tools([search, calculate])
agent.with_thinking("medium")  # mapped to reasoning_effort low/medium/high
result = agent.run("...")
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
