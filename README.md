# AutoStream AI Agent
### Social-to-Lead Agentic Workflow — ServiceHive / Inflx Intern Assignment

A LangGraph-powered conversational AI agent for AutoStream, a fictional SaaS company
offering automated video editing tools for content creators.

The agent handles greetings, answers product questions via RAG, detects high-intent users,
collects lead details one field at a time, and fires a lead-capture tool — all within a
stateful, multi-turn conversation. A Streamlit UI with a live lead dashboard is included.

## Quick Start (local)
a
```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent

# 2. Create virtualenv & install deps
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Set your LLM API key
cp .env.example .env
# Edit .env — add ONE of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY

# 4. Launch Streamlit UI
streamlit run app.py

# — or — plain CLI
python main.py
```

Open **http://localhost:8501** in your browser.

---

## Quick Start (Docker)

```bash
# 1. Set your key
cp .env.example .env
# Edit .env — add your API key

# 2. Build & run
docker compose up --build

# Open http://localhost:8501
```

To run in the background:
```bash
docker compose up --build -d
docker compose logs -f          # tail logs
docker compose down             # stop & remove
```

To run tests inside Docker:
```bash
docker compose run --rm autostream-agent pytest tests/ -v
```

---

## Project Structure

```
autostream-agent/
├── app.py                          ← Streamlit UI (main entry point)
├── main.py                         ← CLI fallback
│
├── agent/
│   ├── __init__.py
│   ├── graph.py                    ← LangGraph state machine + intent classifier
│   └── rag.py                      ← KB loader & context builder
│
├── tools/
│   ├── __init__.py
│   └── lead_capture.py             ← mock_lead_capture() + LangChain @tool wrapper
│
├── knowledge_base/
│   └── autostream_kb.json          ← Pricing, policies, FAQ
│
├── tests/
│   └── test_agent.py               ← Unit tests (21 tests)
│
├── leads.json                      ← Auto-created; persists captured leads to disk
│
├── Dockerfile                      ← Multi-stage Docker build
├── docker-compose.yml              ← Compose with named volume for lead persistence
├── .dockerignore
├── .gitignore
├── .env.example
└── requirements.txt
```

---

## Architecture

### Why LangGraph?

LangGraph was chosen over AutoGen because it gives explicit, inspectable control over
**state transitions** — critical for a lead-qualification flow where the agent must
remember partial data (e.g. name collected but not email yet) across multiple turns without
hallucinating or jumping ahead. AutoGen's multi-agent model adds orchestration overhead
that isn't necessary for a single-agent, linear qualification flow. LangGraph's `StateGraph`
makes the data flow transparent and straightforward to unit-test.

### How State Is Managed

The agent uses a typed `AgentState` TypedDict with seven fields: the full `messages` list
(accumulated via LangGraph's `add_messages` reducer), the latest `intent` label, three
lead-collection fields (`lead_name`, `lead_email`, `lead_platform`), a `lead_captured`
boolean, and the assigned `lead_id`.

Every user turn, the complete state object is passed into the single `respond` node, which:
1. Appends the new `HumanMessage` and classifies intent via keyword matching (no extra LLM call).
2. Calls the LLM with the full history + system prompt (KB embedded at startup).
3. If the LLM emits a `capture_lead` tool call, executes it and stores lead fields.
4. Runs a follow-up LLM call to produce the final human-facing reply.
5. Returns updated state.

Because the **full conversation history** travels inside `AgentState.messages`, context is
retained across 5–6 turns with no external memory store.

### RAG Pipeline

`autostream_kb.json` is loaded once at startup and serialised into a compact context block
injected directly into the LLM system prompt. For this KB size (pricing, policies, FAQ),
inline injection outperforms a vector-search pipeline in both latency and accuracy — no
embedding model or vector database is needed.

---

## WhatsApp Deployment

To deploy this agent on WhatsApp:

1. **Webhook endpoint** — wrap `chat()` in a Flask/FastAPI handler that accepts `POST /webhook`.
   WhatsApp sends `entry[].changes[].value.messages[]`; extract `from` (sender's phone) and
   `text.body` (the message text).

2. **Session persistence** — because HTTP is stateless, serialise `AgentState` to JSON and
   store it in Redis or SQLite keyed on the sender's phone number. Load at request start,
   call `chat(user_input, state)`, save the updated state back.

3. **Reply** — call the WhatsApp Cloud API
   (`POST /v18.0/{phone_number_id}/messages`) with the agent's reply text.

4. **Verification handshake** — handle `GET /webhook` by returning `hub.challenge` when
   `hub.verify_token` matches your configured secret.

---

## Running Tests

```bash
# Local
pytest tests/ -v

# With coverage report
pytest tests/ -v --tb=short --cov=agent --cov=tools --cov-report=term-missing

# Inside Docker
docker compose run --rm autostream-agent pytest tests/ -v
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | One of three | Claude Haiku (claude-haiku-4-5) |
| `OPENAI_API_KEY`    | One of three | GPT-4o-mini |
| `GOOGLE_API_KEY`    | One of three | Gemini 1.5 Flash |

Only **one** key is needed. The agent auto-detects the provider in priority order:
Anthropic → OpenAI → Google.

---

## Evaluation Checklist

| Criterion | Where |
|---|---|
| Agent reasoning & intent detection | `agent/graph.py` → `classify_intent()` + system prompt |
| Correct RAG usage | `agent/rag.py` → KB injected into system prompt at startup |
| Clean state management | `AgentState` TypedDict; full history carried per turn |
| Proper tool calling | `tools/lead_capture.py` → fires only after all 3 fields confirmed |
| Code clarity & structure | Typed, documented, one responsibility per file |
| Real-world deployability | Dockerised with multi-stage build + named volume for persistence |
