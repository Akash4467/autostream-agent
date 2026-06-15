# AutoStream AI Agent
### Agentic sales and support workflow demo

AutoStream AI Agent is a LangGraph-powered conversational AI project built as a stateful assistant for handling repetitive business queries. It combines structured knowledge retrieval, intent detection, lead collection, and workflow orchestration in a Streamlit-based interface.

The project began as a fictional SaaS assistant for AutoStream and now serves as a broader AI sales and support automation demo. It can answer product questions, explain plans and policies, detect user intent, collect lead details step by step, and support real-world deployment scenarios.

## Live Demo

The deployed Streamlit app is available at: [http://13.61.188.64:8501](http://13.61.188.64:8501)

## Quick Start (Local)

```bash
# 1. Clone the repository
 git clone https://github.com/YOUR_USERNAME/autostream-agent.git
 cd autostream-agent

# 2. Create and activate virtual environment
 python -m venv .venv
 source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
 pip install -r requirements.txt

# 4. Configure API keys
 cp .env.example .env
 # Add ONE of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY

# 5. Run the Streamlit app
 streamlit run app.py

# Optional CLI mode
 python main.py
```

Open `http://localhost:8501` in your browser.

## Quick Start (Docker)

```bash
# 1. Configure environment
cp .env.example .env
# Add your API key

# 2. Build and run
 docker compose up --build
```

To run in the background:

```bash
docker compose up --build -d
docker compose logs -f
docker compose down
```

To run tests inside Docker:

```bash
docker compose run --rm autostream-agent pytest tests/ -v
```

## Project Structure

```text
autostream-agent/
├── app.py
├── main.py
├── agent/
│   ├── __init__.py
│   ├── graph.py
│   └── rag.py
├── tools/
│   ├── __init__.py
│   └── lead_capture.py
├── knowledge_base/
│   └── autostream_kb.json
├── tests/
│   └── test_agent.py
├── leads.json
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env.example
└── requirements.txt
```

## Features

- Multi-turn conversational workflow using LangGraph
- Intent detection for greetings, pricing, support, policies, and FAQs
- RAG-style response generation using a structured JSON knowledge base
- Lead collection one field at a time
- Tool-based lead capture flow
- Streamlit interface with interactive demo support
- Docker-ready setup for reproducible deployment
- Public AWS deployment for live access

## Architecture

### Why LangGraph?

LangGraph is used because it provides explicit control over state transitions, which is important for multi-step conversational flows such as lead qualification and support automation. This makes the workflow easier to inspect, test, and extend than a loosely controlled prompt-only approach.

### How State Is Managed

The agent uses a typed `AgentState` object to maintain the full conversation history, current intent, captured lead fields, and lead status. On each turn, the workflow updates the state, classifies intent, generates a response, and optionally triggers lead capture when enough information has been collected.

Because the full message history is carried through the graph, the assistant can preserve context across multiple turns without requiring a separate external memory layer for the core demo.

### RAG Pipeline

The project uses a compact JSON knowledge base for company information, plans, pricing, policies, supported platforms, and FAQs. For this project size, the knowledge base is loaded once and injected into the system workflow rather than requiring a heavier vector database setup.

## Knowledge Base

The reference knowledge base stores AutoStream company metadata, pricing plans, support details, cancellation policy, free trial details, supported platforms, and FAQ content.[1]

## URS

### User Requirements Specification

The system is designed to satisfy the following requirements:

- Users should be able to ask questions in natural language.
- The assistant should identify the intent of the request.
- The assistant should answer product, plan, and policy questions from the knowledge base.
- The system should support multi-turn conversation with state retention.
- The agent should collect lead details in a structured sequence.
- The application should be accessible through a simple web interface.
- The project should be deployable locally, through Docker, and on AWS.

## AWS Deployment

The project is deployed on AWS using a publicly accessible Streamlit setup. The application is currently accessible at [http://13.61.188.64:8501](http://13.61.188.64:8501).

### Deployment Flow

1. Prepare the Python environment and project dependencies.
2. Launch an AWS EC2 instance.
3. Transfer the project files to the server.
4. Install dependencies and configure the `.env` file.
5. Run the Streamlit app on port `8501`.
6. Allow inbound traffic on port `8501` in the AWS security group.
7. Access the application via the EC2 public IP.

### Example Run Command

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## WhatsApp Deployment

To extend this project to WhatsApp deployment:

1. Wrap the chat workflow in a Flask or FastAPI webhook.
2. Accept incoming WhatsApp messages through `POST /webhook`.
3. Persist user session state using Redis or SQLite.
4. Send agent responses back using the WhatsApp Cloud API.
5. Handle webhook verification using `hub.challenge` and a verify token.

## Running Tests

```bash
# Local
pytest tests/ -v

# With coverage
pytest tests/ -v --tb=short --cov=agent --cov=tools --cov-report=term-missing

# In Docker
docker compose run --rm autostream-agent pytest tests/ -v
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | One of three | Claude-based model access |
| `OPENAI_API_KEY` | One of three | OpenAI model access |
| `GOOGLE_API_KEY` | One of three | Gemini model access |

Only one provider key is required. The project can be configured based on the available model provider.

## Use Cases

- Sales query automation
- Customer support automation
- FAQ resolution
- Product information assistant
- Lead qualification workflow demo
- Agentic workflow demonstration for interviews, portfolios, or internships

## Future Improvements

- Replace static KB injection with retrieval pipelines when the dataset grows
- Add authentication and admin management
- Expand analytics for lead tracking and interaction monitoring
- Improve fallback handling for ambiguous intent
- Integrate WhatsApp or CRM workflows in production

## Evaluation Checklist

| Criterion | Implementation Area |
|---|---|
| Agent reasoning and intent detection | `agent/graph.py` |
| Knowledge-based response handling | `agent/rag.py` and `autostream_kb.json` |
| Stateful workflow management | `AgentState` and LangGraph flow |
| Proper tool calling | `tools/lead_capture.py` |
| Code clarity and modularity | Separated `agent`, `tools`, and UI structure |
| Real-world deployability | Docker setup and AWS-hosted Streamlit deployment |
