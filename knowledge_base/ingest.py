import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Load JSON ─────────────────────────────
with open("knowledge_base/autostream_kb.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = []

# ── Company ───────────────────────────────
docs.append(Document(
    page_content=f"{data['company']} - {data['tagline']}",
    metadata={"type": "company"}
))

# ── Plans ────────────────────────────────
for plan_key, plan in data["plans"].items():
    docs.append(Document(
        page_content=f"{plan['name']} costs {plan['price_monthly']} per month. Features: {', '.join(plan['features'])}",
        metadata={"type": "plan", "plan": plan_key}
    ))

# ── Policies ─────────────────────────────
for key, value in data["policies"].items():
    docs.append(Document(
        page_content=f"{key.replace('_',' ')}: {value}",
        metadata={"type": "policy"}
    ))

# ── Platforms ────────────────────────────
docs.append(Document(
    page_content="Supported platforms: " + ", ".join(data["supported_platforms"]),
    metadata={"type": "platforms"}
))

# ── FAQ ───────────────────────────────────
for faq in data["faq"]:
    docs.append(Document(
        page_content=f"Q: {faq['question']} A: {faq['answer']}",
        metadata={"type": "faq"}
    ))

# ── Split ────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)
print("✅ Total chunks:", len(chunks))

# ── Embeddings (ONLY SAFE OPTION) ─────────
print("🟢 Using HuggingFace embeddings (stable)")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ── Vector DB ─────────────────────────────
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vectorstore.persist()

print("✅ Vector DB created successfully!")