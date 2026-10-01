from pathlib import Path
from typing import TypedDict

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END

from crewai import Agent, Task, Crew, LLM

import mlflow


# ============================================================
# W11D5 - TRACKED AND EVALUATED RAG PIPELINE
# ============================================================

print("=" * 70)
print("W11D5 - TRACKED AND EVALUATED RAG PIPELINE")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE = str(
    BASE_DIR / "day4-ragas-evaluation" / "knowledge_base.txt"
)

OLLAMA_URL = "http://127.0.0.1:11434"

LLM_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
RETRIEVAL_K = 3


# ============================================================
# 1. LOAD KNOWLEDGE BASE
# ============================================================

print("\nLoading knowledge base...")

loader = TextLoader(
    KNOWLEDGE_BASE,
    encoding="utf-8"
)

documents = loader.load()

print(f"Documents loaded: {len(documents)}")


# ============================================================
# 2. CREATE DOCUMENT CHUNKS
# ============================================================

print("\nCreating document chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Chunk overlap: {CHUNK_OVERLAP}")


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

print("\nCreating embeddings...")
print(f"Embedding model: {EMBEDDING_MODEL}")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_URL
)


# ============================================================
# 4. CREATE CHROMADB VECTOR STORE
# ============================================================

print("\nCreating ChromaDB vector store...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="w11d5_tracked_rag"
)

print("ChromaDB vector store created successfully.")


# ============================================================
# 5. CREATE RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": RETRIEVAL_K
    }
)

print("Retriever created.")
print(f"Retrieval k: {RETRIEVAL_K}")


# ============================================================
# 6. CREATE LOCAL OLLAMA LLM
# ============================================================

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0
)

print(f"LLM: {LLM_MODEL}")


# ============================================================
# 7. DEFINE LANGGRAPH STATE
# ============================================================

class RAGState(TypedDict):
    question: str
    context: str
    answer: str


# ============================================================
# 8. LANGGRAPH RETRIEVAL NODE
# ============================================================

def retrieve_documents(state: RAGState):
    question = state["question"]

    retrieved_documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )

    return {
        "question": question,
        "context": context
    }


# ============================================================
# 9. LANGGRAPH GENERATION NODE
# ============================================================

def generate_answer(state: RAGState):
    question = state["question"]
    context = state["context"]

    prompt = f"""
You are a helpful RAG assistant.

Answer the question using only the supplied context.

If the context does not contain enough information, say that the
information is not available in the provided knowledge base.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# 10. BUILD LANGGRAPH WORKFLOW
# ============================================================

graph_builder = StateGraph(RAGState)

graph_builder.add_node(
    "retrieve",
    retrieve_documents
)

graph_builder.add_node(
    "generate",
    generate_answer
)

graph_builder.add_edge(
    START,
    "retrieve"
)

graph_builder.add_edge(
    "retrieve",
    "generate"
)

graph_builder.add_edge(
    "generate",
    END
)

rag_graph = graph_builder.compile()

print("LangGraph RAG workflow compiled successfully.")


# ============================================================
# 11. CREATE CREWAI LOCAL OLLAMA LLM
# ============================================================

crew_llm = LLM(
    model="ollama/qwen2.5:3b",
    base_url=OLLAMA_URL
)


# ============================================================
# 12. CREATE CREWAI AGENT
# ============================================================

agent = Agent(
    role="RAG Answer Reviewer",
    goal="Review the generated RAG answer for relevance, clarity, and consistency with the retrieved context.",
    backstory=(
        "You are an AI reviewer working inside a tracked RAG pipeline. "
        "You review answers generated from retrieved knowledge-base content."
    ),
    llm=crew_llm,
    verbose=False
)

print("CrewAI agent created successfully.")


# ============================================================
# 13. RUN RAG QUERY
# ============================================================

print("\n" + "=" * 70)
print("RUNNING RAG QUERY")
print("=" * 70)

question = "What is Retrieval-Augmented Generation?"

result = rag_graph.invoke(
    {
        "question": question,
        "context": "",
        "answer": ""
    }
)

answer = result["answer"]

retrieved_documents = retriever.invoke(question)

context = "\n\n".join(
    document.page_content
    for document in retrieved_documents
)

print("\nLangGraph retrieval completed.")
print(f"Retrieved documents: {len(retrieved_documents)}")

print("\nQUESTION:")
print(question)

print("\nGENERATED ANSWER:")
print(answer)


# ============================================================
# 14. CREWAI REVIEW
# ============================================================

print("\n" + "=" * 70)
print("CREWAI REVIEW")
print("=" * 70)

review_task = Task(
    description=f"""
Review the following RAG answer.

Question:
{question}

Retrieved context:
{context}

Generated answer:
{answer}

Provide a short review containing:
1. Relevance
2. Context consistency
3. Clarity
4. One improvement suggestion

Keep the review concise.
""",
    expected_output=(
        "A concise review covering relevance, context consistency, "
        "clarity, and one improvement suggestion."
    ),
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[review_task],
    verbose=False
)

crew_result = crew.kickoff()

print(crew_result)


# ============================================================
# 15. PREPARE RAGAS EVALUATION DATA
# ============================================================

print("\n" + "=" * 70)
print("RAGAS EVALUATION DATA")
print("=" * 70)

evaluation_data = {
    "question": [question],
    "answer": [answer],
    "contexts": [
        [
            document.page_content
            for document in retrieved_documents
        ]
    ]
}

print("Evaluation dataset prepared successfully.")
print(f"Questions prepared: {len(evaluation_data['question'])}")
print("Fields: question, answer, contexts")


# ============================================================
# 16. MLflow TRACKING
# ============================================================

print("\n" + "=" * 70)
print("MLFLOW TRACKING")
print("=" * 70)

mlflow.set_experiment(
    "W11D5_Tracked_Evaluated_RAG"
)

with mlflow.start_run() as run:

    mlflow.log_param(
        "chunk_size",
        CHUNK_SIZE
    )

    mlflow.log_param(
        "chunk_overlap",
        CHUNK_OVERLAP
    )

    mlflow.log_param(
        "retrieval_k",
        RETRIEVAL_K
    )

    mlflow.log_param(
        "embedding_model",
        EMBEDDING_MODEL
    )

    mlflow.log_param(
        "llm_model",
        LLM_MODEL
    )

    mlflow.log_param(
        "orchestration",
        "LangGraph"
    )

    mlflow.log_param(
        "agent_framework",
        "CrewAI"
    )

    mlflow.log_param(
        "evaluation_framework",
        "Ragas"
    )

    mlflow.log_metric(
        "retrieved_documents",
        len(retrieved_documents)
    )

    mlflow.log_text(
        question,
        "question.txt"
    )

    mlflow.log_text(
        answer,
        "answer.txt"
    )

    mlflow.log_text(
        str(crew_result),
        "crewai_review.txt"
    )

    mlflow.log_text(
        context,
        "retrieved_context.txt"
    )

    print(f"MLflow run ID: {run.info.run_id}")


# ============================================================
# 17. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("W11D5 PROJECT COMPLETED")
print("=" * 70)

print("\nIntegrated components:")
print("1. Knowledge Base")
print("2. ChromaDB")
print("3. Retriever")
print("4. LangGraph")
print("5. Qwen2.5:3B via Ollama")
print("6. CrewAI")
print("7. Ragas evaluation dataset")
print("8. MLflow tracking")

print("\nRAG pipeline execution completed successfully.")
print("MLflow tracking completed successfully.")
print("=" * 70)