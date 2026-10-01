# W11D4 - RAG Evaluation with Ragas
# RAG pipeline checkpoint for LangChain, ChromaDB, Ollama and Ragas.
#
# Completed in this checkpoint:
# - Knowledge base creation
# - Document loading
# - Text chunking
# - Embedding generation
# - ChromaDB vector store
# - Similarity retrieval
# - Ollama LLM connection
# - End-to-end RAG test
# - 10 evaluation questions
#
# Remaining:
# - Generate all 10 Q&A responses
# - Ragas evaluation
# - Metric comparison
# - RAG optimization
# - Before/after evaluation

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ---------------------------------------------------------
# 1. Load the knowledge base
# ---------------------------------------------------------

loader = TextLoader(
    "knowledge_base.txt",
    encoding="utf-8"
)

documents = loader.load()

print("Documents loaded:", len(documents))


# ---------------------------------------------------------
# 2. Split documents into chunks
# ---------------------------------------------------------

chunk_size = 700
chunk_overlap = 100

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))
print("Chunk size:", chunk_size)
print("Chunk overlap:", chunk_overlap)


# ---------------------------------------------------------
# 3. Create embeddings
# ---------------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

print("Embedding model configured: nomic-embed-text")


# ---------------------------------------------------------
# 4. Store documents in ChromaDB
# ---------------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="w11d4_ragas_baseline"
)

print("ChromaDB vector store created successfully.")
print("Stored chunks:", len(chunks))


# ---------------------------------------------------------
# 5. Create retriever
# ---------------------------------------------------------

retrieval_k = 3

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": retrieval_k
    }
)

print("Retriever created successfully.")
print("Retrieval k:", retrieval_k)


# ---------------------------------------------------------
# 6. Connect to Qwen2.5:3B through Ollama
# ---------------------------------------------------------

llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://127.0.0.1:11434",
    temperature=0
)

print("Qwen2.5:3B connected successfully.")


# ---------------------------------------------------------
# 7. End-to-end RAG test
# ---------------------------------------------------------

question = "What is Retrieval-Augmented Generation?"

retrieved_docs = retriever.invoke(question)

context = "\n\n".join(
    document.page_content
    for document in retrieved_docs
)

prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)


print("\n" + "=" * 70)
print("RAG PIPELINE TEST")
print("=" * 70)

print("\nQUESTION:")
print(question)

print("\nRETRIEVED DOCUMENTS:")
print(len(retrieved_docs))

print("\nANSWER:")
print(response.content)


# ---------------------------------------------------------
# 8. Prepare 10 evaluation questions
# ---------------------------------------------------------

questions = [
    "What is Artificial Intelligence?",
    "What is the difference between supervised and unsupervised learning?",
    "What is Deep Learning?",
    "What is Natural Language Processing?",
    "What is a Large Language Model?",
    "What is Retrieval-Augmented Generation?",
    "What is the role of ChromaDB in a RAG system?",
    "What is chunking and why is chunk size important?",
    "What does retrieval k mean in a RAG system?",
    "What is Ragas and what metrics does it provide?"
]


print("\n" + "=" * 70)
print("10 EVALUATION QUESTIONS")
print("=" * 70)

for number, question in enumerate(questions, start=1):
    print(f"{number}. {question}")


# ---------------------------------------------------------
# 9. Current checkpoint
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("W11D4 CHECKPOINT")
print("=" * 70)

print("Knowledge base: COMPLETED")
print("Document loading: COMPLETED")
print("Chunking: COMPLETED")
print("Embeddings: COMPLETED")
print("ChromaDB: COMPLETED")
print("Retriever: COMPLETED")
print("Qwen2.5:3B RAG test: COMPLETED")
print("10-question dataset: PREPARED")

print("\nRagas evaluation: PENDING")
print("RAG optimization: PENDING")
print("Before/after comparison: PENDING")