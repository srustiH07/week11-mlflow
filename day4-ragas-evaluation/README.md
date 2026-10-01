# W11D4 - RAG Evaluation with Ragas

## Objective

Build a Retrieval-Augmented Generation (RAG) pipeline using LangChain,
ChromaDB, Ollama, and prepare the system for evaluation using Ragas.

## Technologies

- Python
- LangChain
- ChromaDB
- Ollama
- Qwen2.5:3B
- nomic-embed-text
- Ragas

## RAG Pipeline

The implemented pipeline follows this workflow:

Knowledge Base
→ Document Loading
→ Text Chunking
→ Embeddings
→ ChromaDB
→ Retriever
→ Retrieved Context
→ Qwen2.5:3B
→ Generated Answer

## Implemented Components

1. Created an AI/ML knowledge base.
2. Loaded the knowledge base using LangChain.
3. Split the document into smaller chunks.
4. Generated embeddings using `nomic-embed-text`.
5. Stored document embeddings in ChromaDB.
6. Created a similarity-based retriever.
7. Configured retrieval with `k=3`.
8. Connected Qwen2.5:3B through Ollama.
9. Tested the complete RAG pipeline.
10. Prepared 10 evaluation questions.

## Baseline Configuration

- Chunk size: 700
- Chunk overlap: 100
- Retrieval k: 3
- Embedding model: `nomic-embed-text`
- Language model: `qwen2.5:3b`

## RAG Test

The RAG pipeline was tested using the question:

> What is Retrieval-Augmented Generation?

The system successfully retrieved relevant document chunks and generated an answer using Qwen2.5:3B.

## Evaluation Metrics

The planned Ragas evaluation uses:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

## Optimization

The planned optimization step will identify the lowest-scoring evaluation metric and modify the RAG configuration, such as:

- Chunk size
- Retrieval k

The system can then be re-evaluated to compare the baseline and optimized configurations.

## Current Checkpoint

### Completed

- Knowledge base
- Document loading
- Chunking
- Embeddings
- ChromaDB
- Retriever
- Qwen2.5:3B RAG test
- 10 evaluation questions

### Pending

- Generate all 10 Q&A responses
- Ragas evaluation
- Metric results
- Lowest-metric optimization
- Before/after comparison

This checkpoint records the RAG pipeline work completed so far without marking the pending evaluation steps as completed.