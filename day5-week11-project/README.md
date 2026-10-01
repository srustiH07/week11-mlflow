# W11D5 - Tracked and Evaluated RAG Pipeline

## Objective

The objective of W11D5 was to integrate the Week 11 RAG components into a single tracked pipeline using the approved AI/ML stack.

## Architecture

Knowledge Base
→ Text Chunking
→ Ollama Embeddings
→ ChromaDB
→ Retriever
→ LangGraph
→ Qwen2.5:3B via Ollama
→ RAG Answer
→ CrewAI Review
→ Ragas Evaluation Dataset
→ MLflow Tracking

## Technologies Used

- Python
- LangChain
- ChromaDB
- Ollama
- Qwen2.5:3B
- Nomic Embed Text
- LangGraph
- CrewAI
- Ragas
- MLflow

## Implementation

The pipeline loads the knowledge base created during W11D4 and divides it into smaller document chunks.

Ollama embeddings are generated using `nomic-embed-text`, and the chunks are stored in ChromaDB.

A similarity retriever retrieves the top 3 relevant documents for a user question.

LangGraph manages the retrieval and answer-generation workflow.

Qwen2.5:3B running locally through Ollama generates the final RAG answer.

CrewAI is used as a review agent to review the generated answer.

Ragas evaluation data containing the question, generated answer, and retrieved contexts is prepared for evaluation.

MLflow tracks the pipeline configuration, retrieved document count, question, answer, CrewAI review, and retrieved context.

## Test Query

What is Retrieval-Augmented Generation?

## Result

The RAG pipeline successfully:

1. Loaded the knowledge base.
2. Created document chunks.
3. Created Ollama embeddings.
4. Created the ChromaDB vector store.
5. Retrieved relevant documents.
6. Executed the LangGraph workflow.
7. Generated an answer using Qwen2.5:3B.
8. Reviewed the answer using CrewAI.
9. Prepared Ragas evaluation data.
10. Logged the pipeline information and outputs using MLflow.

## Configuration

- Chunk size: 700
- Chunk overlap: 100
- Retrieval k: 3
- Embedding model: nomic-embed-text
- LLM: qwen2.5:3b
- Orchestration: LangGraph
- Agent framework: CrewAI
- Evaluation framework: Ragas

## Conclusion

W11D5 integrates the major Week 11 components into a single tracked RAG workflow. The implementation demonstrates retrieval, generation, agent-based review, evaluation-data preparation, and experiment tracking using the approved AI/ML stack.