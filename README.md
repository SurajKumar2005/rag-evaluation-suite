# RAG Retrieval Evaluation & Reranking System

An experimental Retrieval-Augmented Generation (RAG) retrieval evaluation system that combines semantic search, vector databases, cross-encoder reranking, and LLM-as-a-judge evaluation to measure and improve retrieval quality.

## Overview

Retrieval quality is a critical component of a RAG application. Retrieving irrelevant chunks can introduce noise, while missing relevant information can lead to incomplete answers.

This project evaluates a document retrieval pipeline using a collection of technical course transcripts. It compares different chunking and retrieval configurations, introduces cross-encoder reranking, and evaluates retrieval quality using DeepEval.

The objective is to identify a retrieval configuration that provides a strong balance between contextual precision and contextual recall.

## Architecture

```text
Course Transcripts (VTT)
          |
          v
    Text Extraction
          |
          v
  Recursive Character
     Text Splitting
          |
          v
 Hugging Face Embeddings
  all-MiniLM-L6-v2
          |
          v
      ChromaDB
          |
          v
  Initial Semantic Search
       Top 15
          |
          v
  Cross-Encoder Reranking
  ms-marco-MiniLM-L-6-v2
          |
          v
    Final Top 3 Chunks
          |
          v
       DeepEval
          |
          v
 Contextual Recall &
 Contextual Precision
          |
          v
     JSON Reports
```

## Technology Stack

| Category | Technology |
|---|---|
| Programming language | Python |
| Embeddings | Sentence Transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Reranking | Sentence Transformers CrossEncoder |
| Reranker model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Evaluation framework | DeepEval |
| LLM judge | Groq API with Qwen model |
| Data handling | Python, JSON |
| Environment and dependency management | uv |

## Project Structure

```text
rag-evaluation/
│
├── data/
│   └── Course transcripts (.vtt)
│
├── evals/
│   └── eval_retriever.py
│
├── goldens/
│   └── retriever_goldens.json
│
├── reports/
│   ├── baseline_750_no_reranker.json
│   ├── baseline_1000_no_reranker.json
│   ├── baseline_1000_with_reranker.json
│   ├── experiment_A_k15_final4.json
│   ├── experiment_B_k15_final3.json
│   └── experiment_C_k10_final3.json
│
├── src/
│   └── rag_evaluation/
│       ├── __init__.py
│       ├── groq_judge.py
│       └── retriever.py
│
├── test_groq_connection.py
├── test_groq_schema.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

Local databases, API credentials, virtual environments and raw transcript files are excluded from version control.

## Retrieval Pipeline

### 1. Document Processing

- Load VTT transcript files.
- Extract transcript text.
- Split documents using RecursiveCharacterTextSplitter.
- Generate overlapping text chunks.

Selected configuration:

- Chunk size: 1,000 characters
- Chunk overlap: 150 characters

### 2. Embedding Generation

The `sentence-transformers/all-MiniLM-L6-v2` model generates dense vector representations of document chunks.

Embeddings are stored in ChromaDB for similarity-based retrieval.

### 3. Initial Retrieval

ChromaDB performs semantic similarity search and retrieves the top 15 candidate chunks for each query.

### 4. Cross-Encoder Reranking

A cross-encoder evaluates each query-document pair and produces relevance scores.

The candidate chunks are sorted according to their reranker scores, and the top 3 chunks are selected as the final retrieval context.

Selected reranker:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

### 5. Retrieval Evaluation

DeepEval evaluates the retrieved context against a golden dataset containing 15 test cases.

Metrics:

- **Contextual Recall:** Measures how well the retrieved context covers the information required by the expected answer.
- **Contextual Precision:** Measures the relevance and ranking quality of the retrieved context.

The evaluation threshold is 0.7.

A test case passes when both configured metric thresholds are satisfied.

## Experiments and Results

Five retrieval configurations were evaluated.

| Experiment | Chunk Size | Candidate K | Final K | Recall | Precision | Overall Pass Rate |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 1 | 750 | 5 | 5 | 0.8867 | 0.7885 | 73.33% |
| Baseline 2 | 1,000 | 5 | 5 | 0.9889 | 0.7881 | 73.33% |
| Reranker baseline | 1,000 | 15 | 5 | 0.9833 | 0.8412 | 73.33% |
| Experiment A | 1,000 | 15 | 4 | 0.9833 | 0.8611 | 73.33% |
| Experiment B | 1,000 | 15 | 3 | 0.9722 | 0.9167 | **86.67%** |
| Experiment C | 1,000 | 10 | 3 | 0.9056 | 0.8500 | 73.33% |

*Results are from the current 15-question golden evaluation dataset.*

### Selected Configuration: Experiment B

The selected configuration uses:

| Parameter | Selected value |
|---|---|
| Chunk size | 1,000 |
| Chunk overlap | 150 |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| Candidate K | 15 |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| Final K | 3 |

Observed evaluation results:

- Contextual Recall: **0.9722**
- Contextual Precision: **0.9167**
- Overall pass rate: **86.67%**
- Test cases passed: **13/15**
- API error cases: **0**

### Key Findings

1. Increasing chunk size from 750 to 1,000 improved contextual recall in the baseline experiments.
2. Introducing cross-encoder reranking improved contextual precision.
3. Reducing final K from 5 to 3 improved precision while retaining strong recall.
4. Reducing candidate K from 15 to 10 negatively affected both recall and precision in the tested configuration.
5. Experiment B achieved the highest observed precision and overall pass rate among the tested configurations.

## Installation and Setup

### Prerequisites

- Python 3.11
- Git
- uv package manager

### Clone the repository

```bash
git clone https://github.com/SurajKumar2005/rag-evaluation-suite.git
cd rag-evaluation-suite
```

### Install dependencies

```bash
uv sync
```

### Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.8-27b
```

Never commit your actual `.env` file or API credentials.

### Add source transcripts

Place your VTT transcript files inside the `data/` directory.

The local ChromaDB database is generated or loaded by the retriever.

### Run the retriever

```bash
uv run python src/rag_evaluation/retriever.py
```

### Run retrieval evaluation

```bash
uv run python evals/eval_retriever.py
```

Evaluation results are saved as JSON files inside `reports/`.

## Limitations

- The golden evaluation dataset currently contains 15 test cases.
- Evaluation results depend on the quality and coverage of the golden dataset.
- The current implementation focuses on retrieval evaluation rather than a complete user-facing RAG generation application.
- Cross-encoder reranking adds inference latency compared with vector search alone.
- Local embedding and reranking models run on CPU and may have higher latency on lower-specification hardware.
- Results should be validated with a larger and more diverse evaluation dataset before production use.

## Future Improvements

- Expand the golden dataset with more diverse questions.
- Add generation evaluation using faithfulness and answer relevancy metrics.
- Build an interactive PDF question-answering application.
- Display source documents and supporting chunks with generated answers.
- Evaluate hybrid retrieval combining dense and sparse search.
- Investigate query rewriting and additional reranking models.
- Add experiment tracking and reproducible evaluation configurations.

## Author

**Suraj Kumar**

B.Tech Computer Science and Engineering (Hons.)

Interests: Generative AI, Retrieval-Augmented Generation, Machine Learning, NLP and AI Engineering.

GitHub: https://github.com/SurajKumar2005

---

*This project was developed as a hands-on exploration of RAG retrieval quality, evaluation methodology and cross-encoder reranking.*
