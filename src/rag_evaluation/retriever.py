
import os
import re
import glob

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

DATA_DIR = "data"
DB_DIR = "chroma_store_hf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

CANDIDATE_K = 15
FINAL_K = 5

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# 1. LOAD TRANSCRIPTS
# ============================================================

def load_transcripts():

    docs = []

    transcript_files = glob.glob(
        os.path.join(DATA_DIR, "*.vtt")
    )

    if not transcript_files:
        raise FileNotFoundError(
            f"No VTT transcript files found inside '{DATA_DIR}'"
        )

    for path in transcript_files:

        lines = []

        with open(
            path, "r", encoding="utf-8", errors="replace"
        ) as file:

            for line in file:

                line = line.strip()

                if (
                    not line
                    or line == "WEBVTT"
                    or "-->" in line
                ):
                    continue

                lines.append(line)

        text = " ".join(lines)

        match = re.search(
            r"Session[ _]*(\d+)",
            path,
            re.IGNORECASE
        )

        if not match:
            print(f"Skipping file: {path}")
            continue

        session = match.group(1)

        docs.append(
            Document(
                page_content=text,
                metadata={"session": session}
            )
        )

        print(f"Loaded Session {session}")

    if not docs:
        raise ValueError(
            "No valid transcripts could be loaded."
        )

    print(f"\nTotal transcripts loaded: {len(docs)}")

    return docs


# ============================================================
# 2. BUILD / LOAD CHROMADB
# ============================================================

def load_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(DB_DIR):

        print("\nLoading existing ChromaDB...")

        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

    print("\nLoading transcripts...")

    docs = load_transcripts()

    print("\nSplitting transcripts into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(docs)

    print(f"Total chunks created: {len(chunks)}")

    print("\nGenerating embeddings and creating ChromaDB...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("ChromaDB created successfully!")

    return vector_store


# ============================================================
# 3. CROSS-ENCODER RERANKER
# ============================================================

class RerankingRetriever:

    def __init__(
        self,
        vector_store,
        candidate_k=CANDIDATE_K,
        final_k=FINAL_K
    ):

        self.vector_store = vector_store
        self.candidate_k = candidate_k
        self.final_k = final_k

        print("\nLoading cross-encoder reranker...")

        self.reranker = CrossEncoder(
            RERANKER_MODEL,
            device="cpu"
        )

        print("Reranker loaded successfully!")

    def invoke(self, query):

        # Step 1: Retrieve candidate chunks from ChromaDB
        candidates = self.vector_store.similarity_search(
            query,
            k=self.candidate_k
        )

        print(
            f"\nInitial retrieval: {len(candidates)} chunks"
        )

        if not candidates:
            return []

        # Step 2: Create query-document pairs
        pairs = [
            (query, document.page_content)
            for document in candidates
        ]

        # Step 3: Calculate cross-encoder relevance scores
        scores = self.reranker.predict(pairs)

        # Step 4: Combine documents with their scores
        scored_documents = list(
            zip(candidates, scores)
        )

        # Step 5: Sort by descending relevance score
        scored_documents.sort(
            key=lambda item: float(item[1]),
            reverse=True
        )

        # Step 6: Return final top K documents
        reranked_documents = [
            document
            for document, score in scored_documents[:self.final_k]
        ]

        print(
            f"After reranking: {len(reranked_documents)} chunks"
        )

        return reranked_documents


# ============================================================
# 4. BUILD RETRIEVER
# ============================================================

def build_retriever():

    vector_store = load_store()

    retriever = RerankingRetriever(
        vector_store=vector_store,
        candidate_k=CANDIDATE_K,
        final_k=FINAL_K
    )

    return retriever


# ============================================================
# 5. TEST RETRIEVER
# ============================================================

if __name__ == "__main__":

    print("Building RAG retriever...")

    retriever = build_retriever()

    query = "what is regression testing?"

    print(f"\nQuery: {query}")
    print("\nTop 5 reranked chunks:\n")

    results = retriever.invoke(query)

    for index, result in enumerate(results, start=1):

        print(f"Result {index}")

        print(
            f"Session: {result.metadata.get('session', 'Unknown')}"
        )

        print(
            f"Content: {result.page_content[:300]}..."
        )

        print("-" * 70)
