"""
rag_tool.py — Retrieval-Augmented Generation pipeline for uploaded documents.

WHAT RAG SOLVES:
  A 50-page PDF has ~50,000 words. LLM context window = ~4,000-8,000 tokens.
  You cannot paste the whole document. RAG solves this by:
    1. Splitting the doc into small chunks (e.g. 400 words each)
    2. Embedding each chunk (converting text → float vector of 768 numbers)
    3. When you ask a question, embedding the question too
    4. Finding chunks whose vectors are CLOSEST to the question vector
       (similar meaning = close vectors = high cosine similarity)
    5. Sending ONLY the top-k closest chunks to the LLM

THIS FILE IMPLEMENTS RAG FROM SCRATCH (no ChromaDB needed):
  - Text extraction:  pypdf (PDF) + python-docx (Word)
  - Chunking:         manual overlapping window (pure Python)
  - Embeddings:       nomic-embed-text via llm_client (already loaded)
  - Similarity:       cosine similarity with numpy
  - Storage:          in-memory during request (no persistent DB for now)

USED BY: app/mcp/server.py, app/api/rag_routes.py
"""

import os
import math
from typing import List, Tuple

import numpy as np          # for cosine similarity math
from pypdf import PdfReader  # for PDF text extraction
from docx import Document   # for Word .docx text extraction

from app.llm.llm_client import llm_client  # for nomic-embed-text embeddings
from app.core.config import settings


# =============================================================================
# CHUNKING STRATEGY
# =============================================================================
# chunk_size:    number of words per chunk  (larger = more context, fewer chunks)
# chunk_overlap: words repeated between consecutive chunks
#                (prevents losing meaning at chunk boundaries)
#
# Example with chunk_size=10, overlap=3:
#   text: "A B C D E F G H I J K L M N O"
#   chunk 1: "A B C D E F G H I J"
#   chunk 2: "H I J K L M N O P Q"   ← H I J repeated for continuity
CHUNK_SIZE = 400    # words
CHUNK_OVERLAP = 50  # words


class RAGTool:
    """
    Processes uploaded documents for Retrieval-Augmented Generation.

    Pipeline:
      file → extract_text() → chunk_text() → embed_chunks()
           → [user query embedded] → cosine_similarity()
           → top-k chunks → returned to summarizer
    """

    # ─────────────────────────────────────────────────────────────────────────
    # STEP A: TEXT EXTRACTION
    # Different file types need different parsers
    # ─────────────────────────────────────────────────────────────────────────

    def extract_text(self, file_path: str) -> str:
        """
        Extract raw text from a PDF or DOCX file.

        Args:
            file_path: Absolute path to the uploaded file (e.g. "uploads/doc.pdf")

        Returns:
            Extracted plain text string, or error message.

        Detects file type from extension automatically.
        """
        ext = os.path.splitext(file_path)[1].lower()  # ".pdf" or ".docx"
        print(f"[RAG] Extracting text from {ext} file: {file_path}")

        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext in (".docx", ".doc"):
            return self._extract_docx(file_path)
        else:
            return f"ERROR: Unsupported file type '{ext}'. Use PDF or DOCX."

    def _extract_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF using pypdf.

        HOW pypdf WORKS:
          PdfReader loads the PDF → .pages = list of PageObject
          page.extract_text() → returns text on that page as string
          We join all pages with double newlines.
        """
        try:
            reader = PdfReader(file_path)
            pages_text = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(f"[Page {i+1}]\n{text.strip()}")

            full_text = "\n\n".join(pages_text)
            print(f"[RAG] PDF: extracted {len(reader.pages)} pages, {len(full_text)} chars")
            return full_text

        except Exception as e:
            return f"ERROR reading PDF: {str(e)}"

    def _extract_docx(self, file_path: str) -> str:
        """
        Extract text from a Word .docx file using python-docx.

        HOW python-docx WORKS:
          Document() loads the file → .paragraphs = list of Paragraph objects
          paragraph.text → the text content of that paragraph
          We skip empty paragraphs.
        """
        try:
            doc = Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            print(f"[RAG] DOCX: extracted {len(paragraphs)} paragraphs, {len(full_text)} chars")
            return full_text

        except Exception as e:
            return f"ERROR reading DOCX: {str(e)}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP B: CHUNKING
    # Split large text into overlapping windows
    # ─────────────────────────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> List[str]:
        """
        Split text into overlapping word-based chunks.

        WHY WORD-BASED (not character-based):
          Words are the natural unit of meaning.
          400 chars might cut a word in half; 400 words never does.

        WHY OVERLAP:
          If a sentence starts at the end of chunk 1, without overlap
          it would be split — the beginning of the sentence in chunk 1
          and the end in chunk 2. With overlap, both chunks contain
          the full sentence → better retrieval accuracy.

        Args:
            text:       Raw extracted text
            chunk_size: Words per chunk (default: 400)
            overlap:    Words repeated between chunks (default: 50)

        Returns:
            List of text chunk strings.

        Example (chunk_size=5, overlap=2):
            words = [A, B, C, D, E, F, G, H]
            chunks = ["A B C D E", "D E F G H"]
        """
        words = text.split()  # split on whitespace → list of word strings
        total_words = len(words)

        if total_words == 0:
            return []

        # If document is smaller than one chunk, return as-is
        if total_words <= chunk_size:
            print(f"[RAG] Document has {total_words} words — fits in one chunk, no splitting needed")
            return [text]

        chunks = []
        start = 0  # word index to start each chunk

        while start < total_words:
            end = min(start + chunk_size, total_words)  # don't exceed total

            # Slice the word list and re-join into a string
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

            # Move start forward, but subtract overlap so chunks share words
            start += chunk_size - overlap

            # Stop if we've covered all words
            if end == total_words:
                break

        print(f"[RAG] Chunked into {len(chunks)} chunks ({chunk_size} words, {overlap} overlap)")
        return chunks

    # ─────────────────────────────────────────────────────────────────────────
    # STEP C: EMBEDDING
    # Convert text chunks → float vectors using nomic-embed-text
    # ─────────────────────────────────────────────────────────────────────────

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Convert text chunks into embedding vectors using nomic-embed-text.

        WHAT IS AN EMBEDDING:
          Text → float vector of 768 numbers.
          Semantically similar texts → vectors that point in similar directions.

          e.g. embed("king") ≈ embed("queen") (similar direction)
               embed("king") ≠ embed("banana") (different direction)

        HOW WE USE IT:
          Each chunk gets embedded → stored in memory as a list of vectors.
          Later, the query is embedded → compared to all chunk vectors.
          Closest vectors (by cosine similarity) = most relevant chunks.

        Args:
            chunks: List of text strings to embed

        Returns:
            List of float vectors (one per chunk). Each vector = 768 floats.
        """
        print(f"[RAG] Embedding {len(chunks)} chunks with {settings.OLLAMA_EMBED_MODEL}...")

        # llm_client.embed_documents() calls nomic-embed-text for ALL chunks at once
        # Returns: [[0.12, -0.45, 0.89, ...], [0.33, 0.21, -0.67, ...], ...]
        embeddings = llm_client.embed_documents(chunks)

        print(f"[RAG] Embeddings done. Vector size: {len(embeddings[0])} floats per chunk")
        return embeddings

    # ─────────────────────────────────────────────────────────────────────────
    # STEP D: RETRIEVAL
    # Find chunks most relevant to a query using cosine similarity
    # ─────────────────────────────────────────────────────────────────────────

    def find_relevant_chunks(
        self,
        query: str,
        chunks: List[str],
        chunk_embeddings: List[List[float]],
        top_k: int = 4,
    ) -> str:
        """
        Find the top-k chunks most relevant to a query.

        COSINE SIMILARITY:
          Measures the angle between two vectors.
          Score = 1.0 → vectors point same direction (very similar meaning)
          Score = 0.0 → perpendicular (unrelated)
          Score = -1.0 → opposite direction (opposite meaning)

          Formula: cos(θ) = (A · B) / (|A| × |B|)

        STEPS:
          1. Embed the query string (same model as chunks)
          2. Compute cosine similarity between query vector and every chunk vector
          3. Rank chunks by similarity score (highest first)
          4. Return top-k chunks joined as a string

        Args:
            query:            The question/topic to search for
            chunks:           Original text chunks
            chunk_embeddings: Precomputed embeddings for each chunk
            top_k:            Number of chunks to return (default: 4)

        Returns:
            The top-k most relevant chunks joined as one text block.
        """
        print(f"[RAG] Searching for: '{query[:80]}...' in {len(chunks)} chunks")

        # ── Embed the query ───────────────────────────────────────────────
        query_vector = llm_client.embed_query(query)
        query_arr = np.array(query_vector)  # convert to numpy for math

        # ── Compute similarity against all chunk vectors ──────────────────
        scores: List[Tuple[float, int]] = []  # (similarity_score, chunk_index)

        for i, chunk_embedding in enumerate(chunk_embeddings):
            chunk_arr = np.array(chunk_embedding)

            # Cosine similarity formula with numpy
            # np.dot = dot product (A · B)
            # np.linalg.norm = vector magnitude (|A|)
            dot_product = np.dot(query_arr, chunk_arr)
            magnitude = np.linalg.norm(query_arr) * np.linalg.norm(chunk_arr)

            similarity = dot_product / magnitude if magnitude > 0 else 0.0
            scores.append((similarity, i))

        # ── Sort by similarity (highest first) ───────────────────────────
        scores.sort(key=lambda x: x[0], reverse=True)

        # ── Return top-k chunks ───────────────────────────────────────────
        top_chunks = []
        for score, idx in scores[:top_k]:
            print(f"[RAG]   Chunk {idx}: similarity={score:.3f}")
            top_chunks.append(chunks[idx])

        return "\n\n---\n\n".join(top_chunks)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN METHOD: Full Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, file_path: str) -> str:
        """
        Full RAG pipeline: file → relevant text → ready for summarizer.

        Flow:
          1. extract_text()        → raw text from PDF/DOCX
          2. chunk_text()          → list of overlapping chunks
          3. embed_chunks()        → list of vectors
          4. find_relevant_chunks()→ top-4 most relevant chunks
          5. return combined text  → passed to summarizer.summarize()

        For small documents (fits in one chunk):
          Skip embedding step → return text directly (faster)

        Args:
            file_path: Path to the uploaded file (saved in uploads/)

        Returns:
            Relevant text extracted from the document (max ~2000 words),
            ready to be passed directly to summarizer.summarize().
        """
        # ── A: Extract ────────────────────────────────────────────────────
        raw_text = self.extract_text(file_path)
        if raw_text.startswith("ERROR"):
            return raw_text

        word_count = len(raw_text.split())
        print(f"[RAG] Document word count: {word_count}")

        # ── B: If small doc — no RAG needed, return directly ──────────────
        # Small docs fit in the summarizer's context window as-is
        if word_count <= CHUNK_SIZE:
            print("[RAG] Small document — returning full text (no chunking needed)")
            return raw_text

        # ── C: Chunk large document ───────────────────────────────────────
        chunks = self.chunk_text(raw_text)

        # ── D: Embed all chunks ───────────────────────────────────────────
        chunk_embeddings = self.embed_chunks(chunks)

        # ── E: Find most relevant chunks ──────────────────────────────────
        # Query: what do we want to retrieve? → key insights/topics
        # This query is generic so RAG finds the most information-dense parts
        query = (
            "What are the main topics, key insights, important facts, "
            "findings, and conclusions in this document?"
        )
        relevant_text = self.find_relevant_chunks(
            query=query,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            top_k=4,  # return 4 chunks → ~1600 words → fits in summarizer
        )

        print(f"[RAG] Retrieved {len(relevant_text)} chars of relevant content")
        return relevant_text


# ── Singleton ─────────────────────────────────────────────────────────────────
rag_tool = RAGTool()
