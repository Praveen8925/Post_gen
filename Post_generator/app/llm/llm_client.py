"""
llm_client.py — The ONLY file that talks to Ollama directly.

DESIGN DECISIONS:
  1. Class wraps ChatOllama so switching to OpenAI later = change THIS file only
  2. Singleton at bottom = import once, reuse everywhere (no repeated connections)
  3. chat()   = wait for full response (good for short outputs)
  4. stream() = yield chunks as they arrive (good for long posts/summaries)
  5. Embeddings stored here too → RAG pipeline imports llm_client.embeddings

HOW OTHER FILES USE IT:
    from app.llm.llm_client import llm_client

    result = llm_client.chat("You are X.", "Do Y.")
    # returns: "Here is Y..."
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings   # ← gets model name, URL, temperature


class LLMClient:
    """
    Wrapper around Ollama LLM + Embeddings.

    Why a class?
      - Groups related methods together
      - Holds the llm and embeddings objects as instance variables
      - Easy to swap: replace ChatOllama with ChatOpenAI in __init__ only
    """

    def __init__(self):
        # ── Chat model (llama3.2:3b) ──────────────────────────────────────
        # ChatOllama = LangChain wrapper for Ollama chat completions
        # base_url: where Ollama is running (local = http://localhost:11434)
        # model: which model to use (from .env)
        # temperature: 0.0 = deterministic, 1.0 = very creative
        self.llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=settings.OLLAMA_TEMPERATURE,
        )

        # ── Embeddings model (nomic-embed-text) ───────────────────────────
        # Used by RAG pipeline (Step 3) to convert text → float vectors
        # Kept here so embeddings always uses same Ollama instance as chat
        self.embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBED_MODEL,
        )

        print(f"[LLM] Loaded: {settings.OLLAMA_MODEL} @ {settings.OLLAMA_BASE_URL}")

    # ─────────────────────────────────────────────────────────────────────────
    def chat(self, system_prompt: str, user_message: str) -> str:
        """
        Single-turn LLM call. Waits for the full response before returning.

        Use for: summarization, post generation, classification
        Avoid for: very long outputs (user waits too long)

        Args:
            system_prompt: Persona/instructions for the LLM
                           e.g. "You are an expert LinkedIn content writer."
            user_message:  The actual content or question
                           e.g. "Write a post about: [summary]"

        Returns:
            Full LLM response as a plain string.

        Example:
            response = llm_client.chat(
                system_prompt="You are a helpful assistant.",
                user_message="What is RAG in 2 sentences?"
            )
            # "RAG stands for Retrieval-Augmented Generation..."
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = self.llm.invoke(messages)
        return response.content   # .content extracts the string from the message object

    # ─────────────────────────────────────────────────────────────────────────
    def stream(self, system_prompt: str, user_message: str):
        """
        Streaming LLM call. Yields text chunks as they are generated.

        Use for: long post generation where you want to show progress
        How: FastAPI wraps this in StreamingResponse → browser sees live text

        Usage:
            for chunk in llm_client.stream("You are X.", "Write Y."):
                print(chunk, end="", flush=True)
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        for chunk in self.llm.stream(messages):
            if chunk.content:        # skip empty chunks
                yield chunk.content

    # ─────────────────────────────────────────────────────────────────────────
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text strings into embedding vectors.
        Used by RAG pipeline when indexing document chunks into ChromaDB.

        Returns: List of float vectors, one per input string
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """
        Convert a single query string into an embedding vector.
        Used by RAG pipeline when searching for relevant chunks.

        Returns: Single float vector
        """
        return self.embeddings.embed_query(query)


# =============================================================================
# SINGLETON — This runs ONCE when the file is first imported.
# All subsequent imports return this same object (Python caches modules).
#
# Result: Only one Ollama connection is opened, ever.
# Usage everywhere: from app.llm.llm_client import llm_client
# =============================================================================
llm_client = LLMClient()
