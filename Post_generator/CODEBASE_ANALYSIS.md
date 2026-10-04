# Codebase Analysis Report: Content Generator AI

## 1. Project Overview
**Content Generator AI** is a sophisticated AI-powered pipeline designed to automate the creation of high-quality social media content. It transforms diverse raw inputs—such as web URLs, plain text, specific topics, or uploaded documents (PDF/DOCX)—into tailored posts for **LinkedIn** and **Instagram**, complete with structured summaries and AI-generated visual assets.

## 2. System Architecture
The project follows a modular, decoupled architecture separating the user interface, the API layer, and the AI orchestration logic.

### 2.1 High-Level Component Stack
- **Frontend**: Streamlit (`ui.py`) - A data-driven UI for configuration and result visualization.
- **Backend**: FastAPI (`main.py`) - A high-performance ASGI framework providing a set of RESTful endpoints.
- **AI Orchestration**: 
    - **LLM**: Ollama (specifically `llama3.2:3b`) serving as the primary reasoning engine.
    - **Agent Framework**: A custom agent implementation (`app/agent/`) capable of both rule-based pipeline execution and autonomous tool selection.
    - **RAG Engine**: LlamaIndex + ChromaDB for vector storage and retrieval of document-based knowledge.
    - **Protocol**: Integration of the Model Context Protocol (MCP) for standardized tool communication.
- **Infrastructure**:
    - **Vector Database**: ChromaDB for persistent embedding storage.
    - **Image Generation**: NVIDIA NIM (integrated via `media_tool.py`).

### 2.2 Directory Structure
- `app/`
    - `api/`: Route handlers for LLM, Tools, RAG, Generation, and Media.
    - `core/`: Application settings and environment configuration.
    - `llm/`: The `llm_client` wrapper for interacting with Ollama.
    - `schemas/`: Pydantic models for strict data validation and API contracts.
    - `agent/`: Logic for agentic orchestration and tool mapping.
    - `tools/`: Atomic utility modules (Scraper, Summarizer, Search, RAG, PostGen, Media).
- `main.py`: FastAPI entry point and router registration.
- `ui.py`: Streamlit application logic and custom CSS styling.
- `chroma_db/`: Local storage for the vector database.
- `uploads/` & `media_output/`: Storage for user-uploaded files and AI-generated images.

## 3. Functional Pipeline
The system operates as a multi-stage pipeline:

### Stage 1: Input Acquisition
The system supports four primary input modalities:
1. **URL**: Uses `web_scraper.py` (Playwright/BeautifulSoup) to extract article content.
2. **Topic**: Uses `web_search.py` (DuckDuckGo) to gather current information.
3. **Document**: Uses `rag_tool.py` to parse PDF/DOCX, create embeddings, and retrieve relevant context.
4. **Text**: Direct processing of user-provided paragraphs.

### Stage 2: Analysis & Summarization
The raw content is processed by the `summarizer.py` tool, which produces a **Structured Summary** containing:
- **Topic**: The core subject.
- **Summary**: A concise overview.
- **Key Points**: A list of critical takeaways.
- **Hook**: A compelling opening line.
- **Tone & Audience**: Analysis of the target demographic and delivery style.

### Stage 3: Content Generation
The `post_generator.py` takes the structured summary and applies platform-specific heuristics:
- **LinkedIn**: Professional, thought-leader tone, uses unicode bolding for emphasis.
- **Instagram**: Casual, emoji-rich, with a high density of hashtags.

### Stage 4: Visual Enhancement
The `media_tool.py` generates visual assets based on the content analysis:
- **Modes**: AI Smart (Automatic), Manual, or Disabled.
- **Types**: Infographics, Theme Images, Diagrams, and Influencer shots.
- **Future Proofing**: Placeholders exist for "Phase 2" (Video/GIFs via HeyGen).

## 4. Key Technical Features
- **Dual Execution Modes**: 
    - *Pipeline Mode*: A deterministic, fast sequence of tool calls.
    - *Agent Mode*: An autonomous mode where the LLM decides which tools to call and in what order.
- **RAG Implementation**: Efficient handling of large documents by chunking and indexing them into ChromaDB, ensuring the LLM doesn't suffer from context window limitations.
- **Standardized Schemas**: Use of Pydantic ensures that data flowing between the scraper $\to$ summarizer $\to$ post generator is consistent and validated.
- **User-Centric UI**: The Streamlit interface provides real-time "Backend Processing Logs," giving users transparency into the agent's thought process.

## 5. Summary of Technologies Used
| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.12 |
| **API Framework** | FastAPI |
| **Frontend** | Streamlit |
| **LLM Engine** | Ollama (llama3.2:3b) |
| **RAG / Embeddings** | LlamaIndex, ChromaDB |
| **Web Scraping** | Playwright, BeautifulSoup4 |
| **Search** | DuckDuckGo Search |
| **Doc Parsing** | PyPDF, Python-docx |
| **Image Generation** | NVIDIA NIM |
| **Protocol** | MCP (Model Context Protocol) |
