"""
main.py — FastAPI application entry point.

THIS FILE'S ONLY JOB:
  1. Create the FastAPI app instance
  2. Register all routers (each router = one file of related endpoints)
  3. Add middleware (CORS, auth, logging - later)
  4. Handle startup/shutdown tasks

WHAT IT DOES NOT DO:
  - Business logic (that's in tools/, agent/)
  - LLM calls (that's in llm_client.py)
  - Data validation (that's in schemas/models.py)

RUN WITH:
  uvicorn main:app --reload
       ^     ^  ^
       |     |  app = the FastAPI() instance in this file
       |     main = this file (main.py)
       uvicorn = ASGI server that runs FastAPI apps

ADDING NEW STEPS (just add 2 lines per step):
  from app.api.tools_routes import router as tools_router
  app.include_router(tools_router, prefix="/api")
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

# ── Import routers (one per step/feature area) ────────────────────────────────
from app.api.llm_routes import router as llm_router
from app.api.tools_routes import router as tools_router   # Step 2
from app.api.rag_routes import router as rag_router        # Step 3
from app.api.generate_routes import router as generate_router  # Step 4
from app.api.media_routes import router as media_router    # Media Tool


# =============================================================================
# LIFESPAN — Startup and Shutdown Logic
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs at STARTUP.
    Code after  `yield` runs at SHUTDOWN.

    asynccontextmanager = makes this work as an async context manager
    This replaces the old @app.on_event("startup") pattern.
    """
    # ── Startup ───────────────────────────────────────────────────────────
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)    # ensure uploads/ exists
    os.makedirs(settings.CHROMA_DIR, exist_ok=True)    # ensure chroma_db/ exists
    os.makedirs(settings.MEDIA_OUTPUT_DIR, exist_ok=True)  # ensure media_output/ exists
    print(f"[START] {settings.APP_TITLE} v{settings.APP_VERSION}")
    print(f"[LLM]   model       : {settings.OLLAMA_MODEL}")
    print(f"[EMB]   embed model : {settings.OLLAMA_EMBED_MODEL}")
    print(f"[DOCS]  swagger ui  : http://localhost:8000/docs")
    print(f"[DOCS]  redoc       : http://localhost:8000/redoc")

    yield   # ← app runs here (between startup and shutdown)

    # ── Shutdown ──────────────────────────────────────────────────────────
    print("[STOP] Shutting down gracefully...")


# =============================================================================
# APP INSTANCE
# =============================================================================
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=(
        "AI pipeline: Article input -> Agent -> LinkedIn / Instagram post.\n\n"
        "**Build Steps:**\n"
        "- Step 1: LLM (Ollama llama3.2:3b) — `/api/llm/*`\n"
        "- Step 2: MCP Tools — `/api/tools/*` — Scraper, Summarizer, Search, PostGen\n"
        "- Step 3: RAG Pipeline — `/api/rag/*`\n"
        "- Step 4: Agent — `/api/generate/*`\n"
        "- Media Tool — `/api/media/*` — NVIDIA NIM image generation (swap to DALL-E later)"
    ),
    lifespan=lifespan,
)


# =============================================================================
# MIDDLEWARE — Applied to every request/response
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    # allow_origins: which domains can call this API
    # "*" = any domain (fine for development, restrict in production)
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],    # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],    # Content-Type, Authorization, etc.
)


# =============================================================================
# REGISTER ROUTERS
# =============================================================================
# prefix="/api" is added to ALL routes registered here
# So: llm_router has prefix="/llm" → full path = /api/llm/...
app.include_router(llm_router, prefix="/api")
app.include_router(tools_router, prefix="/api")               # Step 2
app.include_router(rag_router, prefix="/api")                 # Step 3
app.include_router(generate_router, prefix="/api")            # Step 4
app.include_router(media_router, prefix="/api")               # Media Tool


# =============================================================================
# ROOT ENDPOINT
# =============================================================================
@app.get("/", tags=["Root"])
def root():
    """
    Landing page — confirms the app is running.
    Visit http://localhost:8000/ in your browser.
    """
    return {
        "app": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "status": "running",
        "swagger_docs": "http://localhost:8000/docs",
        "step1_health": "http://localhost:8000/api/llm/health",
    }
