"""
rag_routes.py — HTTP endpoints for document upload + RAG pipeline (Step 3).

NEW CONCEPT: FILE UPLOAD WITH FastAPI
  JSON endpoints: client sends Content-Type: application/json
  File uploads:   client sends Content-Type: multipart/form-data

  In FastAPI, file uploads use:
    - UploadFile  → the uploaded file object (has .filename, .read(), etc.)
    - Form(...)   → text fields sent alongside the file
    - File(...)   → declares a required file upload

  Client sends:
    POST /api/rag/upload
    Content-Type: multipart/form-data
    Body:
      file=<binary file data>
      platform=linkedin

  FastAPI auto-parses this into UploadFile + plain string.

ENDPOINTS:
  POST /api/rag/extract  → upload file, get extracted text only (for testing)
  POST /api/rag/upload   → upload file, run full pipeline → get posts
"""

import os
import uuid
import shutil

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Literal, Optional

from app.tools.rag_tool import rag_tool
from app.tools.summarizer import summarizer
from app.tools.post_generator import post_generator
from app.core.config import settings


router = APIRouter(prefix="/rag", tags=["Step 3 - RAG"])


# Allowed file extensions and their MIME types
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}


def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save an uploaded file to the uploads/ directory.

    WHY SAVE TO DISK:
      UploadFile is a stream — you can only read it once.
      pypdf and python-docx need a file PATH (not a stream).
      So we save it first, then pass the path to the parsers.

    UNIQUE FILENAME:
      uuid4() generates a random unique ID to prevent filename conflicts.
      e.g. "doc.pdf" → "a3f2b1c4-...pdf" (two users uploading same filename won't collide)

    Args:
        upload_file: FastAPI UploadFile object

    Returns:
        Absolute path to the saved file (e.g. "uploads/abc123.pdf")
    """
    # Validate file extension
    ext = os.path.splitext(upload_file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Generate unique filename → prevents collisions
    unique_name = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_name)

    # Write uploaded content to disk
    # shutil.copyfileobj(src, dst) copies file-like object to another file-like object
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    print(f"[RAG ROUTE] Saved upload: {unique_name} ({upload_file.filename})")
    return file_path


def cleanup_file(file_path: str):
    """Delete the uploaded file after processing to save disk space."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[RAG ROUTE] Cleaned up: {file_path}")
    except Exception as e:
        print(f"[RAG ROUTE] Cleanup warning: {e}")


# =============================================================================
# ENDPOINT 1: Extract Only (for debugging)
# POST /api/rag/extract
# =============================================================================
@router.post("/extract", summary="Test: Upload a file and see extracted text")
async def rag_extract(
    file: UploadFile = File(...),   # File(...) = required file upload
    # The ... means "required" — same as Pydantic's required fields
):
    """
    Uploads a PDF or DOCX and returns the raw extracted text.
    Use this to verify text extraction before running the full pipeline.

    How to test in /docs:
      1. Click "Try it out"
      2. Click "Choose File"
      3. Select a PDF or DOCX
      4. Click "Execute"

    How to test with curl:
        curl -X POST http://localhost:8000/api/rag/extract
             -F "file=@/path/to/document.pdf"
    """
    file_path = save_upload_file(file)

    try:
        extracted_text = rag_tool.extract_text(file_path)
        word_count = len(extracted_text.split())
        return {
            "original_filename": file.filename,
            "word_count": word_count,
            "char_count": len(extracted_text),
            "extracted_text_preview": extracted_text[:1000],  # first 1000 chars
            "full_text": extracted_text,
        }
    finally:
        cleanup_file(file_path)  # always clean up, even if error


# =============================================================================
# ENDPOINT 2: Full RAG Pipeline
# POST /api/rag/upload
# =============================================================================
@router.post("/upload", summary="Upload a document and generate LinkedIn/Instagram posts")
async def rag_upload(
    file: UploadFile = File(...),           # required: the document
    platform: str = Form(default="both"),   # optional form field (default: both)
    # Form() = text fields sent alongside the file in multipart/form-data
):
    """
    Full RAG pipeline:
      Upload PDF/DOCX → extract → chunk → embed → retrieve → summarize → generate post

    For small documents (< 400 words): skips chunking/embedding, processes directly
    For large documents: uses cosine similarity to find the most relevant sections

    platform options: "linkedin" | "instagram" | "both"

    How to test in /docs:
      1. Click "Try it out"
      2. Upload a PDF or DOCX file
      3. Set platform (optional, default: "both")
      4. Click "Execute"

    How to test with curl:
        curl -X POST http://localhost:8000/api/rag/upload
             -F "file=@/path/to/document.pdf"
             -F "platform=linkedin"
    """
    # Validate platform value
    valid_platforms = {"linkedin", "instagram", "both"}
    if platform not in valid_platforms:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid platform '{platform}'. Choose: {valid_platforms}"
        )

    file_path = save_upload_file(file)

    try:
        # ── Step 1: RAG — extract relevant text from document ─────────────
        print(f"[RAG ROUTE] Processing: {file.filename}")
        relevant_text = rag_tool.process(file_path)

        if relevant_text.startswith("ERROR"):
            raise HTTPException(status_code=422, detail=relevant_text)

        # ── Step 2: Summarize the extracted text ──────────────────────────
        summary = summarizer.summarize(relevant_text)

        # ── Step 3: Generate platform-specific posts ──────────────────────
        posts = post_generator.generate(summary, platform)

        # ── Return full result ────────────────────────────────────────────
        return {
            "source": "document_upload",
            "original_filename": file.filename,
            "platform": platform,
            "structured_summary": summary.model_dump(),
            "linkedin_post": posts.get("linkedin_post"),
            "instagram_post": posts.get("instagram_post"),
        }

    finally:
        # Always clean up the uploaded file after processing
        cleanup_file(file_path)
