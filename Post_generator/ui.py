"""
ui.py — Streamlit UI for the Content Generator AI.

LAYOUT:
  Sidebar  → Input controls (type, content, platform, mode, buttons)
  Main     → Log panel + Generated post cards

HOW TO RUN (keep FastAPI running in another terminal first):
  .venv\\Scripts\\streamlit run ui.py

HOW SESSION STATE WORKS IN STREAMLIT:
  Streamlit re-runs the ENTIRE script on every user interaction.
  st.session_state = a dict that persists between re-runs.
  We store the API response in session_state so it survives re-runs.
  The "Refresh" button clears session_state → blank slate.
"""

import streamlit as st
import requests
import json
import time

# ── Page config — must be FIRST streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Content Generator AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api"
TIMEOUT  = 300   # 5 min — local LLM is slow

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Main background */
    .stApp { background: #0f172a; }

    /* Cards */
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card-linkedin { border-left: 4px solid #0077b5; }
    .card-instagram { border-left: 4px solid #e1306c; }
    .card-summary  { border-left: 4px solid #7c3aed; }
    .card-log      { border-left: 4px solid #059669; }
    .card-media    { border-left: 4px solid #f59e0b; }

    /* Tag pills */
    .tag {
        display: inline-block;
        background: #334155;
        color: #94a3b8;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px;
    }

    /* Section headers */
    .section-title {
        color: #94a3b8;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* Post text */
    .post-text {
        color: #e2e8f0;
        font-size: 14px;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    /* Log line */
    .log-line {
        color: #4ade80;
        font-family: monospace;
        font-size: 13px;
        padding: 3px 0;
    }
    .log-step { color: #60a5fa; }
    .log-done { color: #4ade80; }
    .log-warn { color: #fbbf24; }

    /* Key point bullet */
    .kp { color: #a78bfa; font-size: 14px; padding: 4px 0; }

    /* Stat badge */
    .stat {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 16px;
        text-align: center;
    }
    .stat-val { color: #7c3aed; font-size: 22px; font-weight: 700; }
    .stat-lbl { color: #64748b; font-size: 11px; }

    /* Hide streamlit default elements */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Header
    st.markdown("## ✍️ Content Generator AI")
    st.markdown("<p style='color:#64748b; font-size:12px;'>Article → LinkedIn + Instagram Post</p>", unsafe_allow_html=True)
    st.divider()

    # ── Input Type ────────────────────────────────────────────────────────────
    st.markdown("**INPUT TYPE**")
    input_type = st.radio(
        label="input_type",
        options=["URL", "Paste Text", "Topic", "Document"],
        label_visibility="collapsed",
    )

    st.markdown("")  # spacer

    # ── Dynamic input based on type ───────────────────────────────────────────
    content = None
    uploaded_file = None

    if input_type == "URL":
        st.markdown("**Article URL**")
        content = st.text_input(
            "url_input",
            placeholder="https://www.ibm.com/think/topics/rag",
            label_visibility="collapsed",
        )

    elif input_type == "Paste Text":
        st.markdown("**Paste Article (2-3 paragraphs)**")
        content = st.text_area(
            "text_input",
            placeholder="Paste article paragraphs here...",
            height=160,
            label_visibility="collapsed",
        )

    elif input_type == "Topic":
        st.markdown("**Topic / Keyword**")
        content = st.text_input(
            "topic_input",
            placeholder="e.g. why RAG beats fine-tuning LLMs",
            label_visibility="collapsed",
        )

    elif input_type == "Document":
        st.markdown("**Upload PDF or DOCX**")
        uploaded_file = st.file_uploader(
            "doc_upload",
            type=["pdf", "docx", "doc"],
            label_visibility="collapsed",
        )
        content = uploaded_file.name if uploaded_file else None

    st.divider()

    # ── Platform ──────────────────────────────────────────────────────────────
    st.markdown("**PLATFORM**")
    platform_display = st.radio(
        "platform_radio",
        options=["Both", "LinkedIn Only", "Instagram Only"],
        label_visibility="collapsed",
    )
    platform_map = {
        "Both": "both",
        "LinkedIn Only": "linkedin",
        "Instagram Only": "instagram",
    }
    platform = platform_map[platform_display]

    st.divider()

    # ── Agent Mode ────────────────────────────────────────────────────────────
    st.markdown("**AGENT MODE**")
    mode_display = st.radio(
        "mode_radio",
        options=["Pipeline (Fast)", "LLM Agent (Autonomous)"],
        label_visibility="collapsed",
    )
    mode = "pipeline" if "Pipeline" in mode_display else "llm"
    st.markdown(
        f"<p style='color:#64748b; font-size:11px;'>"
        f"{'Rule-based routing. Fast & reliable.' if mode == 'pipeline' else 'LLM decides tool sequence. Slower but agentic.'}"
        f"</p>",
        unsafe_allow_html=True
    )

    st.divider()

    # ── Media Mode ─────────────────────────────────────────────────
    st.markdown("**MEDIA**")
    media_mode_display = st.radio(
        "media_mode_radio",
        options=["🤖 AI Smart (auto-select)", "🎨 Manual Select", "🚫 No Media"],
        label_visibility="collapsed",
    )
    media_mode_map = {
        "🤖 AI Smart (auto-select)": "ai_smart",
        "🎨 Manual Select": "manual",
        "🚫 No Media": "no_media",
    }
    media_mode = media_mode_map[media_mode_display]

    # Media type dropdown — only shown in Manual mode
    media_type = None
    STATIC_TYPES  = ["INFOGRAPHIC", "THEME_IMAGE", "DIAGRAM", "INFLUENCER", "PDF_1PAGER"]
    PHASE2_TYPES  = ["WORKFLOW_GIF", "SHORT_VIDEO", "INFO_GIF"]
    ALL_MEDIA_TYPES = STATIC_TYPES + PHASE2_TYPES

    if media_mode == "manual":
        media_type = st.selectbox(
            "media_type_select",
            options=ALL_MEDIA_TYPES,
            format_func=lambda x: (
                f"{x}" if x in STATIC_TYPES else f"{x} ⏳ Phase 2"
            ),
            label_visibility="collapsed",
        )
        if media_type in PHASE2_TYPES:
            st.markdown(
                "<p style='color:#fbbf24; font-size:11px;'>"
                "⚠️ GIF/Video types are Phase 2 (HeyGen). A placeholder note will be shown."
                "</p>", unsafe_allow_html=True
            )
    elif media_mode == "ai_smart":
        st.markdown(
            "<p style='color:#64748b; font-size:11px;'>"
            "LLM analyzes content and picks the best format automatically."
            "</p>", unsafe_allow_html=True
        )

    st.divider()

    # ── Action Buttons ────────────────────────────────────────────────────────
    generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)
    refresh_btn  = st.button("🔄 Refresh / Clear", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# session_state persists between Streamlit re-runs
# ─────────────────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "error" not in st.session_state:
    st.session_state.error = None


# ─────────────────────────────────────────────────────────────────────────────
# REFRESH BUTTON — clears everything
# ─────────────────────────────────────────────────────────────────────────────
if refresh_btn:
    st.session_state.result = None
    st.session_state.logs   = []
    st.session_state.error  = None
    st.rerun()   # re-run the script from top with cleared state


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE BUTTON — calls the FastAPI backend
# ─────────────────────────────────────────────────────────────────────────────
if generate_btn:

    # ── Validation ───────────────────────────────────────────────────────────
    if input_type == "Document" and uploaded_file is None:
        st.sidebar.error("Please upload a file.")
    elif input_type != "Document" and not content:
        st.sidebar.error("Please provide input content.")

    else:
        st.session_state.result = None
        st.session_state.logs   = []
        st.session_state.error  = None

        # ── Call API ──────────────────────────────────────────────────────────
        with st.spinner("🤖 Agent is working... (local LLM may take 30-60s)"):
            try:
                t_start = time.time()

                if input_type == "Document":
                    # Multipart file upload
                    file_bytes = uploaded_file.read()
                    resp = requests.post(
                        f"{API_BASE}/generate/document",
                        files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                        data={"platform": platform, "mode": mode},
                        timeout=TIMEOUT,
                    )
                else:
                    # Query-param request
                    input_type_map = {
                        "URL": "url",
                        "Paste Text": "text",
                        "Topic": "topic",
                    }
                    params = {
                        "input_type": input_type_map[input_type],
                        "content": content,
                        "platform": platform,
                        "mode": mode,
                        "media_mode": media_mode,
                    }
                    if media_type:
                        params["media_type"] = media_type
                    resp = requests.get(
                        f"{API_BASE}/generate",
                        params=params,
                        timeout=TIMEOUT,
                    )

                elapsed = round(time.time() - t_start, 1)

                if resp.status_code == 200:
                    st.session_state.result = resp.json()
                    st.session_state.result["_client_elapsed"] = elapsed

                    # Build log lines from response data
                    r = st.session_state.result
                    itype = r.get("input_type", input_type.lower())
                    logs = [
                        ("step", "Agent started"),
                    ]
                    if itype == "url":
                        logs.append(("step", "Tool called: web_scraper.scrape(url)"))
                        logs.append(("done", f"Scraped content acquired"))
                    elif itype == "topic":
                        logs.append(("step", "Tool called: web_search.search(topic)"))
                        logs.append(("done", f"Web search results collected"))
                    elif itype == "document":
                        logs.append(("step", "Tool called: rag_tool.process(file)"))
                        logs.append(("done", f"RAG chunking + embedding + retrieval complete"))
                    else:
                        logs.append(("done", "Pasted text received directly"))

                    logs.append(("step", "Tool called: summarizer.summarize(raw_text)"))
                    s = r.get("structured_summary", {})
                    logs.append(("done", f"Summary complete → topic: '{s.get('topic', '')}'"))

                    if platform in ("linkedin", "both"):
                        logs.append(("step", "Tool called: post_generator.generate_linkedin(summary)"))
                        logs.append(("done", f"LinkedIn post generated ({len(r.get('linkedin_post') or '')} chars)"))
                    if platform in ("instagram", "both"):
                        logs.append(("step", "Tool called: post_generator.generate_instagram(summary)"))
                        logs.append(("done", f"Instagram post generated ({len(r.get('instagram_post') or '')} chars)"))

                    # Media log
                    media = r.get("media", {})
                    if media:
                        mm = media.get("media_mode", "")
                        if mm == "no_media":
                            logs.append(("warn", "Media: skipped (No Media selected)"))
                        elif media.get("is_phase2"):
                            logs.append(("warn", f"Media: {media.get('media_type')} is Phase 2 (HeyGen)"))
                        elif media.get("image_path"):
                            logs.append(("step", f"Tool called: media_tool.generate({media.get('media_type')})"))
                            logs.append(("done", f"Image generated → {media.get('image_path')}"))
                        else:
                            logs.append(("warn", f"Media: {media.get('message', 'no image')}"))

                    logs.append(("done", f"Pipeline complete in {elapsed}s"))
                    st.session_state.logs = logs

                else:
                    st.session_state.error = f"API Error {resp.status_code}: {resp.text}"

            except requests.exceptions.ConnectionError:
                st.session_state.error = "Cannot connect to FastAPI backend at http://localhost:8000. Is the server running?"
            except requests.exceptions.Timeout:
                st.session_state.error = "Request timed out. The LLM is still processing — try again or use a shorter input."
            except Exception as e:
                st.session_state.error = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

# ── Error display ─────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(st.session_state.error)

# ── Empty state ───────────────────────────────────────────────────────────────
elif st.session_state.result is None:
    st.markdown("""
    <div style='text-align:center; padding: 80px 0;'>
        <div style='font-size: 56px; margin-bottom: 16px;'>✍️</div>
        <h2 style='color: #e2e8f0; font-weight: 700;'>Content Generator AI</h2>
        <p style='color: #64748b; font-size: 16px; max-width: 480px; margin: 0 auto;'>
            Select an input type from the sidebar, provide your content,
            choose your platform, and click <strong style='color:#7c3aed'>Generate</strong>.
        </p>
        <br/>
        <div style='display:flex; justify-content:center; gap:12px; flex-wrap:wrap;'>
            <span class='tag'>🌐 URL Scraping</span>
            <span class='tag'>📝 Paste Text</span>
            <span class='tag'>🔍 Topic Search</span>
            <span class='tag'>📄 PDF / DOCX</span>
            <span class='tag'>💼 LinkedIn</span>
            <span class='tag'>📸 Instagram</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Result display ────────────────────────────────────────────────────────────
else:
    r = st.session_state.result
    s = r.get("structured_summary", {})

    # ── STATS ROW ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='stat'>
            <div class='stat-val'>{r.get('_client_elapsed', '?')}s</div>
            <div class='stat-lbl'>Total Time</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='stat'>
            <div class='stat-val'>{r.get('agent_mode', 'pipeline')}</div>
            <div class='stat-lbl'>Agent Mode</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='stat'>
            <div class='stat-val'>{r.get('input_type', '?')}</div>
            <div class='stat-lbl'>Input Type</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='stat'>
            <div class='stat-val'>{r.get('platform', '?')}</div>
            <div class='stat-lbl'>Platform</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── TWO COLUMNS: Log | Summary ─────────────────────────────────────────────
    col_log, col_summary = st.columns([1, 1])

    # ── BACKEND LOG ───────────────────────────────────────────────────────────
    with col_log:
        st.markdown("<div class='card card-log'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⚙️ Backend Processing Log</div>", unsafe_allow_html=True)
        for log_type, msg in st.session_state.logs:
            if log_type == "step":
                st.markdown(f"<div class='log-step'>→ {msg}</div>", unsafe_allow_html=True)
            elif log_type == "done":
                st.markdown(f"<div class='log-done'>✓ {msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='log-warn'>⚠ {msg}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── STRUCTURED SUMMARY ────────────────────────────────────────────────────
    with col_summary:
        st.markdown("<div class='card card-summary'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🧠 Structured Summary</div>", unsafe_allow_html=True)
        st.markdown(f"<b style='color:#e2e8f0'>{s.get('topic', '')}</b>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#94a3b8; font-size:13px; margin:8px 0;'>{s.get('summary', '')}</p>", unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='margin-top:12px;'>Key Points</div>", unsafe_allow_html=True)
        for kp in s.get("key_points", []):
            st.markdown(f"<div class='kp'>▸ {kp}</div>", unsafe_allow_html=True)

        hook = s.get("hook", "")
        if hook:
            st.markdown(f"<div style='margin-top:12px; color:#fbbf24; font-size:13px; font-style:italic;'>💡 Hook: {hook}</div>", unsafe_allow_html=True)

        tone = s.get("tone", "")
        audience = s.get("target_audience", "")
        if tone or audience:
            st.markdown(
                f"<div style='margin-top:10px;'>"
                f"<span class='tag'>🎯 {tone}</span> "
                f"<span class='tag'>👥 {audience}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── GENERATED POSTS ───────────────────────────────────────────────────────
    linkedin_post  = r.get("linkedin_post")
    instagram_post = r.get("instagram_post")

    if linkedin_post and instagram_post:
        tab_li, tab_ig = st.tabs(["💼 LinkedIn Post", "📸 Instagram Post"])
    elif linkedin_post:
        tab_li = st.container()
        tab_ig = None
    else:
        tab_li = None
        tab_ig = st.container()

    # LinkedIn
    if linkedin_post and tab_li is not None:
        with tab_li:
            st.markdown("<div class='card card-linkedin'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>💼 LinkedIn Post &nbsp;"
                "<span style='color:#64748b; font-size:11px;'>Professional · Thought-leader · Unicode bold</span>"
                "</div>",
                unsafe_allow_html=True
            )
            hashtags_li = " ".join(s.get("hashtags", {}).get("linkedin", []))
            st.markdown(f"<div class='post-text'>{linkedin_post}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Copyable version in code block
            with st.expander("📋 Copy LinkedIn Post"):
                st.code(linkedin_post, language=None)

    # Instagram
    if instagram_post and tab_ig is not None:
        with tab_ig:
            st.markdown("<div class='card card-instagram'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>📸 Instagram Caption &nbsp;"
                "<span style='color:#64748b; font-size:11px;'>Casual · Emoji-heavy · 15-20 hashtags</span>"
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown(f"<div class='post-text'>{instagram_post}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("📋 Copy Instagram Caption"):
                st.code(instagram_post, language=None)

    # ── HASHTAGS PREVIEW ──────────────────────────────────────────────────────
    hashtags = s.get("hashtags", {})
    if hashtags:
        st.markdown("<br/>", unsafe_allow_html=True)
        with st.expander("🏷️ Hashtags"):
            hc1, hc2 = st.columns(2)
            with hc1:
                st.markdown("**LinkedIn**")
                li_tags = " ".join(hashtags.get("linkedin", []))
                st.markdown(f"<p style='color:#94a3b8'>{li_tags}</p>", unsafe_allow_html=True)
            with hc2:
                st.markdown("**Instagram**")
                ig_tags = " ".join(hashtags.get("instagram", []))
                st.markdown(f"<p style='color:#94a3b8'>{ig_tags}</p>", unsafe_allow_html=True)

    # ── MEDIA RESULT PANEL ────────────────────────────────────────────────────
    import os
    media = r.get("media", {})
    if media and media.get("media_mode") != "no_media":
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("<div class='card card-media'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🎨 Generated Media</div>", unsafe_allow_html=True)

        media_type_shown = media.get("media_type", "")
        media_mode_shown = media.get("media_mode", "")
        image_path       = media.get("image_path")
        is_phase2        = media.get("is_phase2", False)
        msg              = media.get("message", "")
        prompt_used      = media.get("prompt_used", "")

        # Badge row
        st.markdown(
            f"<span class='tag'>📦 {media_type_shown}</span> "
            f"<span class='tag'>🤖 {media_mode_shown}</span>",
            unsafe_allow_html=True
        )
        st.markdown("")

        if is_phase2:
            st.markdown(f"""
            <div style='background:#1e293b; border:2px dashed #f59e0b; border-radius:10px;
                        padding:24px; text-align:center; margin-top:12px;'>
                <div style='font-size:36px;'>⏳</div>
                <div style='color:#f59e0b; font-size:14px; font-weight:600; margin-top:8px;'>
                    {media_type_shown} — Phase 2
                </div>
                <div style='color:#64748b; font-size:12px; margin-top:6px;'>
                    Animated / Video content coming soon via HeyGen API.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif image_path and os.path.exists(image_path):
            st.image(image_path, caption=f"{media_type_shown} — AI Generated", use_container_width=True)
            st.markdown(f"<p style='color:#64748b; font-size:11px;'>Saved: {image_path}</p>", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style='background:#1e293b; border:2px dashed #ef4444; border-radius:10px;
                        padding:24px; text-align:center; margin-top:12px;'>
                <div style='font-size:32px;'>⚠️</div>
                <div style='color:#ef4444; font-size:13px; margin-top:8px;'>Image generation failed</div>
                <div style='color:#64748b; font-size:12px; margin-top:6px;'>{msg}</div>
            </div>
            """, unsafe_allow_html=True)

        if prompt_used:
            with st.expander("📝 Image Prompt Used"):
                st.code(prompt_used, language=None)

        st.markdown("</div>", unsafe_allow_html=True)
