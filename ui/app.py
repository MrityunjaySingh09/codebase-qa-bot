"""
ui/app.py — Streamlit UI for the Codebase Q&A Bot.

UI structure:
  - Sidebar:  repo URL input, indexing progress, indexed repos list,
              settings (top-k, model info)
  - Main:     repo summary card, streaming chat interface,
              source citation expanders per answer

Design:
  - Dark terminal-inspired theme with green accents (code aesthetic)
  - Streaming tokens render in real time using st.write_stream pattern
  - Source citations shown as collapsible expanders with syntax highlighting
  - Session state manages chat history and active repo across reruns
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pathlib import Path

# ─── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Codebase Q&A Bot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

  /* Root variables */
  :root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2128;
    --border:        #30363d;
    --green:         #3fb950;
    --green-dim:     #238636;
    --blue:          #58a6ff;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --text-dim:      #484f58;
    --orange:        #d29922;
    --red:           #f85149;
  }

  /* Global */
  .stApp { background: var(--bg-primary); font-family: 'Inter', sans-serif; }
  .main .block-container { padding: 1.5rem 2rem; max-width: 1100px; }

  /* Hide default Streamlit elements */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] .block-container { padding: 1rem; }

  /* Custom header */
  .app-header {
    display: flex; align-items: center; gap: 12px;
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
  }
  .app-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem; font-weight: 600;
    color: var(--green); margin: 0;
    letter-spacing: -0.5px;
  }
  .app-header .subtitle {
    font-size: 0.78rem; color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
  }

  /* Repo summary card */
  .repo-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.5rem;
  }
  .repo-card h3 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem; color: var(--blue);
    margin: 0 0 0.6rem 0;
  }
  .repo-meta {
    display: flex; flex-wrap: wrap; gap: 10px;
    margin-bottom: 0.8rem;
  }
  .repo-badge {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: var(--text-secondary);
  }
  .repo-badge.green { border-color: var(--green-dim); color: var(--green); }
  .repo-badge.blue  { border-color: #1f6feb; color: var(--blue); }

  /* Chat messages */
  .chat-container { display: flex; flex-direction: column; gap: 1rem; }

  .msg-human {
    display: flex; justify-content: flex-end;
    margin: 0.3rem 0;
  }
  .msg-human .bubble {
    background: #1f3a5f;
    border: 1px solid #1f6feb;
    border-radius: 12px 12px 2px 12px;
    padding: 0.7rem 1rem;
    max-width: 75%;
    font-size: 0.9rem; color: var(--text-primary);
    line-height: 1.5;
  }

  .msg-ai { display: flex; align-items: flex-start; gap: 10px; margin: 0.3rem 0; }
  .msg-ai .avatar {
    width: 28px; height: 28px; border-radius: 50%;
    background: var(--green-dim);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 600; color: white;
    flex-shrink: 0; margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }
  .msg-ai .bubble {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 2px 12px 12px 12px;
    padding: 0.7rem 1rem;
    max-width: 88%;
    font-size: 0.9rem; color: var(--text-primary);
    line-height: 1.6;
  }

  /* Source citation pills */
  .sources-row {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-top: 0.6rem; padding-top: 0.6rem;
    border-top: 1px solid var(--border);
  }
  .source-pill {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: var(--text-secondary);
    cursor: default;
  }
  .source-pill:hover { border-color: var(--blue); color: var(--blue); }

  /* Input area */
  .input-area {
    position: sticky; bottom: 0;
    background: var(--bg-primary);
    padding: 1rem 0 0.5rem 0;
    border-top: 1px solid var(--border);
    margin-top: 1rem;
  }

  /* Streamlit input overrides */
  .stTextInput input, .stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    border-radius: 6px !important;
  }
  .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 2px rgba(63,185,80,0.15) !important;
  }

  /* Buttons */
  .stButton button {
    background: var(--green-dim) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: background 0.15s !important;
  }
  .stButton button:hover { background: var(--green) !important; }

  .stButton.secondary button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
  }

  /* Progress bar */
  .stProgress > div > div > div > div {
    background: var(--green) !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
  }
  .streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
  }

  /* Sidebar label styling */
  .sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 1px;
    margin: 1rem 0 0.3rem 0;
  }

  /* Indexed repo item */
  .repo-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 10px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: border-color 0.15s;
  }
  .repo-item.active { border-color: var(--green); }
  .repo-item:hover  { border-color: var(--blue); }
  .repo-item-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: var(--text-primary);
  }
  .repo-item-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: var(--text-dim);
  }

  /* Welcome screen */
  .welcome {
    text-align: center; padding: 4rem 2rem;
    color: var(--text-secondary);
  }
  .welcome h2 {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary); font-size: 1.3rem;
    margin-bottom: 0.5rem;
  }
  .welcome p { font-size: 0.9rem; line-height: 1.6; }
  .welcome .arrow { font-size: 2rem; margin-bottom: 1rem; color: var(--green); }

  /* Code blocks inside chat */
  .msg-ai pre {
    background: #0d1117 !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.8rem !important;
    overflow-x: auto !important;
  }
  .msg-ai code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
  }

  /* Metrics row */
  .metrics-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 0.5rem 0; }
  .metric-chip {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    text-align: center;
  }
  .metric-chip .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem; font-weight: 600; color: var(--green);
    display: block;
  }
  .metric-chip .lbl {
    font-size: 0.68rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.5px;
  }

  /* Typing indicator */
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
  .typing-cursor {
    display: inline-block; width: 8px; height: 1em;
    background: var(--green); margin-left: 2px;
    animation: blink 1s infinite; vertical-align: text-bottom;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-secondary); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
</style>
""", unsafe_allow_html=True)


# ─── Session state initialisation ──────────────────────────────────────────
def _init_state():
    defaults = {
        "active_repo_url": None,
        "active_repo_info": None,
        "repo_summary": None,
        "repo_summary_text": None,
        "chat_history": [],        # list of (question, answer, sources)
        "lc_history": [],          # list of (question, answer) for LangChain
        "indexing": False,
        "index_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── Helpers ───────────────────────────────────────────────────────────────

def _lang_badge(lang: str) -> str:
    colors = {
        "python": "#3572A5", "javascript": "#f1e05a", "typescript": "#2b7489",
        "java": "#b07219",   "go": "#00ADD8",         "rust": "#dea584",
        "ruby": "#701516",   "cpp": "#f34b7d",        "csharp": "#178600",
    }
    color = colors.get(lang.lower(), "#8b949e")
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'padding:2px 8px;background:#1c2128;border:1px solid #30363d;'
        f'border-radius:20px;font-family:\'JetBrains Mono\',monospace;'
        f'font-size:0.72rem;color:#8b949e;">'
        f'<span style="width:8px;height:8px;border-radius:50%;'
        f'background:{color};display:inline-block;"></span>{lang}</span>'
    )


def _file_badge(path: str, start: int, end: int) -> str:
    fname = path.split("/")[-1]
    return (
        f'<span class="source-pill" title="{path} lines {start}-{end}">'
        f'📄 {fname}:{start}</span>'
    )


# ─── Sidebar ───────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="padding:0.5rem 0 1.2rem 0;border-bottom:1px solid #30363d;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;
                      font-weight:600;color:#3fb950;">⬡ CodebaseQ</div>
          <div style="font-size:0.7rem;color:#484f58;margin-top:2px;">
            RAG · Hybrid Search · Local LLM
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Index a new repo ──
        st.markdown('<div class="sidebar-label">Index Repository</div>',
                    unsafe_allow_html=True)

        repo_url = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/owner/repo",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            index_btn = st.button("⬇ Index Repo", use_container_width=True)
        with col2:
            force = st.checkbox("Re-clone", value=False, help="Force fresh clone")

        if index_btn and repo_url:
            _run_indexing(repo_url.strip(), force_reclone=force)

        # ── Indexed repos ──
        st.markdown('<div class="sidebar-label">Indexed Repos</div>',
                    unsafe_allow_html=True)
        _render_repo_list()

        # ── Settings ──
        with st.expander("⚙ Settings", expanded=False):
            from config import get_settings
            cfg = get_settings()
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.73rem;
                        color:#8b949e;line-height:2;">
              🤖 LLM: <span style="color:#e6edf3">{cfg.ollama_llm_model}</span><br>
              📐 Embed: <span style="color:#e6edf3">{cfg.ollama_embed_model}</span><br>
              🔍 Vector K: <span style="color:#e6edf3">{cfg.vector_top_k}</span><br>
              🔤 BM25 K: <span style="color:#e6edf3">{cfg.bm25_top_k}</span><br>
              🏆 Rerank K: <span style="color:#e6edf3">{cfg.rerank_top_k}</span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🗑 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.lc_history = []
                st.rerun()


def _run_indexing(url: str, force_reclone: bool = False):
    """Run the full ingestion pipeline with live progress in the sidebar."""
    from ingestion import ingest_repo

    progress_bar = st.sidebar.progress(0.0, text="Starting ...")
    log_placeholder = st.sidebar.empty()
    log_lines = []

    def _on_progress(stage: str, msg: str, pct: float):
        progress_bar.progress(min(pct, 1.0), text=msg[:80])
        log_lines.append(msg)
        log_placeholder.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;'
            f'color:#8b949e;max-height:80px;overflow:hidden;">'
            + "<br>".join(log_lines[-4:]) + "</div>",
            unsafe_allow_html=True,
        )

    try:
        result = ingest_repo(url, force_reclone=force_reclone,
                             progress_callback=_on_progress)
        progress_bar.progress(1.0, text="✅ Done!")
        log_placeholder.empty()

        # Switch active repo
        st.session_state.active_repo_url = result["repo_info"].url
        st.session_state.active_repo_info = result
        st.session_state.chat_history = []
        st.session_state.lc_history = []
        st.session_state.repo_summary = result["summary"]
        st.session_state.repo_summary_text = None  # will be generated on main page

        st.sidebar.success(
            f"✅ Indexed {result['embed_stats']['new']} new chunks "
            f"({result['embed_stats']['skipped']} cached)"
        )
        st.rerun()

    except Exception as exc:
        progress_bar.empty()
        log_placeholder.empty()
        st.sidebar.error(f"❌ {exc}")


def _render_repo_list():
    """Show all indexed repos as clickable items."""
    try:
        from ingestion.embedder import list_indexed_repos
        repos = list_indexed_repos()
    except Exception:
        repos = []

    if not repos:
        st.markdown(
            '<div style="font-size:0.78rem;color:#484f58;padding:0.5rem 0;">'
            'No repos indexed yet.</div>',
            unsafe_allow_html=True,
        )
        return

    for repo in repos:
        col_name = repo["collection"]
        # Derive display name from collection name
        prefix = "codebase_"
        slug = col_name[len(prefix):] if col_name.startswith(prefix) else col_name
        display = slug.replace("__", "/")

        # Reconstruct URL from slug
        parts = slug.split("__")
        repo_url = f"https://github.com/{'/'.join(parts)}" if len(parts) == 2 else slug

        is_active = st.session_state.active_repo_url == repo_url
        border = "#3fb950" if is_active else "#30363d"

        st.markdown(
            f'<div class="repo-item{"  active" if is_active else ""}">'
            f'  <span class="repo-item-name">📦 {display}</span>'
            f'  <span class="repo-item-count">{repo["chunk_count"]} chunks</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if not is_active:
            if st.button(f"Switch", key=f"switch_{col_name}",
                         help=f"Switch to {display}"):
                st.session_state.active_repo_url = repo_url
                st.session_state.chat_history = []
                st.session_state.lc_history = []
                st.session_state.repo_summary_text = None
                st.rerun()


# ─── Repo summary card ─────────────────────────────────────────────────────

def render_repo_header():
    info  = st.session_state.active_repo_info
    summary = st.session_state.repo_summary

    if not st.session_state.active_repo_url:
        return

    url = st.session_state.active_repo_url
    parts = url.replace("https://github.com/", "").split("/")
    repo_name = "/".join(parts) if len(parts) == 2 else url

    st.markdown(f"""
    <div class="repo-card">
      <h3>📦 {repo_name}</h3>
      <div class="repo-meta">
        <span class="repo-badge blue">🔗 {url}</span>
    """, unsafe_allow_html=True)

    if summary:
        total = summary.get("total_files", "?")
        langs = summary.get("languages", {})
        st.markdown(
            f'<span class="repo-badge green">📄 {total} files</span>',
            unsafe_allow_html=True,
        )
        lang_badges = " ".join(_lang_badge(l) for l in list(langs.keys())[:6])
        st.markdown(lang_badges + "</div>", unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Generate repo summary text on first visit
    if st.session_state.repo_summary_text is None and info is not None:
        with st.spinner("Generating repo overview ..."):
            try:
                from rag.pipeline import generate_repo_summary, build_file_tree
                from retrieval import vector_search

                file_tree = build_file_tree(info["repo_info"].local_path)
                samples = vector_search(
                    "main entry point architecture overview",
                    info["repo_info"].url, top_k=5,
                )
                st.session_state.repo_summary_text = generate_repo_summary(
                    info["repo_info"].url, file_tree, samples
                )
            except Exception as exc:
                st.session_state.repo_summary_text = f"_Summary unavailable: {exc}_"

    if st.session_state.repo_summary_text:
        with st.expander("📋 Repo Overview", expanded=True):
            st.markdown(st.session_state.repo_summary_text)


# ─── Chat interface ─────────────────────────────────────────────────────────

def render_chat():
    history = st.session_state.chat_history

    if not history:
        st.markdown("""
        <div class="welcome">
          <div class="arrow">⬡</div>
          <h2>Ask anything about the codebase</h2>
          <p>
            Try: <em>"How does authentication work?"</em><br>
            Or: <em>"Where is the database connection initialised?"</em><br>
            Or: <em>"What does the PaymentService class do?"</em>
          </p>
        </div>
        """, unsafe_allow_html=True)
        return

    for i, (question, answer_text, sources) in enumerate(history):
        # Human bubble
        st.markdown(
            f'<div class="msg-human"><div class="bubble">{question}</div></div>',
            unsafe_allow_html=True,
        )

        # AI bubble
        st.markdown('<div class="msg-ai"><div class="avatar">AI</div>'
                    '<div class="bubble" id="ai-msg-{i}">', unsafe_allow_html=True)
        st.markdown(answer_text)

        # Source pills
        if sources:
            pills = " ".join(
                _file_badge(s.file_path, s.start_line, s.end_line)
                for s in sources
            )
            st.markdown(
                f'<div class="sources-row">'
                f'<span style="font-size:0.7rem;color:#484f58;margin-right:4px;">Sources:</span>'
                f'{pills}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Source code expanders
        if sources:
            with st.expander(
                f"🔍 View {len(sources)} source chunk{'s' if len(sources)>1 else ''}",
                expanded=False,
            ):
                for j, src in enumerate(sources, 1):
                    st.markdown(
                        f'<div style="font-family:\'JetBrains Mono\',monospace;'
                        f'font-size:0.75rem;color:#8b949e;margin-bottom:4px;">'
                        f'[{j}] <span style="color:#58a6ff">{src.file_path}</span>'
                        + (f' → <span style="color:#3fb950">{src.symbol_name}</span>'
                           if src.symbol_name else "")
                        + f' <span style="color:#484f58">lines {src.start_line}–{src.end_line}</span>'
                        + f'</div>',
                        unsafe_allow_html=True,
                    )
                    lang = src.language or "text"
                    st.code(src.content, language=lang)

        st.markdown(
            '<hr style="border:none;border-top:1px solid #21262d;margin:0.8rem 0;">',
            unsafe_allow_html=True,
        )


def render_streaming_response(question: str):
    """Run RAG pipeline and stream response into the UI in real time."""
    from rag.pipeline import stream_answer

    repo_url = st.session_state.active_repo_url

    # Human bubble
    st.markdown(
        f'<div class="msg-human"><div class="bubble">{question}</div></div>',
        unsafe_allow_html=True,
    )

    # AI bubble with streaming
    st.markdown('<div class="msg-ai"><div class="avatar">AI</div>'
                '<div class="bubble">', unsafe_allow_html=True)

    answer_placeholder = st.empty()
    sources_placeholder = st.empty()

    full_answer = ""
    retrieved_sources = []

    try:
        for event in stream_answer(
            question,
            repo_url,
            chat_history=st.session_state.lc_history,
        ):
            if event["type"] == "retrieval":
                retrieved_sources = event["results"]
                if retrieved_sources:
                    pills = " ".join(
                        _file_badge(s.file_path, s.start_line, s.end_line)
                        for s in retrieved_sources
                    )
                    sources_placeholder.markdown(
                        f'<div style="font-size:0.72rem;color:#484f58;margin-bottom:8px;">'
                        f'🔍 Searching {len(retrieved_sources)} chunks ... {pills}</div>',
                        unsafe_allow_html=True,
                    )

            elif event["type"] == "token":
                full_answer += event["content"]
                answer_placeholder.markdown(
                    full_answer + '<span class="typing-cursor"></span>',
                    unsafe_allow_html=True,
                )

            elif event["type"] == "done":
                full_answer = event["answer"]
                retrieved_sources = event["sources"]
                sources_placeholder.empty()
                answer_placeholder.markdown(full_answer)

    except Exception as exc:
        full_answer = f"⚠️ Error: {exc}"
        answer_placeholder.markdown(full_answer)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Final source pills
    if retrieved_sources:
        pills = " ".join(
            _file_badge(s.file_path, s.start_line, s.end_line)
            for s in retrieved_sources
        )
        st.markdown(
            f'<div class="sources-row">'
            f'<span style="font-size:0.7rem;color:#484f58;margin-right:4px;">Sources:</span>'
            f'{pills}</div>',
            unsafe_allow_html=True,
        )

    # Persist to session history
    st.session_state.chat_history.append((question, full_answer, retrieved_sources))
    st.session_state.lc_history.append((question, full_answer))

    # Source expander
    if retrieved_sources:
        with st.expander(
            f"🔍 View {len(retrieved_sources)} source chunk{'s' if len(retrieved_sources)>1 else ''}",
            expanded=False,
        ):
            for j, src in enumerate(retrieved_sources, 1):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:0.75rem;color:#8b949e;margin-bottom:4px;">'
                    f'[{j}] <span style="color:#58a6ff">{src.file_path}</span>'
                    + (f' → <span style="color:#3fb950">{src.symbol_name}</span>'
                       if src.symbol_name else "")
                    + f' <span style="color:#484f58">lines {src.start_line}–{src.end_line}</span>'
                    + f'</div>',
                    unsafe_allow_html=True,
                )
                st.code(src.content, language=src.language or "text")

    st.markdown(
        '<hr style="border:none;border-top:1px solid #21262d;margin:0.8rem 0;">',
        unsafe_allow_html=True,
    )

    return full_answer, retrieved_sources


# ─── Input bar ─────────────────────────────────────────────────────────────

def render_input_bar():
    if not st.session_state.active_repo_url:
        return

    st.markdown('<div class="input-area">', unsafe_allow_html=True)

    # Suggested questions (shown only on empty chat)
    if not st.session_state.chat_history:
        suggestions = [
            "How does authentication work?",
            "Where is the database connection initialised?",
            "Show me all API endpoints",
            "What design patterns are used?",
        ]
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            with col:
                if st.button(
                    suggestion,
                    key=f"suggest_{suggestion[:20]}",
                    use_container_width=True,
                ):
                    st.session_state["_pending_question"] = suggestion
                    st.rerun()

    # Main text input + send button
    col_input, col_send = st.columns([10, 1])
    with col_input:
        question = st.text_input(
            "Ask a question",
            key="chat_input",
            placeholder="Ask anything about the codebase ...",
            label_visibility="collapsed",
        )
    with col_send:
        send = st.button("→", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Handle question submission
    pending = st.session_state.pop("_pending_question", None)
    final_question = pending or (question if send and question.strip() else None)

    if final_question:
        render_streaming_response(final_question.strip())
        st.rerun()


# ─── Main app ──────────────────────────────────────────────────────────────

def main():
    # Sidebar
    render_sidebar()

    # Header
    st.markdown("""
    <div class="app-header">
      <div>
        <h1>⬡ Codebase Q&A</h1>
        <div class="subtitle">Hybrid RAG · AST Chunking · Local LLM</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # If no repo selected, show landing
    if not st.session_state.active_repo_url:
        st.markdown("""
        <div class="welcome">
          <div class="arrow">⬡</div>
          <h2>Index a GitHub repo to get started</h2>
          <p>
            Paste any public GitHub URL in the sidebar and click <strong>Index Repo</strong>.<br>
            The bot will clone it, parse every file with AST-aware chunking,<br>
            embed with <code>nomic-embed-text</code>, and store in ChromaDB.<br><br>
            Then ask anything — authentication flow, class responsibilities,<br>
            database setup, API endpoints — and get cited, file-level answers.
          </p>
          <div style="margin-top:1.5rem;display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
            <span style="padding:4px 12px;background:#1c2128;border:1px solid #30363d;
                         border-radius:20px;font-size:0.78rem;color:#8b949e;">
              🐍 Python AST chunking
            </span>
            <span style="padding:4px 12px;background:#1c2128;border:1px solid #30363d;
                         border-radius:20px;font-size:0.78rem;color:#8b949e;">
              🔀 Hybrid BM25 + Vector search
            </span>
            <span style="padding:4px 12px;background:#1c2128;border:1px solid #30363d;
                         border-radius:20px;font-size:0.78rem;color:#8b949e;">
              🏆 Cross-encoder reranking
            </span>
            <span style="padding:4px 12px;background:#1c2128;border:1px solid #30363d;
                         border-radius:20px;font-size:0.78rem;color:#8b949e;">
              🦙 Local Ollama LLM
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Repo header + summary
    render_repo_header()

    # Tabs: Chat | Evaluation
    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

    with tab_chat:
        render_chat()
        render_input_bar()

    with tab_eval:
        render_eval_panel()


# ─── Eval panel ────────────────────────────────────────────────────────────

def render_eval_panel():
    """Evaluation tab: run RAGAS and display scores."""
    from eval.ragas_eval import (
        load_latest_results,
        get_results_markdown_table,
        DEFAULT_QUESTIONS,
    )

    st.markdown("### 📊 RAGAS Evaluation")
    st.markdown(
        '<div style="font-size:0.82rem;color:#8b949e;margin-bottom:1rem;">'
        "Measure RAG pipeline quality: Faithfulness · Answer Relevancy · "
        "Context Precision · Context Recall"
        "</div>",
        unsafe_allow_html=True,
    )

    repo_url = st.session_state.active_repo_url
    if not repo_url:
        st.info("Index a repo first to run evaluation.")
        return

    # Load existing results
    existing = load_latest_results(repo_url)

    col_run, col_info = st.columns([1, 2])
    with col_run:
        num_q = st.slider("Questions to evaluate", 3, len(DEFAULT_QUESTIONS),
                          value=5, step=1)
        run_eval = st.button("▶ Run Evaluation", use_container_width=True)

    with col_info:
        st.markdown(
            '<div style="font-size:0.78rem;color:#8b949e;line-height:1.8;">'
            "⏱ ~2–5 min on CPU<br>"
            "🦙 Uses local Ollama as RAGAS judge<br>"
            "💾 Results saved to <code>eval/results/</code>"
            "</div>",
            unsafe_allow_html=True,
        )

    if run_eval:
        with st.spinner("Running RAGAS evaluation ..."):
            try:
                from eval.ragas_eval import run_evaluation
                results = run_evaluation(
                    repo_url=repo_url,
                    questions=DEFAULT_QUESTIONS[:num_q],
                )
                st.success("✅ Evaluation complete!")
                existing = results
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    # Display results
    if existing:
        agg = existing["aggregate"]

        def _score_color(v):
            if v is None: return "#484f58"
            if v >= 0.8:  return "#3fb950"
            if v >= 0.6:  return "#d29922"
            return "#f85149"

        def _fmt(v):
            return f"{v:.3f}" if v is not None else "N/A"

        metrics = [
            ("Faithfulness",      agg.get("faithfulness"),
             "Are claims backed by retrieved context?"),
            ("Answer Relevancy",  agg.get("answer_relevancy"),
             "Does the answer address the question?"),
            ("Context Precision", agg.get("context_precision"),
             "Are retrieved chunks relevant?"),
            ("Context Recall",    agg.get("context_recall"),
             "Were the right chunks retrieved?"),
        ]

        st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
        for name, val, tip in metrics:
            color = _score_color(val)
            bar_w = int((val or 0) * 60)
            st.markdown(
                f'<div class="metric-chip" title="{tip}">'
                f'  <span class="val" style="color:{color}">{_fmt(val)}</span>'
                f'  <div style="height:4px;background:#21262d;border-radius:2px;'
                f'       margin:4px 0;width:60px;">'
                f'    <div style="height:4px;background:{color};border-radius:2px;'
                f'         width:{bar_w}px;"></div></div>'
                f'  <span class="lbl">{name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Markdown table for copying into README
        with st.expander("📋 Copy Markdown table for README", expanded=False):
            table = get_results_markdown_table(existing)
            st.code(table, language="markdown")

        # Per-question breakdown
        with st.expander("🔍 Per-question breakdown", expanded=False):
            for i, sample in enumerate(existing.get("per_question", []), 1):
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:0.78rem;color:#58a6ff;margin-top:0.8rem;">'
                    f'Q{i}: {sample["question"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="font-size:0.82rem;color:#e6edf3;'
                    f'margin:4px 0 0 0;">{sample["answer"][:300]}'
                    + ("..." if len(sample["answer"]) > 300 else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                if sample.get("source_files"):
                    files = " · ".join(
                        f'<code style="font-size:0.7rem">{f}</code>'
                        for f in sample["source_files"][:3]
                    )
                    st.markdown(
                        f'<div style="margin-top:4px;color:#484f58;'
                        f'font-size:0.72rem;">Sources: {files}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(
            '<div style="color:#484f58;font-size:0.85rem;padding:1rem 0;">'
            "No evaluation results yet. Click ▶ Run Evaluation above."
            "</div>",
            unsafe_allow_html=True,
        )


# ─── Updated main with tabs ─────────────────────────────────────────────────

if __name__ == "__main__":
    main()
