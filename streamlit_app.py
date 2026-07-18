"""Minimal Streamlit frontend for the RAG Technical Documentation Assistant.

Run the FastAPI backend first (uvicorn app.main:app --reload), then:
    streamlit run streamlit_app.py
"""
import requests
import streamlit as st

st.set_page_config(page_title="Technical Docs Assistant", page_icon="📚", layout="wide")

DEFAULT_API_URL = "http://localhost:8000"

# --- session state -----------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources"?, "query_id"?}]

# --- sidebar -------------------------------------------------------------
with st.sidebar:
    st.title("📚 Docs Assistant")
    st.caption("Self-corrective RAG · LangGraph + FastAPI")

    st.text_input("API URL", value=DEFAULT_API_URL, key="api_url_input", help="Where the FastAPI backend is running")
    API_URL = st.session_state.api_url_input or DEFAULT_API_URL

    st.divider()

    # Indexed documents
    st.subheader("Indexed documents")
    try:
        resp = requests.get(f"{API_URL}/documents", timeout=5)
        if resp.ok:
            data = resp.json()
            st.metric("Total chunks", data.get("total_chunks", 0))
            for src, count in data.get("sources", {}).items():
                st.caption(f"• {src} — {count} chunks")
        else:
            st.warning(f"Backend returned {resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the API. Is `uvicorn app.main:app --reload` running?")
    except Exception as e:
        st.error(f"Error: {e}")

    st.divider()

    # Ingest new file
    st.subheader("Add a document")
    uploaded = st.file_uploader("Upload a .md or .txt file", type=["md", "txt"])
    if uploaded and st.button("Ingest file", use_container_width=True):
        with st.spinner("Ingesting..."):
            try:
                files = {"files": (uploaded.name, uploaded.getvalue(), "text/plain")}
                r = requests.post(f"{API_URL}/ingest", files=files, timeout=60)
                if r.ok:
                    st.success(f"Indexed {r.json().get('chunks_indexed', 0)} chunks from {uploaded.name}")
                else:
                    st.error(f"Ingest failed: {r.text}")
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# --- main chat area --------------------------------------------------------
st.header("Ask about FastAPI, Pydantic, LangGraph, ChromaDB, or RAG")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                    for s in msg["sources"]:
                        st.caption(f"- **{s['source']}** (chunk {s['chunk_index']}) — graded *{s.get('grade', '?')}*")
            meta_bits = []
            if msg.get("query_type"):
                meta_bits.append(f"type: `{msg['query_type']}`")
            if msg.get("retries_used") is not None:
                meta_bits.append(f"retries: `{msg['retries_used']}`")
            if msg.get("hallucination_grade"):
                meta_bits.append(f"grounded: `{msg['hallucination_grade']}`")
            if msg.get("latency_ms") is not None:
                meta_bits.append(f"{msg['latency_ms']} ms")
            if meta_bits:
                st.caption(" · ".join(meta_bits))

            # Feedback buttons, keyed uniquely per message
            if msg.get("query_id"):
                col1, col2, _ = st.columns([1, 1, 8])
                fb_key = f"fb_{msg['query_id']}"
                if fb_key not in st.session_state:
                    with col1:
                        if st.button("👍", key=f"up_{i}"):
                            try:
                                requests.post(f"{API_URL}/feedback", json={"query_id": msg["query_id"], "rating": "up"}, timeout=5)
                                st.session_state[fb_key] = "up"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Feedback failed: {e}")
                    with col2:
                        if st.button("👎", key=f"down_{i}"):
                            try:
                                requests.post(f"{API_URL}/feedback", json={"query_id": msg["query_id"], "rating": "down"}, timeout=5)
                                st.session_state[fb_key] = "down"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Feedback failed: {e}")
                else:
                    st.caption(f"Feedback recorded: {'👍' if st.session_state[fb_key] == 'up' else '👎'}")

question = st.chat_input("Ask a question about the indexed docs...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, grading, and generating..."):
            try:
                payload = {"question": question}
                if st.session_state.session_id:
                    payload["session_id"] = st.session_state.session_id

                r = requests.post(f"{API_URL}/query", json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()

                st.session_state.session_id = data.get("session_id")
                answer = data.get("answer", "(no answer returned)")
                st.markdown(answer)

                assistant_msg = {
                    "role": "assistant",
                    "content": answer,
                    "sources": data.get("sources", []),
                    "query_type": data.get("query_type"),
                    "retries_used": data.get("retries_used"),
                    "hallucination_grade": data.get("hallucination_grade"),
                    "latency_ms": data.get("latency_ms"),
                    "query_id": data.get("query_id"),
                }
                st.session_state.messages.append(assistant_msg)
                st.rerun()

            except requests.exceptions.ConnectionError:
                err = "Can't reach the API backend. Start it with `uvicorn app.main:app --reload` first."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except requests.exceptions.HTTPError as e:
                err = f"API error: {e.response.status_code} — {e.response.text}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"Something went wrong: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})