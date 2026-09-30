import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from connect_memory_with_llm import (
    BM25Index,
    load_vectorstore_and_indexes,
    get_embedder,
    embed_query,
    retrieve_documents,
    build_context_from_results,
    extract_text_from_pdf,
    extract_text_from_image,
    call_groq,
)

VECTOR_DIR = Path("vectorstore")
CONV_DIR = Path("conversations")
CONV_DIR.mkdir(exist_ok=True)

SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
DEFAULT_MODELS = ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"]
MAX_HISTORY_TURNS = 4

SYSTEM_INSTRUCTION = (
    "You are a careful medical assistant. You MUST use only the provided context. "
    "If the answer is not present in the context, say you don't know. "
    "Do not give treatment plans, dosages, or emergency advice. Always recommend seeing a doctor."
)

PROMPT_TEMPLATE_RAG = """System: {system}

Chat history:
{chat_history}

Retrieved context (from medical documents):
{context}

User question:
{question}

Answer carefully and concisely. Mention when the information comes from the Gale Encyclopedia/context.
"""

PROMPT_TEMPLATE_DOC_ANALYSIS = """System: {system}

The user has uploaded a medical document (e.g., blood report, prescription, medical summary).
Using ONLY the text below, provide:
1. A concise summary (2–4 sentences).
2. Highlight any abnormal findings or important points.
3. Suggest 3–5 questions the patient should ask their doctor.
4. Avoid giving exact treatment or dosage advice.

Uploaded document text:
{doc_text}

User question (if any):
{question}

Answer:
"""

def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "model" not in st.session_state:
        st.session_state["model"] = DEFAULT_MODELS[0]

def parse_users_from_env():
    env_val = os.getenv("APP_USERS", "").strip()
    if not env_val:
        return {"pranav": "medibot123", "admin": "admin123"}
    users = {}
    for pair in env_val.split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            users[u.strip()] = p.strip()
    return users

def add_message(role: str, content: str, sources=None):
    st.session_state["messages"].append(
        {
            "role": role,
            "content": content,
            "sources": sources or [],
            "ts": datetime.now().isoformat(),
        }
    )

def autosave_chat():
    username = st.session_state.get("username") or "default"
    path = CONV_DIR / f"auto_{username}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(st.session_state["messages"], f, ensure_ascii=False, indent=2)

def load_chat_from_file(name: str):
    path = CONV_DIR / name
    try:
        with path.open("r", encoding="utf-8") as f:
            st.session_state["messages"] = json.load(f)
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to load chat: {e}")

def build_chat_history_text(limit_turns: int = MAX_HISTORY_TURNS) -> str:
    msgs = [m for m in st.session_state["messages"] if m["role"] in ("user", "assistant")]
    msgs = msgs[-(limit_turns * 2):]
    lines = []
    for m in msgs:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)

def main():
    st.set_page_config(page_title="MediBot.ai", page_icon="🩺")
    init_session_state()

    users = parse_users_from_env()
    if not st.session_state["authenticated"]:
        col_left, col_center, col_right = st.columns([1, 7, 1])
        with col_center:
            st.title("MediBot.ai - Your Personal Healthcare Assistant")
            st.subheader("Please log in to continue")
            username = st.text_input("Username", key="login_user_center")
            password = st.text_input("Password", type="password", key="login_pass_center")
            if st.button("Login"):
                if username in users and users[username] == password:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.success(f"Logged in as {username}")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        return

    st.title("MediBot — Your Personal Healthcare Assistant")

    embeddings, docs, bm25 = load_vectorstore_and_indexes()
    if embeddings is None or docs is None:
        st.warning("Vectorstore not found. Run `python create_memory_for_llm.py` first for RAG features.")

    st.sidebar.subheader("👤 User")
    st.sidebar.write(f"Logged in as **{st.session_state['username']}**")

    mode = st.sidebar.radio("Choose your Mode", ["Chat with MediBot", "Document Analysis"])
    st.session_state["model"] = st.sidebar.selectbox("Choose LLM Model (Groq)", DEFAULT_MODELS, index=0)

    conv_files = sorted([p.name for p in CONV_DIR.glob("*.json")], reverse=True)
    load_sel = st.sidebar.selectbox("Load chat", ["(none)"] + conv_files)
    if load_sel != "(none)":
        load_chat_from_file(load_sel)
    st.sidebar.caption("Chat is saved automatically.")

    if st.sidebar.button("Clear chat"):
        st.session_state["messages"] = []
        st.rerun()

    for m in st.session_state["messages"]:
        role = m["role"]
        content = m["content"]
        if role == "user":
            st.chat_message("user").markdown(content)
        else:
            cm = st.chat_message("assistant")
            cm.markdown(content)
            if m.get("sources"):
                with cm.expander("Sources / citations"):
                    for s in m["sources"]:
                        st.write(f"- {s}")

    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["messages"] = []
        st.rerun()

    if mode == "Document Analysis":
        st.subheader("Upload a medical document (like blood report, prescription)")

        uploaded = st.file_uploader(
            "Upload PDF or image",
            type=["pdf"] + SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=False,
        )

        user_q = st.text_input(
            "Optional: Ask something about this document (otherwise a general summary will be generated)",
            key="doc_question",
        )

        if uploaded is not None:
            try:
                if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
                    text = extract_text_from_pdf(uploaded)
                else:
                    text = extract_text_from_image(uploaded)

                if not text.strip():
                    st.error("Could not extract any text from the uploaded file.")
                else:
                    st.success("Text extracted from document.")
                    with st.expander("Show raw extracted text"):
                        st.text_area("Document text", text, height=250)

                    prompt = PROMPT_TEMPLATE_DOC_ANALYSIS.format(
                        system=SYSTEM_INSTRUCTION,
                        doc_text=text[:8000],
                        question=user_q or "(no specific question)",
                    )

                    if st.button("Analyze document"):
                        try:
                            answer = call_groq(prompt, model=st.session_state["model"])
                            add_message("user", f"[DOC] {uploaded.name}: {user_q or '(no question, summary requested)'}")
                            add_message("assistant", answer, sources=[f"Uploaded document: {uploaded.name}"])
                            autosave_chat()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Groq error: {e}")

            except Exception as e:
                st.error(f"Error processing document: {e}")

        return

    user_input = st.chat_input("Ask your medical question here...")

    if user_input:
        add_message("user", user_input)

        if embeddings is None or docs is None or bm25 is None:
            answer_text = (
                "Vectorstore is not available, so I cannot look into your medical PDFs. "
                "Please run `create_memory_for_llm.py` to build the document index."
            )
            add_message("assistant", answer_text)
            autosave_chat()
            st.rerun()
            return

        results = retrieve_documents(
            user_input, embeddings=embeddings, docs=docs, bm25=bm25, k=4
        )
        context, sources = build_context_from_results(results)

        max_score = max((r["score"] for r in results), default=0.0)
        if max_score < 0.05:
            answer_text = (
                "I could not find enough relevant information in the provided medical documents "
                "to confidently answer this question. Please consult a doctor or provide more specific context."
            )
            add_message("assistant", answer_text, sources=sources)
            autosave_chat()
            st.rerun()
            return

        chat_history_text = build_chat_history_text(MAX_HISTORY_TURNS)

        prompt = PROMPT_TEMPLATE_RAG.format(
            system=SYSTEM_INSTRUCTION,
            chat_history=chat_history_text,
            context=context,
            question=user_input,
        )

        try:
            answer = call_groq(prompt, model=st.session_state["model"])
        except Exception as e:
            answer = f"Error calling LLM: {e}"

        add_message("assistant", answer, sources=sources)
        autosave_chat()
        st.rerun()

if __name__ == "__main__":
    main()
