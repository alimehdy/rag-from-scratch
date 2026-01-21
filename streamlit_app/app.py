import sys
from pathlib import Path
from ollama import chat
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
    sys.path.append(str(ROOT_DIR / "rag_core"))
import streamlit as st
from rag_core.rag_pipeline import pipeline
from config.rag_settings import llm_model_name

# from rag_core.rag_pipeline import pipeline
# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="RAG Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Sidebar (PDF retrieval placeholder)
# -------------------------------
st.sidebar.title("📂 Relevant PDFs")
st.sidebar.info("Later, you can add PDF search results here based on RAG pipeline.")

# -------------------------------
# Top navigation pages
# -------------------------------
pages = ["Chat", "Embeddings", "Previous Search"]
selected_page = st.tabs(pages)

# -------------------------------
# Chat Page
# -------------------------------
with selected_page[0]:
    st.header("💬 Chat Interface")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        st.text_area(
            key="user_input",
            placeholder="Type your message:",
            label="User Message",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Send")

    if submitted:
        user_input = st.session_state.user_input
        st.write(f"Your query: {user_input}")
        if user_input.strip():
            st.session_state.chat_history.append(
                {"role": "user", "text": user_input}
            )

            with st.spinner("Processing your query..."):
                resp, relevant_files, exec_time = pipeline(user_input)
            st.write(relevant_files)
            st.write(f"**Assistant:** {resp}")
            for i, doc in enumerate(relevant_files):
                pdf_path = Path(doc["text_path"])

                if st.sidebar.button(doc["title"], key=f"pdf_{i}"):
                    with open(pdf_path, "rb") as f:
                        st.sidebar.download_button(
                            label="Download PDF",
                            data=f,
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            key=f"download_{i}"
                        )
            st.caption(f"⏱ {exec_time:.2f} seconds")

    for msg in st.session_state.chat_history:
        st.markdown(
            f"**You:** {msg['text']}"
            if msg["role"] == "user"
            else f"**Assistant:** {msg['text']}"
        )
# -------------------------------
# Embeddings Page
# -------------------------------
with selected_page[1]:
    st.header("🧠 Embeddings")
    st.write("This page will show embedding visualizations or management tools.")

# -------------------------------
# Previous Search Page
# -------------------------------
with selected_page[2]:
    st.header("🔍 Previous Search")
    st.write("This page will list past queries and retrieved documents.")
