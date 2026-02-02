import time
import streamlit as st
from rag_core.rag_pipeline import search_and_retrieve
from pathlib import Path

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="RAG Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Session State Initialization
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "reranked_results" not in st.session_state:
    st.session_state.reranked_results = None

if "relevant_files" not in st.session_state:
    st.session_state.relevant_files = None

if "tracked_time" not in st.session_state:
    st.session_state.tracked_time = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "llm_answer" not in st.session_state:
    st.session_state.llm_answer = None
# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📄 Sources Used")
st.sidebar.info(
    "Documents used to generate the answer will appear here.\n\n"
    "This helps you understand where the information comes from."
)

# -------------------------------
# Pages
# -------------------------------
pages = ["Chat", "Knowledge Base", "Evaluation"]
selected_page = st.tabs(pages)

# ==========================================================
# 💬 CHAT PAGE
# ==========================================================
with selected_page[0]:

    st.header("💬 Ask Your Knowledge Base")
    st.caption("Ask questions and get answers based only on your uploaded documents.")

    with st.expander("💡 Example questions you can ask"):
        st.markdown("- What does the contract say about termination?")

    # -------------------------------
    # Question Form
    # -------------------------------
    with st.form("chat_form", clear_on_submit=True):
        st.text_area(
            key="user_input",
            placeholder="Ask a question about your documents…",
            label="User Message",
            label_visibility="collapsed",
            height=120
        )
        submitted = st.form_submit_button("Ask")

    # -------------------------------
    # When user submits a question
    # -------------------------------
    if submitted and st.session_state.user_input.strip():

        user_input = st.session_state.user_input
        st.session_state.last_question = user_input
        st.session_state.chat_history.append({"role": "user", "text": user_input})

        with st.spinner("🤖 Looking through your documents…"):
            llm_answer, reranked_results, relevant_files, tracked_time = search_and_retrieve(user_input)

        # Save results to session state (THIS IS THE FIX 🔥)
        st.session_state.reranked_results = reranked_results
        st.session_state.relevant_files = relevant_files
        st.session_state.tracked_time = tracked_time
        st.session_state.llm_answer = llm_answer

    # ==========================================================
    # DISPLAY LLM RESULTS 
    # ==========================================================
    if st.session_state.llm_answer and st.session_state.relevant_files and st.session_state.reranked_results:
        st.markdown("### 🤖 Answer")

        response_container = st.empty()
        full_response = ""

        for token in llm_answer:
            full_response += token
            response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)

        st.session_state.llm_answer = full_response
    
    # if st.session_state.reranked_results:

        reranked_results = st.session_state.reranked_results
        relevant_files = st.session_state.relevant_files
        tracked_time = st.session_state.tracked_time
        user_input = st.session_state.last_question

        st.markdown("### 🔎 Here’s what we found for you")

        with st.container(border=True):
            st.caption("These are the exact sections from your files.")

            for i, (file_info, chunk_info) in enumerate(zip(relevant_files, reranked_results), 1):
                score = file_info.get("rerank_score", 0)
                source = file_info.get("title", "Unknown source")
                text = chunk_info.get("sentence_chunk", "")
                badge = "⭐ Best Match" if i == 1 else f"Document {i}"

                with st.expander(f"{badge} — {source}", expanded=(i == 1)):
                    if score:
                        st.progress(min(score, 1.0))
                        st.caption(f"Relevance score: {score:.2f}")
                    st.write(text)

        # Sidebar PDFs
        st.sidebar.markdown("### 📎 Related Documents")
        for idx, doc in enumerate(relevant_files):
            pdf_path = Path(doc["text_path"])
            if not pdf_path.exists():
                continue

            label = f"⭐ Best Source — {doc['title']}" if idx == 0 else doc["title"]

            with open(pdf_path, "rb") as f:
                st.sidebar.download_button(
                    label=label,
                    data=f,
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=f"pdf_download_{idx}"
                )

        # -------------------------------
        # Performance Metrics
        # -------------------------------
        st.divider()
        st.markdown("#### ⏱ Retrieval Performance")

        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval", f"{tracked_time.get('retrieving_time', 0):.2f}s")
        col2.metric("Reranking", f"{tracked_time.get('reranking_time', 0):.2f}s")
        col3.metric("LLM", f"{tracked_time.get('llm_executing_time', 0):.2f}s")
        total_time = tracked_time.get('retrieving_time', 0) + tracked_time.get('reranking_time', 0) + tracked_time.get('llm_executing_time', 0)
        st.caption(f"⏱ Answer generated in {total_time:.2f} seconds")

        # ==========================================================
        # ⭐ EVALUATION FORM (NO MORE AUTO RERUN)
        # ==========================================================
        with st.expander("⭐ Evaluate this answer", expanded=True):
            with st.form("evaluation_form"):
                rating = st.slider("Helpfulness", 1, 5, 3)
                feedback = st.text_area("Optional feedback")

                eval_submitted = st.form_submit_button("Submit evaluation")

                if eval_submitted:
                    user_evaluation = {
                        "question": user_input,
                        "rating": rating,
                        "feedback": feedback
                    }
                    st.toast("Thanks! Your feedback was saved 🙌", icon="✅")
                    st.write(user_evaluation)
    elif st.session_state.last_question: 
        st.info("🔍 No matching context was retrieved from the knowledge base for this query.")

# ==========================================================
# 🧠 KNOWLEDGE BASE PAGE
# ==========================================================
with selected_page[1]:
    st.header("🧠 Knowledge Base")
    st.info("Upload and manage documents used to answer questions.")

# ==========================================================
# ⚖️ EVALUATION PAGE
# ==========================================================
with selected_page[2]:
    st.header("⚖️ Evaluation")
    st.info("Review answer quality and compare different AI models.")
