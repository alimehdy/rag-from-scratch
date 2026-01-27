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
# Sidebar (Related documents)
# -------------------------------
st.sidebar.title("📄 Sources Used")
st.sidebar.info(
    "Documents used to generate the answer will appear here.\n\n"
    "This helps you understand where the information comes from."
)

# -------------------------------
# Top navigation pages
# -------------------------------
pages = ["Chat", "Knowledge Base", "Evaluation"]
selected_page = st.tabs(pages)

# -------------------------------
# Chat Page
# -------------------------------
with selected_page[0]:
    st.header("💬 Ask Your Knowledge Base")
    st.caption(
        "Ask questions and get answers based only on your uploaded documents."
    )

    with st.expander("💡 Example questions you can ask"):
        st.markdown("""
        - What does the contract say about termination?  
        """)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        st.text_area(
            key="user_input",
            placeholder="Ask a question about your documents…",
            label="User Message",
            label_visibility="collapsed",
            height=120
        )
        submitted = st.form_submit_button("Ask")

    if submitted:
        user_input = st.session_state.user_input
        for msg in st.session_state.chat_history:
            st.markdown(
                f"**You:** {user_input}"
            )
        
        
        if user_input.strip():
            st.session_state.chat_history.append(
                {"role": "user", "text": user_input}
            )

            with st.spinner("🤖 Looking through your documents…"):
                reranked_results, relevant_files, retrieving_exec_time, reranking_exec_time, total_retrieving_time = search_and_retrieve(user_input)

            if relevant_files and reranked_results:
                st.markdown("### 🔎 Retrieved Context (What the AI Reads)")

                with st.container(border=True):
                    st.caption("These are the document parts sent to the AI model to generate the answer.")

                    for i, (file_info, chunk_info) in enumerate(zip(relevant_files, reranked_results), 1):

                        score = file_info.get("rerank_score", 0)
                        source = file_info.get("title", "Unknown source")
                        text = chunk_info.get("sentence_chunk", "")
                        # Highlight best chunk
                        badge = "⭐ Best Match" if i == 1 else f"Document {i}"

                        with st.expander(f"{badge} — {source}", expanded=(i == 1)):
                            if score:
                                st.progress(min(score, 1.0))
                                st.caption(f"Relevance score: {score:.2f}")

                            st.write(text)
                st.sidebar.markdown("### 📎 Related Documents")

                for idx, doc in enumerate(relevant_files):
                    pdf_path = Path(doc["text_path"])

                    if not pdf_path.exists():
                        continue

                    label = doc["title"]

                    if idx == 0:
                        label = f"⭐ Best Source — {label}"

                    with open(pdf_path, "rb") as f:
                        st.sidebar.download_button(
                            label=label,
                            data=f,
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            key=f"pdf_download_{idx}"
                        )
                    with st.expander("⭐ Evaluate this answer", expanded=True):
                        rating = st.slider(
                            "Helpfulness",
                            min_value=1,
                            max_value=5,
                            value=3
                        )

                        feedback = st.text_area(
                            "Optional feedback",
                            placeholder="Tell us what could be better"
                        )

                        if st.button("Submit evaluation"):
                            st.toast("Thanks! Your feedback was saved 🙌", icon="✅")

                # -----------------------------------
                # ⏱ Retrieval & Reranking Metrics
                # -----------------------------------
                st.divider()

                st.markdown("#### ⏱ Retrieval Performance")

                col1, col2, col3 = st.columns(3)

                # Handle dict OR float
                col1.metric("Retrieval", f"{retrieving_exec_time.get('retrieval', 0):.2f}s")
                col2.metric("Reranking", f"{reranking_exec_time.get('reranking', 0):.2f}s")
                col3.metric("Total", f"{total_retrieving_time.get('total', 0):.2f}s")
                
                # st.markdown("### 🤖 Answer")
                # st.write(resp)

                
            else:
                st.markdown("""
                <div style="text-align:center; color:gray;">
                <h4>🔍 No matching information found</h4>
                <p>Try rephrasing your question or using simpler keywords.</p>
                </div>
                """, unsafe_allow_html=True)

            st.caption(f"⏱ Answer generated in {total_retrieving_time:.2f} seconds")

    
# -------------------------------
# Knowledge Base Page
# -------------------------------
with selected_page[1]:
    st.header("🧠 Knowledge Base")
    st.caption("Upload and manage documents used to answer questions.")
    st.info(
        "Add PDFs or files here to expand what the assistant can answer.\n\n"
        "Once uploaded, documents are automatically prepared for search."
    )

# -------------------------------
# Evaluation Page
# -------------------------------
with selected_page[2]:
    st.header("⚖️ Evaluation")
    st.caption("Review answer quality and compare different AI models.")
    st.info(
        "This page helps you evaluate how well answers perform over time.\n\n"
        "Use ratings and charts to decide which model works best for your data."
    )
