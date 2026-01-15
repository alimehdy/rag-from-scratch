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

@st.cache_resource(show_spinner="Loading LLM...")
def load_llm(model_name: str):
    # Warm the model once
    print("🧠 Loading LLM model into memory...")
    chat(
        model=model_name,
        messages=[{"role": "system", "content": "Warmup"}],
    )
    return model_name
llm_model = load_llm(llm_model_name)  # Example model name

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
st.sidebar.title("📂 PDF Retrieval")
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
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input box
    user_input = st.text_area(placeholder="Type your message:", label="")

    # Handle submission
    resp = ""
    if st.button("Send"):
        if user_input.strip():
            st.write(f"You entered: {user_input}")
            # Append user message to chat history
            st.session_state.chat_history.append({"role": "user", "text": user_input})
            with st.spinner("Processing your query..."):
                resp, exec_time = pipeline(user_input, llm_model)
            st.write(f"**Assistant:** {resp}")
            st.caption(f"⏱ {exec_time:.2f} seconds") 
            # Placeholder for AI response
            # st.session_state.chat_history.append({"role": "assistant", "text": f"Echo: {resp}"})
            # st.experimental_rerun()

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Assistant:** {msg['text']}")

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
