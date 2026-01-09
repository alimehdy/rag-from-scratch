import streamlit as st
import time
# from rag_core.rag_pipeline import pipeline

st.set_page_config(page_title="RAG Assistant", layout="wide")

# ---------- Session State ----------
if "history" not in st.session_state:
    st.session_state.history = []

if "chat" not in st.session_state:
    st.session_state.chat = []

if "selected_query" not in st.session_state:
    st.session_state.selected_query = None

# ---------- Layout ----------
left, center, _ = st.columns([1, 3, 0.2])

# ---------- Left: Search History ----------
with left:
    st.subheader("🧭 History")

    for i, q in enumerate(reversed(st.session_state.history)):
        if st.button(q, key=f"hist_{i}"):
            st.session_state.selected_query = q

# ---------- Center: Chat ----------
with center:
    st.subheader("💬 Conversation")

    for item in st.session_state.chat:
        st.markdown(f"**🧑 You:** {item['question']}")

        answer_box = st.empty()
        streamed = ""

        for token in item["answer"].split():
            streamed += token + " "
            answer_box.markdown(f"**🤖 Assistant:** {streamed}")
            time.sleep(0.02)

        with st.expander("📚 Sources"):
            for src in item["sources"]:
                st.markdown(f"**{src['title']}**")
                st.caption(src['chunk'])

        col1, col2 = st.columns(2)
        with col1:
            st.button("👍 Helpful", key=f"up_{item['id']}")
        with col2:
            st.button("👎 Not helpful", key=f"down_{item['id']}")

# ---------- Bottom Input ----------
st.markdown("---")
query = st.chat_input("Ask something...")
query

# ---------- Handle New Input ----------
if query:
    st.session_state.history.append(query)

#     with st.spinner("Thinking..."):
#         answer, sources = pipeline(query)

#     st.session_state.chat.append({
#         "id": len(st.session_state.chat),
#         "question": query,
#         "answer": answer,
#         "sources": sources
#     })

#     st.experimental_rerun()

# # ---------- Replay from History ----------
# if st.session_state.selected_query:
#     q = st.session_state.selected_query
#     st.session_state.selected_query = None

#     with st.spinner("Re-running query..."):
#         answer, sources = pipeline(q)

#     st.session_state.chat.append({
#         "id": len(st.session_state.chat),
#         "question": q,
#         "answer": answer,
#         "sources": sources
#     })

#     st.experimental_rerun()
