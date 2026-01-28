from ollama import chat
from config.rag_settings import system_prompt, temperature, max_tokens, llm_streaming, llm_model_name
# import streamlit as st

# @st.cache_resource(show_spinner="Loading LLM...")
# def load_llm(model_name: str):
#     # Warm the model once
#     print("🧠 Loading LLM model into memory...")
#     chat(
#         model=model_name,
#         messages=[{"role": "system", "content": "Warmup"}],
#     )
#     return model_name


def build_llm_prompt(reranked_chunks, user_query):
  context_blocks = []

  for idx, item in enumerate(reranked_chunks):
    block = f"""
    Title: {item['title']}
    Page: {item['page']}
    File Path: {item['text_path']}
    Rerank Score: {item['rerank_score']}
    Content:
    {item['sentence_chunk']}
    """
    context_blocks.append(block.strip())

  context = "\n\n---\n\n".join(context_blocks)
  prompt = f"""
              Sources:
              {context}

              Question:
              {user_query}

              Answer:
            """

  return prompt

def call_llm(user_prompt:str) -> str:
    stream = chat(
        model=llm_model_name,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options = {
            "stream": llm_streaming,
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        stream=True        
        
    )
    try: 
      for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]
      # only if streaming is False
      # return response["message"]["content"] 
    except Exception as e: 
      yield f"\n⚠️ Unexpected error: {str(e)}"
      # only if streaming is False
      # print("⚠️ Unexpected response format:", response) 
      # return f"⚠️ Unexpected response format: {response}"

# if __name__ == "__main__":
#     user_prompt = input("Enter your prompt: ")
#     system_prompt = "You are a helpful assistant that provides accurate information based on the provided context."
#     response = call_llm(user_prompt)
#     print(response)