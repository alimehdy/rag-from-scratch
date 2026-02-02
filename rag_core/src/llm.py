import time
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

def call_llm_with_stream(user_prompt:str, execution_time: dict) -> any:
    start_time = time.perf_counter()
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
    except Exception as e: 
      yield f"\n⚠️ Unexpected error: {str(e)}"
    finally:
      execution_time["llm_executing_time"] = time.perf_counter() - start_time
    #   yield {"__end__": llm_exec_time}

def call_llm_no_stream(user_prompt:str) -> str:
    start_time = time.perf_counter()
    response = chat(
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
        stream=False        
        
    )
    llm_exec_time = time.perf_counter() - start_time
    try: 
      return (response["message"]["content"] , llm_exec_time)
    except Exception as e: 
      return (f"⚠️ Unexpected error: {str(e)}", llm_exec_time)

# if __name__ == "__main__":
#     user_prompt = input("Enter your prompt: ")
#     system_prompt = "You are a helpful assistant that provides accurate information based on the provided context."
#     response = call_llm(user_prompt)
#     print(response)