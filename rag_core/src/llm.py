from ollama import chat
from config.rag_settings import system_prompt, temperature, max_tokens, llm_streaming

def build_llm_prompt(reranked_chunks, all_chunks, user_query):
  context_blocks = []
  context_chunks = [chunk[1] for chunk in reranked_chunks]
  context_chunks_indices = [chunk[0] for chunk in reranked_chunks]
  context_chunks_metadata = [
    {
        "index": i,
        "title": all_chunks[i].get("title"),
        "file": all_chunks[i].get("text_file_name"),
        "page": all_chunks[i].get("page_number"),
        "chunk_content": all_chunks[i].get("sentence_chunk")
    }
    for i in context_chunks_indices
  ]

  for idx, item in enumerate(context_chunks_metadata):
    block = f"""
    Title: {item['title']}
    File: {item['file']}
    Page: {item['page']}
    Content:
    {item['chunk_content']}
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

def call_llm(user_prompt:str, model:str) -> str:
    
    response = chat(
        model=model,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options = {
            "stream": llm_streaming,
            "temperature": temperature,
            "max_tokens": max_tokens
        },        
        
    )
    try: 
      return response["message"]["content"] 
    except Exception as e: 
      print("⚠️ Unexpected response format:", response) 
      return f"⚠️ Unexpected response format: {response}"

# if __name__ == "__main__":
#     user_prompt = input("Enter your prompt: ")
#     system_prompt = "You are a helpful assistant that provides accurate information based on the provided context."
#     response = call_llm(user_prompt)
#     print(response)