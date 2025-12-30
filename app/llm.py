from ollama import chat


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

def call_llm(user_prompt:str, model:str="phi3") -> str:
    system_prompt = f"""
    You are a knowledgeable assistant.
    Use ONLY the information provided in the sources below to answer the question.
    If the answer cannot be found in the sources, say "I don't know".
    When answering:
    - Cite the source number(s) you used, including title (file name) and page number.
    - Do NOT add information not present in the sources.
    However, you add a new section at the end where you ask the user if he wants to elaborate a little bit more.
    If yes, elaborate a little bit emphasizing that the elaboration was out of the contexts provided.
    """
    response = chat(
        model=model,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options = {
            "stream": False,
            "temperature":0.7,
            "max_tokens":1000
        },        
        
    )
    return response["message"]["content"]

# if __name__ == "__main__":
#     user_prompt = input("Enter your prompt: ")
#     system_prompt = "You are a helpful assistant that provides accurate information based on the provided context."
#     response = call_llm(user_prompt)
#     print(response)