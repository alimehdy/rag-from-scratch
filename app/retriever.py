
import faiss

from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import numpy as np
import pymupdf

def search_docs(query: str, faiss_indexes, embedding_model_name, k:int=10, distance_threshold:float=0.75):
  # Read indexex
  index = faiss.read_index(faiss_indexes)
  embedding_model = SentenceTransformer(embedding_model_name)
  embedded_query = embedding_model.encode(query)
  query_vector = np.array(embedded_query).astype('float32').reshape(1, -1)
  distances, data_indices = index.search(query_vector, k)  # Distances and Indices
  filtered_results = [
    {
      "index": int(idx),
      "distance": float(dist)
    }
    for idx, dist in zip(data_indices[0], distances[0])
    if idx != -1 and dist <= distance_threshold
  ]
  filtered_indices = [[item['index'] for item in filtered_results]]
  filtered_scores = [[item['distance'] for item in filtered_results]]
  return (query, distances, data_indices, filtered_indices, filtered_scores)



def apply_reranking(filtered_indices, all_chunks, user_query):
  reranker = FlagReranker("BAAI/bge-reranker-v2-gemma", use_fp16=True)
  # Get retrieved docs from all_chunks
  indices = filtered_indices[0]
  filtered_docs = [
      all_chunks[int(i)]['sentence_chunk'] for i in indices
  ]

  pairs = [(user_query, doc) for doc in filtered_docs]
  scores = reranker.compute_score(pairs, normalize=True)
  print('\nScores: ', scores)
  reranked_docs = sorted(
      zip(indices, filtered_docs, scores),
      key=lambda x: x[2],
      reverse=True
  )
  # print('\nReranked Docs: ',reranked_docs)
  return reranked_docs

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
              You are a knowledgeable assistant.

              Use ONLY the information provided in the sources below to answer the question.
              If the answer cannot be found in the sources, say "I don't know".

              When answering:
              - Cite the source number(s) you used, including title and page number.
              - Do NOT add information not present in the sources.
              However, you add a new section at the end where you ask the user if he wants to elaborate a little bit more.
              If yes, elaborate a little bit emphasizing that the elaboration was out of the contexts provided.

              Sources:
              {context}

              Question:
              {user_query}

              Answer:
            """

  return prompt

def display_page(data_indices, chunks):
  for index in data_indices[0]:
    file_name = chunks[index]['text_file_name']
    chunk_content = chunks[index]['sentence_chunk']
    print(chunk_content)
    if '/content/' in file_name:
      file_name = file_name.replace('/content/', '')
    print(file_name)
    page_number = chunks[index]['page_number']-1
    print(page_number)
    pdf_path = pymupdf.open(file_name)
    page = pdf_path.load_page(page_number)
    # Set desired DPI
    dpi = 300
    zoom = dpi / 72  # convert DPI to zoom factor
    mat = pymupdf.Matrix(zoom, zoom)
    # Convert page to image (pixmap)
    pix = page.get_pixmap(matrix=mat)

    # Display in colab
    # from IPython.display import Image, display
    pix.save("page.png")
    # display(Image(filename="page.png"))