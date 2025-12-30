
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
    pix.save(f"media/result_rank_{index}_page_{page_number}.png")
    # display(Image(filename="page.png"))