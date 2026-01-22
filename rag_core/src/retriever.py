
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pymupdf
from config.rag_settings import (embedding_model_name, top_k_retrieval,
                                 milvus_collection_name)
from pymilvus import utility, Collection
from rag_core.src.embeddings import connect_to_milvus
def search_docs_milvus(query: str, milvus_collection_name:str=milvus_collection_name,
                       embedding_model_name:str=embedding_model_name, k:int=top_k_retrieval, distance_threshold:float=0.75):
  # Connect to Milvus
  if connect_to_milvus():
    if utility.has_collection(milvus_collection_name):
      collection = Collection(milvus_collection_name)
      embedding_model = SentenceTransformer(embedding_model_name)
      embedded_query = embedding_model.encode(query)
      query_vector = [embedded_query.astype('float32').tolist()]
      search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
      }
      results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param=search_params,
        limit=k,
        output_fields=["chunk_metadata"]
      )

      filtered_results = []
      for hit in results[0]:
        if hit.distance <= distance_threshold:
          filtered_results.append({
              "id": hit.id,
              "distance": hit.distance,
              "sentence_chunk": hit.entity.get("chunk_metadata")["sentence_chunk"],
              "text_file_name": hit.entity.get("chunk_metadata")["text_file_name"],
              "text_path": hit.entity.get("chunk_metadata")["text_path"],
              "title": hit.entity.get("chunk_metadata")["title"],
              "page": hit.entity.get("chunk_metadata")["page_number"]
          })
      return filtered_results
    else:
      raise ValueError(f"Collection {milvus_collection_name} does not exist in Milvus.")
  else: 
    raise ConnectionError("Could not connect to Milvus.")
  

def search_docs_faiss(query: str, faiss_indexes, embedding_model_name, k:int=10, distance_threshold:float=0.75):
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