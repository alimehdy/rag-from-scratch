import os
from chunker import text_chunking
from loader import get_list_of_available_pdfs, open_and_read_pdf
from sentence_transformers import SentenceTransformer
# from huggingface_hub import login
import json
from tqdm import tqdm
from typing import Optional
import faiss
import numpy as np

def embed_chunks(chunks: list, embedding_model, metadata_json_path, hf_token:Optional[str]=None):
  # Use this only if we're getting the embedder online
  # login(token=hf_token)
  embedding_model = SentenceTransformer(model_name_or_path = embedding_model)
  embedding_model.to("cpu")
  print(f"Number of chunks to embed: {len(chunks)}")
  i = 1
  for item in tqdm(chunks):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])
    i = i-1
  # save to json to recover instead of re-embedding
  chunks_array = [x['embedding'].tolist() for x in chunks]
  # with open(metadata_json_path + '/' + 'embedded_chunks.json', 'w') as f:
  #   json.dump(chunks_array, f)
  return chunks

def save_to_faiss(embeddings: list,
                  embed_index_path: str,
                  save_to_local: bool=False,
                  distance_metric: Optional[str]='L2',
                  file_name: str = "embeddings.index"):
  # transform to float32
  matrix = np.array(embeddings).astype('float32')
  num_vectors, dimension = matrix.shape
  print(dimension)
  match distance_metric:
    case 'L2':
      index = faiss.IndexFlatL2(dimension)
    case 'dot':
      index = faiss.IndexFlatIP(dimension)
    case _:
      raise ValueError("Distance metric must be 'L2' or 'dot'")

  # Add vectors to index
  index.add(matrix)
  if save_to_local:
    faiss.write_index(index, embed_index_path + '/' +file_name)

  return {
        "status": "success",
        "num_vectors": num_vectors,
        "dimension": dimension,
        "metric_type": distance_metric,
        "index_type": type(index).__name__,
        "saved_to": file_name,
    }

if __name__ == "__main__":
  json_path = "./rag_core/embeddings/" # Update this path as needed
  index_path = "./rag_core/embeddings/" # Update this path as needed
  embedding_model_name = "./rag_core/models/embedders/BAAI/bge-large-en-v1.5" # Update this path as needed
  folder_path = "./rag_core/data"  # Update this path as needed
  pdf_list = get_list_of_available_pdfs(folder_path)
  print(f"Found {len(pdf_list)} PDF files.")
  pdf_pages = open_and_read_pdf(pdf_list, json_path)
  
  print(f"Extracted {len(pdf_pages)} pages from the PDFs.")
  # print(pdf_pages[0])
  print("Chunking process started...")
  all_chunks = text_chunking(pdf_pages)
  print(f"Generated {len(all_chunks)} text chunks from the pages.")
  print("Embedding process started...")
  res = embed_chunks(all_chunks, embedding_model_name, metadata_json_path=json_path)
  print("Saving embeddings to FAISS index...")
  faiss_info = save_to_faiss(
      embeddings=[item["embedding"] for item in res],
      embed_index_path=index_path,
      save_to_local=True,
      distance_metric='L2',
      file_name="embeddings.index"
  )