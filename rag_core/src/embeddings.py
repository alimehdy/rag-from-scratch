import sys
from pathlib import Path
from xmlrpc import client

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from chunker import text_chunking
from loader import get_list_of_available_pdfs, open_and_read_pdf
from sentence_transformers import SentenceTransformer
# from huggingface_hub import login
from pymilvus import (client, Collection, 
                      FieldSchema, CollectionSchema, DataType, 
                      MilvusClient)
from tqdm import tqdm
from typing import Optional
import faiss
import numpy as np
from config.rag_settings import (embedding_model_name, folder_path, 
     metadata_json_path, index_path, distance_metric, embeddings_file_name,
     milvus_host, milvus_port, milvus_collection_name, 
     milvus_embedding_dim, milvus_distance_metric, milvus_client_uri)



def embed_chunks(chunks: list, embedding_model, metadata_json_path, hf_token:Optional[str]=None):
  # Use this only if we're getting the embedder online
  # login(token=hf_token)
  embedding_model = SentenceTransformer(model_name_or_path = embedding_model_name)
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

def save_to_milvus(embeddings: list,
                  collection_name:str=milvus_collection_name,
                  chunks_metadata: list=[],
                  host:str=milvus_host,
                  port:str=milvus_port,
                  distance_metric:str=milvus_distance_metric,
                  dimension:int=milvus_embedding_dim
                  ):
  # Milvus client
  client = MilvusClient(uri=milvus_client_uri)
  """or connections.connect() if we want to connect to a server 
  (installing milvus server using docker is needed)
  """
  # create schema
  milvus_schema = CollectionSchema(fields=[
      FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
      FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
      FieldSchema(name="chunk_metadata", dtype=DataType.JSON)
  ], description="RAG Embeddings Collection")
  # Create collection
  if not Collection.exists(collection_name):
      print(f"Collection {collection_name} does not exists. Creating...")
      collection = Collection(name=collection_name, schema=milvus_schema)
  else:
      print(f"Collection {collection_name} exists. Using existing collection.")
      collection = Collection(name=collection_name, schema=milvus_schema)
  insert_data = [
      embeddings,
      [chunk['metadata'] for chunk in chunks_metadata]
  ]
  # Insert data into collection
  collection.insert(insert_data)
  # Build index for faster search
  index_params = {
      "index_type": "IVF_FLAT",
      "metric_type": distance_metric,
      "params": {"nlist": dimension}
  }
  collection.create_index(field_name="embedding", index_params=index_params)

  # Load collection to memory
  collection.load()
  return {
        "status": "success",
        "num_vectors": len(embeddings),
        "dimension": dimension,
        "metric_type": distance_metric,
        "collection_name": collection_name,
  }

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
  
  pdf_list = get_list_of_available_pdfs(folder_path)
  print(f"Found {len(pdf_list)} PDF files.")
  pdf_pages = open_and_read_pdf(pdf_list, metadata_json_path)
  
  print(f"Extracted {len(pdf_pages)} pages from the PDFs.")
  # print(pdf_pages[0])
  print("Chunking process started...")
  all_chunks = text_chunking(pdf_pages)
  print(f"Generated {len(all_chunks)} text chunks from the pages.")
  print("Embedding process started...")
  res = embed_chunks(all_chunks, embedding_model_name, metadata_json_path=metadata_json_path)
  
  # Saving to Milvus
  print("Saving embeddings to Milvus...")
  milvus_info = save_to_milvus(embeddings=[item["embedding"] for item in res],
                              chunks_metadata=all_chunks)
  print("Milvus save info:", milvus_info)
  # print("Saving embeddings to FAISS index...")
  # faiss_info = save_to_faiss(
  #     embeddings=[item["embedding"] for item in res],
  #     embed_index_path=index_path,
  #     save_to_local=True,
  #     distance_metric=distance_metric,
  #     file_name=embeddings_file_name
  # )