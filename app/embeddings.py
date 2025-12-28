from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import json
from tqdm import tqdm
from typing import Optional
import faiss
import numpy as np

def embed_chunks(chunks: list, embedding_model, hf_token):
  login(token=hf_token)
  embedding_model = SentenceTransformer(model_name_or_path = embedding_model, device="cpu")
  embedding_model.to("cpu")
  print(f"Number of chunks to embed: {len(chunks)}")
  i = 1
  for item in tqdm(chunks):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])
    i = i-1
  # save to json to recover instead of re-embedding
  chunks_array = [x['embedding'].tolist() for x in chunks]
  with open('embedded_chunks.json', 'w') as f:
    json.dump(chunks_array, f)

  return chunks



def save_to_faiss(embeddings: list,
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
    faiss.write_index(index, file_name)

  return {
        "status": "success",
        "num_vectors": num_vectors,
        "dimension": dimension,
        "metric_type": distance_metric,
        "index_type": type(index).__name__,
        "saved_to": file_name,
    }