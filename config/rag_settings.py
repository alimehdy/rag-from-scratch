# import os
# from dotenv import load_dotenv

# load_dotenv()  # Load environment variables from .env file
# hf_token = os.getenv("hf_token")

# if not hf_token:
#     raise ValueError("Hugging Face token not found in environment variables.")

# Database settings
folder_path = "./rag_core/data/"  # Path to the folder containing PDFs
# Embedding settings
embedding_model_name = "./rag_core/models/embedders/BAAI/bge-large-en-v1.5"
json_chunks = "./rag_core/embeddings/chunks_metadata.json"  # Update this path as needed
embeddings = "./rag_core/embeddings/embeddings.index"
embeddings_file_name = "embeddings.index"
embedder_path = "./rag_core/models/embedders/"
metadata_json_path = "./rag_core/embeddings/"
index_path = "./rag_core/embeddings/" # Update this path as needed
distance_metric = "L2"  # Options: 'L2' or 'dot'

# Chunking settings
MARKDOWN_SEPARATORS = [
      "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", ". ", " ", ""
  ]
chunk_size = 1000  # Number of characters per chunk
chunk_overlap = 50  # Number of overlapping characters between chunks
metadata_path = "./rag_core/embeddings/"
metadata_file_name = "chunks_metadata.json"
# LLM settings
llm_model_name = "qwen2.5:3b"  # Update this with your desired LLM model name
temperature = 0.7
max_tokens = 1000
llm_streaming = False
system_prompt = f"""
    You are a knowledgeable assistant.
    Use ONLY the information provided in the sources below to answer the question.
    If the answer cannot be found in the sources, say "I don't know".
    When answering:
    - It is essesntial that you cite the source number(s) you used, including title (file name) and page number.
    - Do NOT add information not present in the sources.
    """

# Retrieval settings
top_k_retrieval = 10
distance_threshold = 0.75

# Reranker settings
reranking_model_name = "./rag_core/models/rerankers/BAAI/bge-reranker-v2-gemma"  # Update this with your desired reranking model name
local_files_only = True
reranker_path = "./rag_core/models/rerankers/"
reranker_max_tokens = 256

# Milvus settings
milvus_host = "127.0.0.1" # or "localhost"
milvus_port = "19530"
milvus_collection_name = "rag_collection"
milvus_distance_metric = "L2" # Options: 'L2', 'IP'
milvus_client_uri = "./milvus_db.db"
milvus_embedding_dim = 1024  # Update this based on the embedding model used
nlist = 128 # It should be sqrt of the number of embeddings, but to save memory we set it lower for now.
