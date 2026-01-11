import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from src.chunker import read_json_file
# from app.embeddings import embed_chunks, save_to_faiss
from src.retriever import search_docs, apply_reranking, display_page
from src.llm import build_llm_prompt, call_llm
from config.rag_settings import (embedding_model_name, json_chunks,
                                 embeddings, llm_model_name, top_k_retrieval,
                                 distance_threshold)
def pipeline():
    print("APP STARTED")
    # Main variables
    """
    TODO: 1. Make the paths configurable via a config file or command line arguments
          2. Add error handling for file operations and user inputs
          3. Modularize the code further if needed
          4. Add logging instead of print statements for better traceability
          5. Switch retrieving to read the chunk_id from the metadata json instead of all_chunks
          6. Add AI Judge or RAG evaluation pipeline
          7. Save evaluation results to a file or local database
    """
    
    all_chunks = read_json_file(json_chunks)

    user_query = input("Enter your query: ")
    print(f"You entered: {user_query}")
    results = search_docs(user_query, embeddings, embedding_model_name, k=top_k_retrieval, distance_threshold=distance_threshold)
    
    # Get filtered indices
    print("Filtered Indices:")
    filtered_indices = results[3]
    print("filtered_indices:", filtered_indices)
    print("Reranking started...")
    reranked_results = apply_reranking(filtered_indices, all_chunks, user_query)
    reranked_indices = [[idx[0] for idx in reranked_results]]
    print("reranked_results:", reranked_indices)
    print("Building LLM prompt...")
    prompt = build_llm_prompt(reranked_results, all_chunks, user_query)
    print("Sending prompt to LLM...")
    response = call_llm(prompt, model=llm_model_name)
    print("LLM Response:")
    print(response)
    # print("Displaying top result page...")
    # display_page(reranked_indices, all_chunks)

if __name__ == "__main__":
    pipeline()
