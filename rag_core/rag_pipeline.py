import sys
from pathlib import Path
import time
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from rag_core.src.chunker import read_json_file
# from app.embeddings import embed_chunks, save_to_faiss
from rag_core.src.retriever import search_docs_faiss, display_page, search_docs_milvus
from rag_core.src.reranker import apply_reranking
from rag_core.src.llm import build_llm_prompt, call_llm
from config.rag_settings import (embedding_model_name, json_chunks,
                                 embeddings, llm_model_name, top_k_retrieval,
                                 distance_threshold)
def pipeline(user_query):
    start_time = time.perf_counter()
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
          8. Load model in streamlit app and avoid reloading on every query
    """
    
    # all_chunks = read_json_file(json_chunks)

    # user_query = input("Enter your query: ")
    print(f"You entered: {user_query}")
    results = search_docs_milvus(user_query)
    relevant_chunks_execution_time = time.perf_counter() - start_time
    # results = search_docs_faiss(user_query, embeddings, embedding_model_name, k=top_k_retrieval, distance_threshold=distance_threshold)
    print(results)
    if results:
        print("Reranking started...")
        results = apply_reranking(results, user_query)
        reranked_results = results[0]
        relevant_files = results[1]
        reranker_execution_time = time.perf_counter() - start_time
        # reranked_indices = [[idx[0] for idx in reranked_results]]
        print("reranked_results:", reranked_results)
        # return (reranked_results, reranker_execution_time)
        print("Building LLM prompt...")
        prompt = build_llm_prompt(reranked_results, user_query)
        prompt_execution_time = time.perf_counter() - start_time
        print("Sending prompt to LLM...")
        # return (prompt, prompt_execution_time)
        response = call_llm(prompt)
        print("LLM Response:")
        print(response)
        # TODO: add to offline storage or database
        # add separate timing logs for each step
        execution_time = time.perf_counter() - start_time
        # try:
        return (response, relevant_files, execution_time)
    else:
        execution_time = time.perf_counter() - start_time
        return ("No relevant data found for your query!", None, execution_time)
    # except Exception as e:
    # return f"⚠️ Error returning response and execution time: {e}"
    # print("Displaying top result page...")
    # display_page(reranked_indices, all_chunks)

if __name__ == "__main__":
    pipeline()
