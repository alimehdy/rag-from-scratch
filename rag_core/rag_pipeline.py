import time
from datetime import datetime
from typing import Optional
# from app.embeddings import embed_chunks, save_to_faiss
from rag_core.src.retriever import search_docs_milvus
from rag_core.src.reranker import apply_reranking
from rag_core.src.llm import build_llm_prompt, call_llm_with_stream
from rag_core.src.utils import generate_uuid_and_hash, get_device
from rag_core.src.db_handler import insert_rag_trace

"""
    TODO: 1. Make the paths configurable via a config file or command line arguments --> Done
          2. Add error handling for file operations and user inputs
          3. Modularize the code further if needed --> Done
          4. Add logging instead of print statements for better traceability
          5. Switch retrieving to read the chunk_id from the metadata json instead of all_chunks
          6. Add AI Judge or RAG evaluation pipeline
          7. Save evaluation results to a file or local database
          8. Load model in streamlit app and avoid reloading on every query
          9. Turn this into a CLI tool
          10. Package it with Docker
          11. Deploy it so Streamlit + RAG run like a service
"""

def search_and_retrieve(user_query):
    _uuid, timestamp, user_query_hash = generate_uuid_and_hash(user_query)
    device = get_device()
    start_time = time.perf_counter()
    execution_time = {
        "retrieving_time": None,
        "reranking_time": None,
        "llm_executing_time": None,
    }
    reranked_results = None
    relevant_files = None
    response = ""
    try:
        print("APP STARTED")
        # Main variables
        # all_chunks = read_json_file(json_chunks)
        # user_query = input("Enter your query: ")
        print(f"You entered: {user_query}")
        results = search_docs_milvus(user_query)
        execution_time["retrieving_time"] = time.perf_counter() - start_time
        # results = search_docs_faiss(user_query, embeddings, embedding_model_name, k=top_k_retrieval, distance_threshold=distance_threshold)
        print(results)
        if results:
            print("Reranking started...")
            reranking_start_time = time.perf_counter()
            results = apply_reranking(results, user_query)
            reranked_results = results[0]
            relevant_files = results[1]
            execution_time["reranking_time"] = time.perf_counter() - reranking_start_time
            # reranked_indices = [[idx[0] for idx in reranked_results]]
            print("reranked_results:", reranked_results)
            print("relevant_files:", relevant_files)
            # return (reranked_results, relevant_files, execution_time)
            print("Building LLM prompt...")
            llm_start_time = time.perf_counter()
            if reranked_results and relevant_files:
                prompt = build_llm_prompt(reranked_results, user_query)
                print("Sending prompt to LLM...")
                response_generator = call_llm_with_stream(prompt, execution_time)
                for chunk in response_generator:
                    response += chunk
                print("LLM Response:")
                print(response)
                return (response, reranked_results, relevant_files, execution_time, _uuid, user_query_hash, timestamp)
            else:
                return ("No relevant data found for your query!", None, None, execution_time, _uuid, user_query_hash, timestamp)
        else:
            execution_time["retrieving_time"] = time.perf_counter() - start_time
            execution_time["reranking_time"] = None
            execution_time["llm_executing_time"] = None
            return ("No relevant data found for your query!", None, None, execution_time, _uuid, user_query_hash, timestamp)
    except Exception as e:
        return (f"⚠️ Error returning response and execution time: {e}", None, None, None, _uuid, user_query_hash, timestamp)
    finally:
        execution_time["total_time"] = time.perf_counter() - start_time
        # Collect all trace data and persist it for evaluation
        print(f"uuid: {_uuid}, user query: {user_query_hash}, time: {timestamp}, response: {response}, reranked_docs: {reranked_results}, timing info: {execution_time}, device: {device}")
        print("\n")
        insert_rag_trace(
                    _uuid=_uuid,
                    user_query_hash=user_query_hash,
                    timestamp=timestamp,
                    user_prompt=user_query, llm_response=response,
                    retrieved_docs=None, reranked_docs=reranked_results,
                    timing_info=execution_time,
                    device=device
                )
    print("Displaying top result page...")
    display_page(reranked_indices, all_chunks)

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    search_and_retrieve(user_query)

