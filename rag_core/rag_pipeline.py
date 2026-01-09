import json
# from app.config import hf_token
# from app.loader import get_list_of_available_pdfs, open_and_read_pdf
# from app.chunker import text_chunking, read_json_file
# from app.embeddings import embed_chunks, save_to_faiss
from app.retriever import search_docs, apply_reranking, display_page
from app.llm import build_llm_prompt, call_llm
import os
def pipeline():
    print("APP STARTED")
    # Main variables
    """TODO: avoid saving large chunks metadata in json and move each array
    of metadata into the chunk itself."""
    
    # TODO: make the model name used all across and when changed only change in one place
    embedding_model_name = "./rag_core/models/embedders/BAAI/bge-large-en-v1.5"
    folder_path = "./rag_core/data"  # Update this path as needed
    json_chunks = "./rag_core/embeddings/pdf_pages.json"  # Update this path as needed
    all_chunks = []
    with open(json_chunks, "r") as f:
        all_chunks = json.load(f)
    user_query = input("Enter your query: ")
    print(f"You entered: {user_query}")
    results = search_docs(user_query, "./rag_core/embeddings/embeddings.index", embedding_model_name, k=10, distance_threshold=0.75)
    
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
    response = call_llm(prompt)
    print("LLM Response:")
    print(response)
    # print("Displaying top result page...")
    # display_page(reranked_indices, all_chunks)

if __name__ == "__main__":
    pipeline()
