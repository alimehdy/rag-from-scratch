from rag_core.app.config import hf_token
from rag_core.app.loader import get_list_of_available_pdfs, open_and_read_pdf
from rag_core.app.chunker import text_chunking, read_json_file
from rag_core.app.embeddings import embed_chunks, save_to_faiss
from rag_core.app.retriever import search_docs, apply_reranking, display_page
from rag_core.app.llm import build_llm_prompt, call_llm
import os
def pipeline():
    print("APP STARTED")
    # Main variables
    # TODO: move the embedding methods to embeddings.py
    # This part must be clean and directly starts with the retrieval-augmented generation pipeline
    # If index file does not exist ask the users if they want to embedd their data
    embedding_model_name = "./rag_core/models/embedders/BAAI/bge-large-en-v1.5"
    folder_path = "./rag_core/data"  # Update this path as needed
    pdf_list = get_list_of_available_pdfs(folder_path)
    all_chunks = []

    if "embeddings.index" not in os.listdir("./rag_core/embeddings/"):
        print("Embeddings index already exists. Exiting to avoid reprocessing.")
        """Main function to run the application logic."""
        print(f"Hugging Face Token exists")
        print("Program is running as a standalone script.")
        print("Loading PDFs from the specified folder...")

        if pdf_list:
            print(f"Found {len(pdf_list)} PDF files.")
            pdf_pages = open_and_read_pdf(pdf_list)
            print(f"Extracted {len(pdf_pages)} pages from the PDFs.")
            # print(pdf_pages[0])
            print("Chunking process started...")
            all_chunks = text_chunking(pdf_pages)
            print(f"Generated {len(all_chunks)} text chunks from the pages.")
            print("Embedding process started...")
            res = embed_chunks(all_chunks, embedding_model_name, hf_token)
            print("Saving embeddings to FAISS index...")
            faiss_info = save_to_faiss(
                embeddings=[item["embedding"] for item in res],
                save_to_local=True,
                distance_metric='L2',
                file_name="embeddings.index"
            )
        else:
            print("No PDF files found.")
            return
    if not all_chunks:
        pdf_pages = open_and_read_pdf(pdf_list)
        all_chunks = text_chunking(pdf_pages)
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
