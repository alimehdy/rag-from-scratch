from app.config import hf_token
from app.loader import get_list_of_available_pdfs, open_and_read_pdf
from app.chunker import text_chunking
from app.embeddings import embed_chunks, save_to_faiss

def main():
    print("APP STARTED")
    """Main function to run the application logic."""
    print(f"Hugging Face Token exists")
    print("Program is running as a standalone script.")
    print("Loading PDFs from the specified folder...")

    # Main variables
    embedding_model_name = "BAAI/bge-large-en-v1.5"

    folder_path = "data"  # Update this path as needed
    pdf_list = get_list_of_available_pdfs(folder_path)

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

if __name__ == "__main__":    
    main()
