from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

def download_model(model_name:str):
    # Download the model from Hugging Face
    # model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    model = SentenceTransformer(model_name_or_path=model_name)
    # Save locally
    model.save(f"./models/embedders/{model_name}")

def download_reranker_model(model_name:str):
    reranker = FlagReranker(model_name)
    reranker.model.save_pretrained(f"./models/rerankers/{model_name}")
    reranker.tokenizer.save_pretrained(f"./models/rerankers/{model_name}")

if __name__ == "__main__":
    # download_model("BAAI/bge-large-en-v1.5")
    
    # EMBEDDING_MODEL_PATH = "BAAI/bge-reranker-v2-gemma"
    EMBEDDING_MODEL_PATH = "BAAI/bge-m3"
    download_model(EMBEDDING_MODEL_PATH)
    # Load model from local folder
    # embedder = SentenceTransformer(EMBEDDING_MODEL_PATH)
    # download_reranker_model(EMBEDDING_MODEL_PATH)