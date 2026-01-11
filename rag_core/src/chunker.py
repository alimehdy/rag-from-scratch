import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import json

from config.rag_settings import MARKDOWN_SEPARATORS, chunk_size, chunk_overlap, metadata_path, metadata_file_name
def text_chunking(content: list) -> list:
  # saving the original list
  all_chunks = []
  text_splitter = RecursiveCharacterTextSplitter(
      separators=MARKDOWN_SEPARATORS, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
      add_start_index=True, strip_whitespace=True
  )

  chunk_index = 0
  for item in tqdm(content):
    chunks = text_splitter.split_text(item["text"])
    # Build structured chunk dicts
    chunk_dicts = []
    for chunk in chunks:
        chunk_dict = {
            "chunk_index": chunk_index,
            "page_number": item["page_number"],
            "title": item["title"],
            "author": item["author"],
            "subject": item["subject"],
            "creatror": item["creator"],
            "text_path": item["text_path"],
            "text_file_name": item["text_file_name"],
            "creationDate": item["creationDate"],
            "modDate": item["modDate"],
            "sentence_chunk": chunk,
            "chunk_char_count": len(chunk),
            "chunk_word_count": len(chunk.split(" ")),
            "chunk_token_count": len(chunk) / 4
        }
        chunk_dicts.append(chunk_dict)
        chunk_index += 1

    # Save inside each item (optional)
    item["chunk_array"] = chunks
    item["chunk"] = chunk_dicts

    # We can filter the results and remove small chunks
    # Append to global list
    all_chunks.extend(chunk_dicts)
    with open(metadata_path + metadata_file_name, 'w') as f:
        json.dump(all_chunks, f)
  return all_chunks

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data