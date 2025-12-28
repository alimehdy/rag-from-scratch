"""
That can be done using two ways:
1. Splitting on ". "
2. Using a library (spaCy, nltk, etc.)
3. Using recusrsive chunking
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
def text_chunking(content: list) -> list:
  # saving the original list
  all_chunks = []
  # Separators
  MARKDOWN_SEPARATORS = [
      "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", ". ", " ", ""
  ]

  text_splitter = RecursiveCharacterTextSplitter(
      separators=MARKDOWN_SEPARATORS, chunk_size=1000, chunk_overlap=50,
      add_start_index=True, strip_whitespace=True
  )


  for item in tqdm(content):
    chunks = text_splitter.split_text(item["text"])

    # Build structured chunk dicts
    chunk_dicts = []
    for chunk in chunks:
        chunk_dict = {
            "page_number": item["page_number"],
            "title": item["title"],
            "text_file_name": item["text_file_name"],
            "sentence_chunk": chunk,
            "chunk_char_count": len(chunk),
            "chunk_word_count": len(chunk.split(" ")),
            "chunk_token_count": len(chunk) / 4
        }
        chunk_dicts.append(chunk_dict)

    # Save inside each item (optional)
    item["chunk_array"] = chunks
    item["chunk"] = chunk_dicts

    # We can filter the results and remove small chunks
    # Append to global list
    all_chunks.extend(chunk_dicts)
  return all_chunks