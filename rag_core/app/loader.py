import json
import os
import re
import pymupdf
import pandas as pd
from tqdm import tqdm

# Reading locally
def get_list_of_available_pdfs(folder_path, from_drive:bool=False) -> list:
  if '/content' not in folder_path and not from_drive:
    folder_path = folder_path
  if from_drive:
    folder_path = '/content/drive/' + folder_path
  pdf_list = []
  for filename in os.listdir(folder_path):
    if filename.endswith('.pdf'):
      pdf_list.append(os.path.join(folder_path, filename))
  return pdf_list


# Text formatting
def text_formatter(text: str) -> str:
  """Perform minor formatting on text"""
  # if the word is broken between two lines
  cleaned_text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
  cleaned_text = cleaned_text.replace("\n", " ").strip()
  # perform additional formatting
  return cleaned_text

# reading file's pages and augmenting with additional metadata
def open_and_read_pdf(pdf_list: list, json_path: str) -> list[dict]:
  pdf_pages = []
  for file in pdf_list:
    doc = pymupdf.open(file)
    for page_number, page in tqdm(enumerate(doc)):
      text = page.get_text()
      text = text_formatter(text)

      # Fill blank pages and full image covered page with default text
      if not text.strip():
        text = 'Blank Page - No Text'
      pdf_pages.append({
      "page_number": page_number + 1,
      "page_char_count": len(text),
      "page_word_count": len(text.split(" ")),
      "page_sentence_count_raw": len(text.split(". ")),
      "page_token_count": len(text)/4,
      "text": text,
      "text_path": file,
      "text_file_name": doc.name,
      'format': doc.metadata.get('format'),
      'title': doc.metadata.get('title'),
      'author': doc.metadata.get('author'),
      'subject': doc.metadata.get('subject'),
      'keywords': doc.metadata.get('keywords'),
      'creator': doc.metadata.get('creator'),
      'producer': doc.metadata.get('producer'),
      'creationDate': doc.metadata.get('creationDate'),
      'modDate': doc.metadata.get('modDate'),
      'trapped': doc.metadata.get('trapped'),
      'encryption': doc.metadata.get('encryption')
      })

    with open(json_path + '/pdf_pages.json', 'w') as f:
      json.dump(pdf_pages, f)
  return pdf_pages

def pdf_stats(content:list) -> pd.DataFrame:
  df = pd.DataFrame(content)
  return df.describe().round(2)