
import streamlit as st

from FlagEmbedding import FlagReranker
from config.rag_settings import (reranking_model_name, reranker_max_tokens,
                                 local_files_only)

@st.cache_resource(show_spinner="Loading the reranker...")
def load_reranker():
  reranker = FlagReranker(reranking_model_name, 
                                          use_fp16=True,
                                          local_files_only=local_files_only)
  # Warn up the re-ranker
  _ = reranker.compute_score([("What is the capital of France", "Paris is the capital of France and the largest city of the country")],
                             normalize=True)
  return reranker


def apply_reranking(results, user_query):
  reranked_results = []
  # reranker = FlagReranker(reranking_model_name, 
  #                                         use_fp16=True,
  #                                         local_files_only=local_files_only)

  reranker = load_reranker()

  """
  With reranker_max_tokens set, we cut the number of tokens set to in each doc
  to decrease computation and make the reranker works faster
  """
  pairs = [(user_query, doc["sentence_chunk"][:reranker_max_tokens]) for doc in results]
  scores = reranker.compute_score(pairs, normalize=True)
  print('\nScores: ', scores)
  for doc, score in zip(results, scores):
    doc_with_score = doc.copy()
    doc_with_score["rerank_score"] = score
    reranked_results.append(doc_with_score)

  # Sort by reranker score (higher is better)
  reranked_results = sorted(
  reranked_results,
  key=lambda x: x["rerank_score"],
  reverse=True
  )
  relevant_files = {
    doc["text_path"]: {
        "text_path": doc["text_path"],
        "title": doc["title"],
        "reranking_score": doc["rerank_score"]
    }
    for doc in reranked_results
  }.values()

  relevant_files = list(relevant_files)

  return (reranked_results, relevant_files)



