from FlagEmbedding import FlagReranker
from config.rag_settings import (reranking_model_name, 
                                 local_files_only)

def apply_reranking(results, user_query):
  reranked_results = []
  reranker = FlagReranker(reranking_model_name, 
                                          use_fp16=True,
                                          local_files_only=local_files_only)

  pairs = [(user_query, doc["sentence_chunk"]) for doc in results]
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
  return reranked_results



