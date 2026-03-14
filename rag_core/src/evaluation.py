from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    context_precision,
    context_recall
)
from rag_core.src.db_handler import get_data_for_evaluation
from rag_core.src.utils import tuple_to_dict
# https://docs.ragas.io/en/stable/getstarted/
# https://www.leoniemonigatti.com/blog/rag-evaluation-with-ragas.html

def evaluate_response():
    try:
        # Get data from the database for evaluation
        evaluation_data = []
        result = get_data_for_evaluation()
        # Transform the data into the format expected by RAGAS: list of dicts
        evaluation_data = tuple_to_dict(result, column_names=["trace_uuid", "user_query_hash", "created_at", "user_prompt", "llm_response"])
        print("Evaluation Data:", evaluation_data[0])
        result = evaluate(
            evaluation_data,
            metrics=[
                faithfulness,
                context_precision,
                context_recall
            ]
        )
        print("Evaluation Results:", result)
        return result
    except Exception as e:
        return f"Error during evaluation: {e}"

if __name__ == "__main__":
    evaluate_response()