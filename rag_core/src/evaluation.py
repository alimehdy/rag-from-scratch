import ragas
from db_handler import get_data_for_evaluation
# https://docs.ragas.io/en/stable/getstarted/
# https://www.leoniemonigatti.com/blog/rag-evaluation-with-ragas.html

def evaluate_response():
    try:
        # Get data from the database for evaluation
        data = get_data_for_evaluation()
        print(data)
        return True
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return f"Error during evaluation: {e}"

if __name__ == "__main__":
    evaluate_response()