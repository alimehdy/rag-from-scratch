import psycopg2
import json
from datetime import datetime
from config.rag_settings import (llm_model_name, embedding_model_name, distance_metric,
                                 llm_streaming,
                                 temperature, max_tokens, system_prompt,
                                 chunk_size, chunk_overlap, top_k_retrieval,
                                 reranking_model_name)



# -------------------------------
# DB connection parameters
# -------------------------------
DB_HOST = "localhost"     # or docker host
DB_PORT = 5431
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "abc123"

# -------------------------------
# Function to insert a dummy record
# -------------------------------

def establish_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("Database connection established successfully.")
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def insert_user_feedback(trace_uuid, user_query_hash, timestamp, feedback_on, feedback_rating, feedback_value):
    try:
        conn = establish_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO rag_schema.user_feedbacks (
                            trace_uuid, user_query_hash, created_at,
                            feedback_on, feedback_rating, feedback_value)
                        VALUES (%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            trace_uuid,
                            user_query_hash,
                            timestamp,
                            feedback_on,
                            feedback_rating,
                            feedback_value
                        )
                    )
            res = conn.commit()
            print(f"User feedback insert result: {res}")
            cur.close()
            conn.close()
            return True
    except Exception as e:
        conn.rollback()
        print(f"Error inserting user feedback: {e}")
        return False

def insert_rag_trace(_uuid, user_query_hash, timestamp, user_prompt, llm_response,
                     retrieved_docs, reranked_docs, timing_info, device
                     ):
    try:
        conn = establish_db_connection()
        if conn:
            llm_model_info = {
                "model_name": llm_model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "streaming": llm_streaming
            }
            embedding_model_info = {
                "model_name": embedding_model_name,
                "distance_metric": distance_metric,
            }
            reranking_model_info = {
                "model_name": reranking_model_name
            }
            chunking_info = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO rag_schema.rag_traces (
                            trace_uuid, user_query_hash, created_at, device,
                            system_prompt, user_prompt, llm_response,
                            retrieved_docs, reranked_docs, timing_info,
                            llm_model_info, embedding_model_info, reranking_model_info,
                            chunking_info)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            _uuid,
                            user_query_hash,
                            timestamp,
                            device,
                            system_prompt,
                            user_prompt,
                            llm_response,
                            json.dumps(retrieved_docs),
                            json.dumps(reranked_docs),
                            json.dumps(timing_info),
                            json.dumps(llm_model_info),
                            json.dumps(embedding_model_info),
                            json.dumps(reranking_model_info),
                            json.dumps(chunking_info)
                        )
                    )
            res = conn.commit()
            print(f"Db insert result: {res}")
            cur.close()
            conn.close()
            return True
        else:
            return False
    except Exception as e:
        conn.rollback()
        print(f"Error inserting trace: {e}")
        return False

def get_data_for_evaluation():
    try:
        conn = establish_db_connection()
        if conn:
            cur = conn.cursor()
            data = cur.execute("""
                SELECT trace_uuid, user_query_hash, created_at, user_prompt, llm_response
                FROM rag_schema.rag_traces
                WHERE trace_uuid not in (
                    SELECT trace_uuid FROM rag_schema.rag_response_evaluations)
                """
            )
            return data.fetchall()
    except Exception as e:
        print(f"Error fetching data for evaluation: {e}")
        return f"Error fetching data for evaluation: {e}"

# -------------------------------
# Run the function
# -------------------------------
if __name__ == "__main__":
    # nothing for now, this is just a module to be imported and used in the main pipeline
    pass
