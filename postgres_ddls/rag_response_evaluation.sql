CREATE TABLE rag_schema.rag_response_evaluation (
    id bigint CONSTRAINT rag_eval_id_not_null NOT NULL,
    trace_uuid uuid CONSTRAINT rag_eval_trace_uuid_not_null NOT NULL,
    context_precision numeric,
    context_recall numeric,
    faithfulness numeric,
    answer_relevancy numeric
);