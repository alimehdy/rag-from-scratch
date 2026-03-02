CREATE TABLE rag_schema.user_feedbacks (
    id bigint NOT NULL,
    trace_uuid uuid NOT NULL,
    user_query_hash text,
    created_at timestamp with time zone DEFAULT now(),
    feedback_on text,
    feedback_rating integer,
    feedback_value text
)
;