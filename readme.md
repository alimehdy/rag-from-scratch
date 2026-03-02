<h1>Required installations</h1>
<h2>Docker & Milvus</h2>

1. Install docker desktop

2. Create a folder named "milvus" inside the project's directory
3. Download the milvus yaml file into the milvus folder using the link below:
https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/deployments/docker/standalone/docker-compose.yml

4. Inside the /milvus folder run:
> docker compose up -d

5. Once done, you'll notice some files added under the /milvus/volumes/. Also a container was created and can be seen in Docker desktop.

6. To make sure milvus is up and running:
> docker ps

7. Each time we need to run milvus:
> cd your-project/milvus
> docker compose up -d

8. We can turn docker down to free the RAM/CPU resources
> docker compose down

9. To check the collections using the UI:
> http://127.0.0.1:9091/webui

<h2>Docker & Postgres</h2>
1. Run the command:
> docker run --name rag_db -e POSTGRES_PASSWORD=A77T9$kL2@vQ7#pZ4!x -p 5431:5432 -d postgres

2. Open an interactive terminal session inside the container and create the table saved in an sql file

> docker exec -it rag_db psql -U postgres -d postgres
> create schema rag_schema;
> CREATE TABLE IF NOT EXISTS rag_schema.rag_traces (id BIGSERIAL PRIMARY KEY, trace_uuid UUID NOT NULL, user_query_hash TEXT, created_at TIMESTAMPTZ DEFAULT NOW(), device TEXT, system_prompt TEXT, user_prompt TEXT, llm_response TEXT, retrieved_docs JSONB, reranked_docs JSONB, timing_info JSONB, llm_model_info JSONB, embedding_model_info JSONB, reranking_model_info JSONB, chunking_info JSONB, success BOOLEAN DEFAULT TRUE, error_message TEXT);

> CREATE TABLE IF NOT EXISTS rag_schema.user_feedbacks (id BIGSERIAL PRIMARY KEY, trace_uuid UUID NOT NULL, user_query_hash TEXT, created_at TIMESTAMPTZ DEFAULT NOW(), feedback_on TEXT, feedback_rating INT, feedback_value TEXT);

> CREATE TABLE IF NOT EXISTS rag_schema.rag_response_evaluation; (id BIGSERIAL PRIMARY KEY, trace_uuid UUID NOT NULL, context_precision DECIMAL, context_recall DECIMAL, faithfulness DECIMAL, answer_relevancy DECIMAL);


3. Set the default schema permanently
> ALTER ROLE postgres SET search_path TO rag_schema;
<!-- Test it -->
> \dt+

4. Check the docker host
> docker ps;
