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
