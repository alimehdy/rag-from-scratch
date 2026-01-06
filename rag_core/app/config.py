import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
hf_token = os.getenv("hf_token")

if not hf_token:
    raise ValueError("Hugging Face token not found in environment variables.")