import uuid
from hashlib import md5
from datetime import datetime
import torch

def tuple_to_dict(data, column_names):
    # Get column names from the cursor description
    result = []
    for row in data:
        result.append({col: val for col, val in zip(column_names, row)})
    return result
def get_device():
    # Check if CUDA is available, otherwise use CPU
    try:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except:
        print("Defaulting to CPU.")
        return "cpu"
def generate_uuid_and_hash(user_query: str):
    _uuid = str(uuid.uuid4())
    timestamp = datetime.now()
    user_query_hash = str(md5((user_query+timestamp.isoformat()).encode('utf-8')).hexdigest())
    return _uuid, timestamp,user_query_hash

if __name__ == "__main__":
    print(f"Device: {get_device()}")
