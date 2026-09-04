import httpx
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("CEREBRAS_API_KEY")

r = httpx.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "gpt-oss-120b",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "tools": []},
        timeout=60,
    )  

print(r.json()) 
