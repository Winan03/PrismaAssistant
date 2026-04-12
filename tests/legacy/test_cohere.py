import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GITHUB_COHERE_TOKEN")
MODEL = "cohere/Cohere-command-r-plus-08-2024"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

data = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Extrae en JSON: {resultado: 'AUC 0.95'} del texto: The model achieved AUC of 0.95"}
    ],
    "temperature": 0.1,
    "max_tokens": 200
}

response = requests.post(
    "https://models.github.ai/inference/chat/completions",
    headers=headers,
    json=data,
    timeout=30
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")