
import requests
import json
import os

def check_cerebras_models():
    api_key = "csk-wxypx2x3n55wpdvxyrewpx84e4nc8cyy58mpn6jhyyje262y"
    url = "https://api.cerebras.ai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [m['id'] for m in data.get('data', [])]
            print("=== MODELOS DISPONIBLES EN CEREBRAS ===")
            for m in models:
                print(f"- {m}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error fatal: {str(e)}")

if __name__ == "__main__":
    check_cerebras_models()
