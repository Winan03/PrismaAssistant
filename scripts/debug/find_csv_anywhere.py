import os

def search_recursive(path, filename_part):
    print(f"Searching in {path}...")
    for root, dirs, files in os.walk(path):
        # Skip heavy dirs to speed up
        dirs[:] = [d for d in dirs if d not in ["venv", ".git", ".cache", "offload_cache", "chroma_db", "node_modules", ".pytest_cache", "__pycache__"]]
        for file in files:
            if filename_part in file:
                print("FOUND:", os.path.join(root, file))

search_recursive("D:\\9 ciclo", "FINAL_70percent.csv")
