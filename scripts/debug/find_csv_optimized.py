import os

def find_file_opt(name, path, skip_dirs):
    print(f"Searching in {path}...")
    for root, dirs, files in os.walk(path):
        # Modify dirs in-place to prune them
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if name in file:
                print("FOUND:", os.path.join(root, file))

skip = ["venv", ".git", ".cache", "offload_cache", "chroma_db", "node_modules", ".pytest_cache", "__pycache__"]
find_file_opt("70percent", "d:\\9 ciclo\\2025-2\\Tesis 1\\PRODUCTO DE TESIS\\prisma_assistant", skip)
find_file_opt("70percent", "C:\\Users\\PC\\.gemini\\antigravity\\brain\\b8a1ac05-1e29-4073-a1f6-48418c08d0b2", skip)
