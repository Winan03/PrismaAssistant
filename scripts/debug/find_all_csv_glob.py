import os

workspace = "d:\\9 ciclo\\2025-2\\Tesis 1\\PRODUCTO DE TESIS\\prisma_assistant"
for root, dirs, files in os.walk(workspace):
    # skip venv and .git to be fast
    dirs[:] = [d for d in dirs if d not in ["venv", ".git", ".cache", "offload_cache", "chroma_db", "node_modules"]]
    for file in files:
        if file.endswith(".csv"):
            print("CSV FOUND:", os.path.join(root, file))
