import os

def list_level(path, max_depth=3, current_depth=0):
    if current_depth > max_depth:
        return
    indent = "  " * current_depth
    try:
        items = os.listdir(path)
        for item in items:
            full = os.path.join(path, item)
            if os.path.isdir(full):
                # Skip venv/git to be fast
                if item in ["venv", ".git", ".cache", "offload_cache", "chroma_db", "node_modules"]:
                    continue
                print(f"{indent}[DIR] {item}")
                list_level(full, max_depth, current_depth + 1)
            else:
                if "FINAL_70percent.csv" in item or "csv" in item.lower():
                    print(f"{indent}[FILE] {item} -> {full}")
    except Exception as e:
        print(f"{indent}[ERROR] {item}: {e}")

print("Listing D:\\9 ciclo:")
list_level("D:\\9 ciclo")
