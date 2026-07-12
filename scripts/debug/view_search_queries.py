with open("modules/core/search_engine.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find all occurrences of "pico" and "semantic_queries" in search_engine.py
lines = content.splitlines()
for i, line in enumerate(lines):
    if "pico" in line.lower() or "semantic_queries" in line.lower() or "expand_" in line.lower():
        safe_line = line.encode('ascii', errors='replace').decode('ascii')
        print(f"Line {i+1}: {safe_line.strip()}")
