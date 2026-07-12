with open("modules/core/search_engine.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(949, 980):
    safe_line = lines[i].encode('ascii', errors='replace').decode('ascii')
    print(f"{i+1}: {safe_line.strip()}")
