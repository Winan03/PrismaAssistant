with open("modules/core/search_engine.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Safely print lines 1220 to 1275
for idx in range(1220, 1275):
    if idx < len(lines):
        line = lines[idx]
        # Replace non-ascii characters to print safely on CP1252
        safe_line = line.encode('ascii', errors='replace').decode('ascii')
        print(f"{idx+1}: {safe_line.strip()}")
