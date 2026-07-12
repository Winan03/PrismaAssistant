with open("modules/ai/synthesis.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(3129, 3180):
    safe_line = lines[i].encode('ascii', errors='replace').decode('ascii')
    print(f"{i+1}: {safe_line.strip()}")
