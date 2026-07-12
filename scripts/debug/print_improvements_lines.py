with open("modules/logic/screening_improvements.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(160, 220):
    safe_line = lines[i].encode('ascii', errors='replace').decode('ascii')
    print(f"{i+1}: {safe_line.strip()}")
