with open("modules/core/search_engine.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Total lines:", len(lines))
for i, line in enumerate(lines):
    if "citation" in line.lower() or "threshold" in line.lower() or "boost" in line.lower():
        print(f"Line {i+1}: {line.strip()}")
