with open("modules/logic/grok_filter.py", "r", encoding="utf-8") as f:
    content = f.read()

lines = content.splitlines()
for i, line in enumerate(lines):
    if "model" in line.lower() or "generate" in line.lower() or "llm" in line.lower() or "prompt" in line.lower():
        safe_line = line.encode('ascii', errors='replace').decode('ascii')
        print(f"Line {i+1}: {safe_line.strip()}")
