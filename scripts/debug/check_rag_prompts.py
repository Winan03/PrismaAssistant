with open("modules/ai/rag_analyzer.py", "r", encoding="utf-8") as f:
    content = f.read()

import re
lines = content.splitlines()
for i, line in enumerate(lines):
    if "prompt" in line.lower() or "instruction" in line.lower() or "generate" in line.lower() or "model" in line.lower() or '"""' in line or "'''" in line:
        safe_line = line.encode('ascii', errors='replace').decode('ascii')
        print(f"Line {i+1}: {safe_line.strip()}")
