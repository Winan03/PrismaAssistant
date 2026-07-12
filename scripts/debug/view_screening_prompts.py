with open("modules/logic/screening.py", "r", encoding="utf-8") as f:
    content = f.read()

import re

# Find all occurrences of double or triple quotes that contain prompt patterns
# or look at lines in screening.py that contain prompts
lines = content.splitlines()
for i, line in enumerate(lines):
    if '"""' in line or "'''" in line or "prompt =" in line or "system_prompt =" in line or "instruction =" in line:
        safe_line = line.encode('ascii', errors='replace').decode('ascii')
        print(f"Line {i+1}: {safe_line.strip()}")
