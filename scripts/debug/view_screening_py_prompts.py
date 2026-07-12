with open("modules/logic/screening.py", "r", encoding="utf-8") as f:
    content = f.read()

import re

# Find all string literals (triple single/double quotes) in screening.py and print their starting line, ending line, and first 10 lines
matches = list(re.finditer(r'(""\"[\s\S]*?""\"|\'\'\'[\s\S]*?\'\'\')', content))
print("Total triple-quoted strings found:", len(matches))
for i, m in enumerate(matches):
    start_line = content[:m.start()].count('\n') + 1
    end_line = content[:m.end()].count('\n') + 1
    # Check if the string seems to be a prompt
    text = m.group(1)
    if "you are" in text.lower() or "role" in text.lower() or "criterios" in text.lower() or "pregunta" in text.lower() or "select" in text.lower() or "classify" in text.lower() or "systematic" in text.lower() or "instruction" in text.lower():
        print(f"\n--- String {i+1} at lines {start_line}-{end_line} is a candidate prompt: ---")
        first_few = "\n".join(text.split("\n")[:15])
        safe_first = first_few.encode('ascii', errors='replace').decode('ascii')
        print(safe_first)
        print("...")
