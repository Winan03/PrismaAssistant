import os
import re

directories = ["modules/logic", "utils", "modules/core"]
spanish_patterns = [
    r"Act[úu]a como",
    r"Tu objetivo es",
    r"Genera un",
    r"Responde estrictamente",
    r"REGLAS CR[ÍI]TICAS",
    r"PROHIBIDO",
    r"Solo responde con"
]

found_any = False
for d in directories:
    if not os.path.exists(d):
        continue
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Check for each pattern
                for pattern in spanish_patterns:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    if matches:
                        found_any = True
                        print(f"=== Found pattern '{pattern}' in {path} ===")
                        for m in matches:
                            start_idx = max(0, m.start() - 100)
                            end_idx = min(len(content), m.end() + 200)
                            snippet = content[start_idx:end_idx]
                            safe_snippet = snippet.encode('ascii', errors='replace').decode('ascii')
                            print(f"  Match near character {m.start()}:\n{safe_snippet}\n  ...\n")

if not found_any:
    print("No Spanish prompts found in modules/logic, utils, or modules/core.")
