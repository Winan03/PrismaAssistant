import os
import re

directories = ["modules", "utils"]
prompt_regex = re.compile(r'"""[\s\S]*?(?:you are|act as|systematic review|pico|research question|crit[ée]rio|excluy|incluy|synthes|resumen|resuma|s[íi]ntesis|guidelines|instrucciones)[\s\S]*?"""', re.IGNORECASE)

for d in directories:
    if not os.path.exists(d):
        continue
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                matches = list(prompt_regex.finditer(content))
                if matches:
                    print(f"File: {path} (found {len(matches)} potential prompts)")
                    for idx, m in enumerate(matches):
                        snippet = m.group().strip()
                        lines = snippet.splitlines()
                        header = lines[0] if lines else ""
                        body = "\n".join(lines[1:5]) if len(lines) > 1 else ""
                        safe_header = header.encode('ascii', errors='replace').decode('ascii')
                        safe_body = body.encode('ascii', errors='replace').decode('ascii')
                        print(f"  Prompt {idx+1}: {safe_header}\n{safe_body}\n  ...")
                    print("-" * 60)
