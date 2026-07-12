with open("modules/logic/screening.py", "r", encoding="utf-8") as f:
    content = f.read()

import re

# Look for prompts in screening.py and see if they contain spanish words
prompt_patterns = re.findall(r'prompt\s*=\s*(?:f?"""[\s\S]*?"""|f?\'\'\'[\s\S]*?\'\'\'|f?".*?"|f?\'.*?\')', content)
for i, p in enumerate(prompt_patterns):
    # Check if there are any Spanish characters (like á, é, í, ó, ú, ñ) or specific Spanish words
    spanish_words = ["criterio", "inclusión", "exclusión", "analisis", "defecto", "juguete"]
    has_spanish = False
    for word in spanish_words:
        if word in p.lower():
            has_spanish = True
            break
    if has_spanish:
        print(f"Prompt {i+1} has Spanish words:")
        safe_p = p.encode('ascii', errors='replace').decode('ascii')
        print(safe_p[:300])
        print("...")
