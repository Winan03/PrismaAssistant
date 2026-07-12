import os
import re

logic_dir = "modules/logic"
for root, dirs, files in os.walk(logic_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # find all triple-quoted strings
            matches = re.findall(r'(""\"[\s\S]*?""\"|\'\'\'[\s\S]*?\'\'\')', content)
            print(f"=== File: {path} (found {len(matches)} triple quoted strings) ===")
            for idx, m in enumerate(matches):
                # Check if it has Spanish characters or typical prompt words like "criterios", "pregunta", "artículo", "filtro"
                spanish_words = ["criterio", "pregunta", "artículo", "filtro", "evalúa", "responde", "instrucción"]
                if any(w in m.lower() for w in spanish_words):
                    first_few = "\n".join(m.split("\n")[:4])
                    print(f"  String {idx+1} matches:")
                    # Replace emojis or non-ascii
                    safe_first = first_few.encode('ascii', errors='replace').decode('ascii')
                    print("  " + safe_first + "\n  ...\n")
