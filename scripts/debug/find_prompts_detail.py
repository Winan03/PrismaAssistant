import re

files_to_check = [
    "modules/ai/synthesis.py",
    "modules/ai/rag_analyzer.py"
]

for file in files_to_check:
    print(f"=== File: {file} ===")
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Let's find triple quoted strings containing "Actúa como" or "Act?a como"
    matches = re.finditer(r'("""[\s\S]*?Act[úu]a como[\s\S]*?""")', content)
    for m in matches:
        start_idx = m.start()
        end_idx = m.end()
        # Find line numbers
        start_line = content[:start_idx].count('\n') + 1
        end_line = content[:end_idx].count('\n') + 1
        print(f"Match lines {start_line} - {end_line}:")
        snippet = m.group(1)
        safe_snippet = snippet.encode('ascii', errors='replace').decode('ascii')
        first_few_lines = "\n".join(safe_snippet.split("\n")[:4])
        print(first_few_lines + "\n...")
