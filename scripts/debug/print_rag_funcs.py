with open("modules/ai/rag_analyzer.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

print("--- Function _query_categories ---")
for i in range(520, 560):
    safe_line = lines[i].encode('ascii', errors='replace').decode('ascii')
    print(f"{i+1}: {safe_line.strip()}")

print("\n--- Function refine_categories ---")
for i in range(680, 725):
    safe_line = lines[i].encode('ascii', errors='replace').decode('ascii')
    print(f"{i+1}: {safe_line.strip()}")
