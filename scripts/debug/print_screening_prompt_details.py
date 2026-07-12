with open("modules/logic/screening.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

def print_range(start, end, label):
    print(f"=== {label} (Lines {start}-{end}) ===")
    for idx in range(start - 1, min(len(lines), end)):
        safe_line = lines[idx].encode('ascii', errors='replace').decode('ascii')
        print(f"{idx+1}: {safe_line.strip()}")
    print("-" * 60)

# Let's inspect the specific prompt ranges
print_range(68, 85, "Prompt 1: Extract comparison poles")
print_range(805, 835, "Prompt 2: Screening instruction / criteria matching")
print_range(955, 985, "Prompt 3: Criteria classification")
print_range(1050, 1080, "Prompt 4: Screening evaluation")
print_range(1265, 1295, "Prompt 5: Reranker/Arbitrator")
