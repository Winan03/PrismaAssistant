with open("modules/ai/synthesis.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

def print_around(line_num, context_lines=20):
    start = max(0, line_num - 1 - context_lines)
    end = min(len(lines), line_num - 1 + context_lines)
    print(f"--- Around Line {line_num} ---")
    for i in range(start, end):
        safe_line = lines[i].encode('ascii', errors='replace').decode('ascii')
        print(f"{i+1}: {safe_line.strip()}")
    print("\n")

# Let's print around the remaining Spanish prompt lines
print_around(1041, 15)
print_around(3136, 15)
print_around(3150, 15)
print_around(3537, 15)
print_around(4038, 15)
print_around(4412, 15)
