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

# Let's print around some key line numbers
print_around(266, 15)
print_around(312, 15)
print_around(809, 15)
print_around(881, 15)
