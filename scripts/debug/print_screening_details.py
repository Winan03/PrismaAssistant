with open("modules/logic/screening.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

def print_lines(start, end):
    print(f"\n--- Lines {start}-{end} ---")
    for idx in range(start - 1, end):
        if idx < len(lines):
            safe_line = lines[idx].encode('ascii', errors='replace').decode('ascii')
            print(f"{idx+1}: {safe_line.strip()}")

print_lines(65, 90)
print_lines(670, 695)
print_lines(805, 845)
print_lines(1045, 1120)
print_lines(1265, 1300)
