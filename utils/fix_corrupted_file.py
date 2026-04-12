
import os

def fix_synthesis():
    path = r'd:\9 ciclo\2025-2\Tesis 1\PRODUCTO DE TESIS\prisma_assistant\modules\synthesis.py'
    out_path = r'd:\9 ciclo\2025-2\Tesis 1\PRODUCTO DE TESIS\prisma_assistant\modules\synthesis_fixed.py'
    
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return
        
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    print(f"Total lines in corrupted file: {len(lines)}")
    
    # El archivo está duplicado: cada línea de código seguida de una línea vacía
    fixed_lines = []
    # Usamos un step de 2 para saltarnos las líneas de relleno
    for i in range(0, len(lines), 2):
        fixed_lines.append(lines[i])
        
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
        
    print(f"Saved {len(fixed_lines)} lines to {out_path}")

if __name__ == "__main__":
    fix_synthesis()
