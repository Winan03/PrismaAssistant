import os

def find_file(name, path):
    print(f"Searching in {path}...")
    for root, dirs, files in os.walk(path):
        for file in files:
            if name in file:
                print("FOUND:", os.path.join(root, file))

find_file("70percent", "d:\\9 ciclo\\2025-2\\Tesis 1\\PRODUCTO DE TESIS\\prisma_assistant")
find_file("70percent", "C:\\Users\\PC\\.gemini\\antigravity\\brain\\b8a1ac05-1e29-4073-a1f6-48418c08d0b2")
