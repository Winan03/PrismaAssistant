import os

logs_dir = "d:\\9 ciclo\\2025-2\\Tesis 1\\PRODUCTO DE TESIS\\prisma_assistant\\logs"
if os.path.exists(logs_dir):
    print("Logs dir exists!")
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if "csv" in file.lower() or "FINAL" in file or "70percent" in file:
                print("FOUND FILE:", os.path.join(root, file))
else:
    print("Logs dir does not exist!")
