import os

logs_dir = "d:\\9 ciclo\\2025-2\\Tesis 1\\PRODUCTO DE TESIS\\prisma_assistant\\logs"
if os.path.exists(logs_dir):
    print("Logs dir exists!")
    print("Files in logs:", os.listdir(logs_dir))
else:
    print("Logs dir does not exist at absolute path!")
