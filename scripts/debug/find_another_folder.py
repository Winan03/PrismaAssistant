import os

print("Directories in D:")
for item in os.listdir("D:\\"):
    path = os.path.join("D:\\", item)
    if os.path.isdir(path):
        try:
            for sub in os.listdir(path):
                if "Tesis 1" in sub or "PRODUCTO DE TESIS" in sub:
                    print(f"FOUND: {os.path.join(path, sub)}")
                    full_path = os.path.join(path, sub)
                    for sub2 in os.listdir(full_path):
                        print(f"  -> {sub2}")
                        if "prisma_assistant" in sub2:
                            p_assistant = os.path.join(full_path, sub2)
                            print(f"     -> {os.listdir(p_assistant)}")
                            # Check logs inside prisma_assistant
                            logs_p = os.path.join(p_assistant, "logs")
                            if os.path.exists(logs_p):
                                print(f"       -> logs: {os.listdir(logs_p)}")
        except Exception as e:
            pass
