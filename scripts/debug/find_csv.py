import os

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if name in file:
                print(os.path.join(root, file))

print("Searching in D:\\9 ciclo:")
find_file("70percent", "D:\\9 ciclo")

print("Searching in C:\\Users\\PC\\.gemini:")
find_file("70percent", "C:\\Users\\PC\\.gemini")
