import os
import shutil

cache_dir = ".cache"
if os.path.exists(cache_dir):
    print(f"Clearing cache directory: {cache_dir}")
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    print("Cache cleared successfully.")
else:
    print("No cache directory found.")
