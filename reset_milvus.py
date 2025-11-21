import os
import config

milvus_file = os.path.abspath(config.MILVUS_URI)
if os.path.exists(milvus_file):
    os.remove(milvus_file)
    print(f"milvus.db eliminado: {milvus_file}")
else:
    print("milvus.db no existe")