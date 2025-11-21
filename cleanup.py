"""
Script de limpieza para eliminar cache viejo y ChromaDB
"""
import shutil
import os
from pathlib import Path

def cleanup():
    """Elimina cache y bases de datos temporales"""
    
    # Eliminar cache
    cache_dir = Path(".cache")
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print("✅ Cache eliminado")
        except Exception as e:
            print(f"⚠️ Error eliminando cache: {e}")
    
    # Eliminar ChromaDB (opcional - resetea la base vectorial)
    chroma_dir = Path("chroma_db")
    if chroma_dir.exists():
        response = input("¿Eliminar ChromaDB también? (s/n): ")
        if response.lower() == 's':
            try:
                shutil.rmtree(chroma_dir)
                print("✅ ChromaDB eliminado")
            except Exception as e:
                print(f"⚠️ Error eliminando ChromaDB: {e}")
    
    print("\n✅ Limpieza completada. Ejecuta 'python main.py' para iniciar con datos frescos.")

if __name__ == "__main__":
    cleanup()